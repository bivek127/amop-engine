"""The benchmark harness — spec Section 20.2/20.3.

Runs each scenario through the same orchestrator entry point the CLI
uses (nothing is special-cased for benchmarking), then scores the result
against ground truth: the task's terminal state from Postgres, the
`agent_actions` audit rows, and a real re-run of the fixture's own test
suite against whatever code the chain actually produced.

**Scoped to this run's own task IDs.** The database holds 27 milestones
of history, including the stranded non-terminal tasks logged as debt at
Milestone 16. A metric computed over all of `tasks` would silently fold
that in and quietly differ from run to run. Every query here filters to
the IDs this run created.

**Re-running the fixture's suite is the point.** A chain reports its own
success; this checks it. The workspace the chain left behind is mounted
into a fresh sandbox and the fixture's tests are run again, so a
"RESOLVED" task whose tests still fail scores as a failure. That is the
benchmarking form of the rule Milestone 14's Reporter established and
Milestone 26 had to re-establish the hard way.
"""

import time
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from amop.audit.actions import MODEL_CALL_TOOL_NAME
from amop.codebase_intel.indexer import detect_stack
from amop.database.models import AgentAction, Task, TaskTransition
from amop.evaluation.metrics import (
    SUCCESS_TRANSITION,
    Report,
    TaskOutcome,
    compute_all,
)
from amop.evaluation.scenarios import (
    BUG_FIX,
    DEPENDENCY_UPDATE,
    OPTIMIZATION,
    Scenario,
)
from amop.orchestrator.task import create_task
from amop.sandbox import tools as sandbox_tools
from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext


def _noop(message: str) -> None:
    pass


async def _run_one(
    session: AsyncSession,
    scenario: Scenario,
    *,
    model,
    emit: Callable[[str], None],
) -> tuple[str, str | None]:
    """Run one scenario. Returns (task_id, error).

    Imports are local because the three orchestrators pull in the whole
    agent/sandbox stack; importing them at module scope would make even
    `amop evaluate --help` pay for it.
    """
    task = await create_task(
        session,
        task_type=scenario.task_type,
        task_context={"prompt": scenario.description, "benchmark": scenario.name},
    )
    task_id = str(task.id)

    try:
        if scenario.task_type == BUG_FIX:
            from amop.orchestrator.chain import run_fix

            await run_fix(
                session,
                task,
                description=scenario.description,
                repo_path=scenario.repo_path,
                model=model,
                emit=emit,
            )
        elif scenario.task_type == OPTIMIZATION:
            from amop.orchestrator.optimize import run_optimization

            await run_optimization(
                session,
                task,
                repo_path=scenario.repo_path,
                model=model,
                entry_point=scenario.entry_point or "benchmark.py",
                emit=emit,
            )
        elif scenario.task_type == DEPENDENCY_UPDATE:
            from amop.orchestrator.deps import run_dependency_update

            await run_dependency_update(
                session,
                task,
                repo_path=scenario.repo_path,
                model=model,
                manifest=scenario.manifest or "requirements.txt",
                emit=emit,
            )
        else:
            return task_id, f"unknown task_type {scenario.task_type!r}"
    except Exception as exc:  # noqa: BLE001
        # A crashed scenario is a real result, not a reason to abandon
        # the suite -- the remaining scenarios still carry information,
        # and a run that dies halfway produces no comparison at all.
        return task_id, f"{type(exc).__name__}: {exc}"

    return task_id, None


async def _verify_tests(scenario: Scenario, task_id: str) -> bool | None:
    """Re-run the fixture's own suite against the code the chain
    produced. None if it couldn't be run at all.

    Deliberately independent of anything the chain reported: a fresh
    sandbox over the workspace it left behind, using the same
    `run_tests` tool and the same stack detection the chain itself uses,
    so the check can't drift from how tests are really run.
    """
    if scenario.task_type != BUG_FIX:
        # Optimization/dependency runs are scored by their own
        # orchestrator's enforced checks (measured improvement, blast
        # radius + test verdict), not by this re-run.
        return None

    workspace = Path(sandbox_tools.SCRATCH_DIR) / task_id
    if not workspace.is_dir():
        return None

    verify_id = f"eval-verify-{task_id[:8]}"
    manager = None
    try:
        # SandboxManager() is INSIDE the try because constructing it
        # talks to the Docker daemon, and that can fail. Found the hard
        # way: Docker Desktop died mid-benchmark under memory pressure,
        # and because this construction sat outside the try, one
        # unverifiable scenario took down an entire multi-hour run
        # instead of being recorded as tests_pass=None and moving on.
        stack = detect_stack(workspace).get("primary") or "python"
        manager = SandboxManager()
        sandbox = manager.create(verify_id, workspace, stack=stack)
        ctx = ToolContext(
            agent_name="evaluation",
            scratch_dir=workspace,
            mode="observer",
            sandbox=sandbox,
            stack=stack,
        )
        from amop.sandbox.tools import run_tests

        result = await run_tests(ctx)
    except Exception:  # noqa: BLE001
        # An unverifiable scenario is recorded as tests_pass=None, which
        # fix_success_rate already excludes from its denominator -- so a
        # verification outage lowers confidence in the run rather than
        # silently scoring the model as having failed.
        return None
    finally:
        if manager is not None:
            manager.destroy(verify_id)

    if not result.success or not result.output:
        return None
    return bool(result.output.get("all_passed"))


async def _ground_truth(
    session: AsyncSession,
    scenario: Scenario,
    task_id: str,
    error: str | None,
    duration_seconds: float | None = None,
) -> TaskOutcome:
    """Assemble one scenario's outcome from the database plus the
    independent test re-run. Nothing here reads a chain's own report."""
    task = (
        await session.execute(select(Task).where(Task.id == task_id))
    ).scalar_one_or_none()

    counts = (
        await session.execute(
            select(
                func.count(AgentAction.id).filter(
                    AgentAction.tool_name != MODEL_CALL_TOOL_NAME
                ),
                func.count(AgentAction.id).filter(
                    AgentAction.tool_name == MODEL_CALL_TOOL_NAME
                ),
                func.coalesce(func.sum(AgentAction.input_tokens), 0),
                func.coalesce(func.sum(AgentAction.output_tokens), 0),
            ).where(AgentAction.task_id == task_id)
        )
    ).one()

    # Whether the Reviewer ever approved, read from the hash-chained
    # transition trail rather than inferred from the terminal state --
    # see metrics.SUCCESS_TRANSITION for why the terminal state can't
    # carry this.
    reached_pr = (
        await session.execute(
            select(func.count(TaskTransition.id)).where(
                TaskTransition.task_id == task_id,
                TaskTransition.to_state == SUCCESS_TRANSITION,
            )
        )
    ).scalar_one() > 0

    tests_pass = await _verify_tests(scenario, task_id)

    return TaskOutcome(
        task_id=task_id,
        scenario=scenario.name,
        task_type=scenario.task_type,
        state=task.state if task else "MISSING",
        created_at=task.created_at if task else None,
        resolved_at=task.resolved_at if task else None,
        tool_calls=int(counts[0] or 0),
        model_calls=int(counts[1] or 0),
        input_tokens=int(counts[2] or 0),
        output_tokens=int(counts[3] or 0),
        tests_pass=tests_pass,
        error=error,
        reached_pr_creation=reached_pr,
        duration_seconds=duration_seconds,
    )


async def run_suite(
    session: AsyncSession,
    scenarios: list[Scenario],
    *,
    model,
    model_name: str,
    repeat: int = 1,
    emit: Callable[[str], None] = _noop,
) -> Report:
    """Run every scenario `repeat` times, score each, compute the report.

    `model_name` is carried separately from `model` because the report
    labels a comparison, and the object itself doesn't reliably name
    what it is (a fallback may substitute a different one mid-run --
    Section 11.2's "intentional visibility").

    `repeat` exists because a single pass is not a measurement. Observed
    directly: the same model on the same fixture passed one run and
    failed the next. At n=1 a two-model comparison can be entirely
    noise, so the report shows the spread across repetitions rather than
    one sample dressed up as a result.
    """
    outcomes: list[TaskOutcome] = []
    total = len(scenarios) * repeat
    step = 0
    for scenario in scenarios:
        for attempt in range(1, repeat + 1):
            step += 1
            suffix = f" run {attempt}/{repeat}" if repeat > 1 else ""
            emit(f"[{step}/{total}] {scenario.name} ({scenario.fixture}){suffix}")
            started = time.monotonic()
            task_id, error = await _run_one(
                session, scenario, model=model, emit=_noop
            )
            elapsed = time.monotonic() - started
            outcome = await _ground_truth(
                session, scenario, task_id, error, duration_seconds=elapsed
            )
            outcomes.append(outcome)
            verdict = (
                "PASS" if outcome.succeeded and outcome.tests_pass is not False
                else "FAIL"
            )
            emit(
                f"    -> {verdict}  state={outcome.state}  "
                f"tests={outcome.tests_pass}  tokens={outcome.total_tokens}  "
                f"({elapsed:.0f}s)"
            )

    return compute_all(model_name, outcomes)
