"""Optimization lifecycle — spec Section 6.6.

The agent proposes; the stopwatch disposes. Everything consequential
here is measured, not reported:

  * `baseline_ms` / `optimized_ms` come from real before/after benchmark
    runs against the SAME entry point and the same fixed input;
  * `improvement_pct` is computed from those two numbers;
  * a change below `optimizer.min_improvement_pct` is REVERTED, per
    6.6's explicit instruction to hand back `no_improvement` "rather
    than shipping a marginal or noisy-benchmark change";
  * a change that breaks the test suite is reverted regardless of how
    fast it is.

That last one is an addition to 6.6's literal text, and a deliberate
one: the spec's threshold is purely about speed, but "faster" is a
worthless verdict if behavior changed. The cheapest way to make code
fast is to make it wrong, and an agent optimizing against a timer has
every incentive to find that out. So correctness is checked first and
gates the speed question entirely.
"""

import asyncio
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from amop.agents.handoffs import OptimizationReport
from amop.agents.optimizer import OptimizerAgent
from amop.database.models import Task
from amop.orchestrator.concurrency import gated_by_task_slot
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import transition
from amop.sandbox import repo as git_repo
from amop.sandbox import tools as sandbox_tools  # noqa: F401 -- registers the sandboxed tools
from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext, invoke_tool

# Section 6.6: optimizer.min_improvement_pct, default 10%.
DEFAULT_MIN_IMPROVEMENT_PCT = 10.0
MIN_IMPROVEMENT_PCT = float(
    os.environ.get("AMOP_MIN_IMPROVEMENT_PCT", str(DEFAULT_MIN_IMPROVEMENT_PCT))
)

# Section 6.6: suggestor.
DEFAULT_PERMISSION_MODE = "suggestor"

BENCHMARK_ITERATIONS = 5


@dataclass
class OptimizationResult:
    task: Task | None
    report: OptimizationReport
    tool_calls: list[dict] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    diff: str = ""


def _noop(_message: str) -> None:
    pass


def improvement_pct(baseline_ms: float, optimized_ms: float) -> float:
    """Percent faster, positive = better.

    Guards a zero baseline explicitly: a benchmark that measured 0ms
    means the harness failed, not that the code is infinitely fast, and
    dividing by it would produce an "improvement" out of nothing.
    """
    if baseline_ms <= 0:
        return 0.0
    return ((baseline_ms - optimized_ms) / baseline_ms) * 100.0


async def _benchmark(ctx: ToolContext, entry_point: str) -> float | None:
    result = await invoke_tool(
        "run_benchmark",
        {"path": entry_point, "iterations": BENCHMARK_ITERATIONS},
        ctx,
        agent_name="orchestrator",
    )
    if not result.success or not isinstance(result.output, dict):
        return None
    return float(result.output.get("median_ms", 0.0))


@gated_by_task_slot
async def run_optimization(
    session: AsyncSession,
    task: Task,
    *,
    repo_path: Path,
    model,
    entry_point: str = "benchmark.py",
    scratch_root: Path | None = None,
    mode: str = DEFAULT_PERMISSION_MODE,
    min_improvement_pct: float = MIN_IMPROVEMENT_PCT,
    emit: Callable[[str], None] = _noop,
) -> OptimizationResult:
    scratch_root = Path(scratch_root or sandbox_tools.SCRATCH_DIR)
    scratch_dir = (scratch_root / str(task.id)).resolve()
    task_id = str(task.id)

    await asyncio.to_thread(git_repo.materialize, Path(repo_path), scratch_dir)
    manager = await asyncio.to_thread(SandboxManager)
    sandbox = await asyncio.to_thread(manager.create, task_id, scratch_dir)

    stages: list[str] = []

    def stage(message: str) -> None:
        stages.append(message)
        emit(message)

    def _fail(report: OptimizationReport) -> OptimizationResult:
        return OptimizationResult(task=task, report=report, stages=stages)

    try:
        await asyncio.to_thread(git_repo.init_baseline, sandbox)
        await asyncio.to_thread(
            git_repo.create_branch, sandbox, f"amop/opt-{uuid.UUID(task_id).hex[:8]}"
        )
        ctx = ToolContext(
            agent_name="optimizer",
            scratch_dir=scratch_dir,
            mode=mode,
            sandbox=sandbox,
            repo_path=str(Path(repo_path).resolve()),
            db_session=session,
        )

        # --- baseline, before the agent touches anything ---------------
        baseline_ms = await _benchmark(ctx, entry_point)
        if baseline_ms is None:
            stage(f"NO_IMPROVEMENT: could not benchmark {entry_point} -- nothing to compare against")
            return _fail(
                OptimizationReport(
                    task_id=task_id,
                    status="no_improvement",
                    diagnostic=f"baseline benchmark of {entry_point} failed",
                )
            )
        stage(f"BASELINE: {baseline_ms:.1f} ms (median of {BENCHMARK_ITERATIONS})")

        agent = OptimizerAgent(model, ctx)
        agent_result = await agent.run(
            f"The benchmark entry point is {entry_point}. Profile it, find "
            "the real bottleneck, and make it faster without changing "
            "behavior."
        )
        tool_calls = [{"agent": agent.name, **c} for c in agent_result.tool_calls]

        reported = (
            agent_result.handoff
            if isinstance(agent_result.handoff, OptimizationReport)
            else OptimizationReport(task_id=task_id)
        )

        changed = await asyncio.to_thread(git_repo.changed_files, sandbox)
        stage(f"OPTIMIZING: files changed {changed or '[]'}")

        def _revert_report(diagnostic: str) -> OptimizationReport:
            return reported.model_copy(
                update={
                    "task_id": task_id,
                    "status": "no_improvement",
                    "baseline_ms": round(baseline_ms, 3),
                    "optimized_ms": 0.0,
                    "improvement_pct": 0.0,
                    "files_changed": changed,
                    "reverted": True,
                    "diagnostic": diagnostic,
                }
            )

        if not changed:
            stage("NO_IMPROVEMENT: the optimizer changed nothing")
            report = reported.model_copy(
                update={
                    "task_id": task_id,
                    "status": "no_improvement",
                    "baseline_ms": round(baseline_ms, 3),
                    "optimized_ms": round(baseline_ms, 3),
                    "improvement_pct": 0.0,
                    "files_changed": [],
                    "reverted": False,
                    "diagnostic": "no change was made",
                }
            )
            await _land(session, task, report, emit)
            return OptimizationResult(
                task=task, report=report, tool_calls=tool_calls, stages=stages
            )

        # --- correctness gates speed (see module docstring) ------------
        stage("TESTING: confirming behavior is unchanged")
        test_result = await invoke_tool("run_tests", {}, ctx, agent_name="orchestrator")
        tests_passed = bool(
            test_result.success
            and isinstance(test_result.output, dict)
            and test_result.output.get("all_passed")
        )
        stage(f"TESTING: all_passed={tests_passed}")
        if not tests_passed:
            await asyncio.to_thread(git_repo.revert_to_baseline, sandbox)
            stage("NO_IMPROVEMENT: behavior changed (suite red) -- reverted")
            report = _revert_report("the test suite failed after the optimization")
            await _land(session, task, report, emit)
            return OptimizationResult(
                task=task, report=report, tool_calls=tool_calls, stages=stages
            )

        # --- the measurement that decides everything -------------------
        optimized_ms = await _benchmark(ctx, entry_point)
        if optimized_ms is None:
            await asyncio.to_thread(git_repo.revert_to_baseline, sandbox)
            stage("NO_IMPROVEMENT: post-change benchmark failed -- reverted")
            report = _revert_report("post-change benchmark failed to run")
            await _land(session, task, report, emit)
            return OptimizationResult(
                task=task, report=report, tool_calls=tool_calls, stages=stages
            )

        gain = improvement_pct(baseline_ms, optimized_ms)
        stage(
            f"MEASURED: {baseline_ms:.1f} ms -> {optimized_ms:.1f} ms "
            f"({gain:+.1f}%, threshold {min_improvement_pct:.0f}%)"
        )

        if gain < min_improvement_pct:
            await asyncio.to_thread(git_repo.revert_to_baseline, sandbox)
            stage(
                f"NO_IMPROVEMENT: {gain:+.1f}% is below the "
                f"{min_improvement_pct:.0f}% threshold -- reverted"
            )
            report = reported.model_copy(
                update={
                    "task_id": task_id,
                    "status": "no_improvement",
                    "baseline_ms": round(baseline_ms, 3),
                    "optimized_ms": round(optimized_ms, 3),
                    "improvement_pct": round(gain, 2),
                    "files_changed": changed,
                    "reverted": True,
                    "diagnostic": (
                        f"measured {gain:+.1f}%, below the "
                        f"{min_improvement_pct:.0f}% minimum -- not worth a diff"
                    ),
                }
            )
            await _land(session, task, report, emit)
            return OptimizationResult(
                task=task, report=report, tool_calls=tool_calls, stages=stages
            )

        diff = await asyncio.to_thread(git_repo.diff_against_baseline, sandbox)
        report = reported.model_copy(
            update={
                "task_id": task_id,
                "status": "improved",
                "baseline_ms": round(baseline_ms, 3),
                "optimized_ms": round(optimized_ms, 3),
                "improvement_pct": round(gain, 2),
                "files_changed": changed,
                "reverted": False,
                "diagnostic": None,
            }
        )
        stage(f"IMPROVED: {gain:.1f}% faster, kept")
        await _land(session, task, report, emit)
        return OptimizationResult(
            task=task, report=report, tool_calls=tool_calls, stages=stages, diff=diff
        )
    finally:
        await asyncio.to_thread(manager.destroy, task_id)


async def _land(
    session: AsyncSession,
    task: Task,
    report: OptimizationReport,
    emit: Callable[[str], None],
) -> None:
    """Record the outcome on the borrowed bug_fix machine.

    Same borrowing as orchestrator/deps.py, and the same caveat: there
    is no OPTIMIZING state, so an improvement walks the normal
    CODING -> TESTING -> REVIEWING -> PR_CREATION -> WAITING_FOR_APPROVAL
    path and a no_improvement lands at NEEDS_HUMAN_INPUT. Written out
    explicitly rather than hidden, so the borrowed path stays visible.
    """
    improved = report.status == "improved"
    path: list[TaskState] = []
    if TaskState(task.state) is TaskState.CREATED:
        path += [
            TaskState.TRIAGING,
            TaskState.INVESTIGATING,
            TaskState.PLANNING_FIX,
            TaskState.CODING,
        ]
    path += (
        [
            TaskState.TESTING,
            TaskState.REVIEWING,
            TaskState.PR_CREATION,
            TaskState.WAITING_FOR_APPROVAL,
        ]
        if improved
        else [TaskState.NEEDS_HUMAN_INPUT]
    )
    try:
        for state in path:
            await transition(
                session,
                task,
                state,
                actor="agent:optimizer",
                trigger=f"optimization {report.status}",
            )
    except Exception as exc:  # noqa: BLE001 -- a bad landing must not lose the report
        emit(f"warning: could not record terminal state ({exc})")
