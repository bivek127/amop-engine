"""Dependency update lifecycle — spec Section 6.7.

Everything in this module exists because DependencyUpdater is the one
agent spec'd to run `autonomous` by default. Its two consequential
decisions are therefore made here, in code, from the repository's own
state:

  * blast radius -- the changed-file count comes from git, and exceeding
    `max_files_for_auto_fix` reverts the work outright rather than
    trusting the agent's sense of how big its own change was;
  * whether it worked -- `tests_passed` comes from the orchestrator's own
    full-suite run, not the agent's report.

Both mirror chain.py's existing treatment of the Coder (files read back
from git, test results overwritten from real pytest output). The agent
supplies judgment about *which* version to move to; software supplies
the verdict on whether that was allowed to stand.
"""

import asyncio
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from amop.agents.dependency_updater import DependencyUpdaterAgent
from amop.agents.handoffs import DependencyUpdateReport
from amop.database.models import Task
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import transition
from amop.sandbox import repo as git_repo
from amop.sandbox import tools as sandbox_tools  # noqa: F401 -- registers the sandboxed tools
from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext, invoke_tool

# Section 6.7: dependency_updater.max_files_for_auto_fix, default 3.
DEFAULT_MAX_FILES_FOR_AUTO_FIX = 3
MAX_FILES_FOR_AUTO_FIX = int(
    os.environ.get(
        "AMOP_MAX_FILES_FOR_AUTO_FIX", str(DEFAULT_MAX_FILES_FOR_AUTO_FIX)
    )
)

# Manifests are the thing being bumped, so they don't count against the
# budget -- 6.7's cap is on "the fix", the source changes a new version
# forces, not on the bump that triggered them. Counting the manifest
# would silently reduce a documented budget of 3 to an actual 2.
MANIFEST_FILENAMES = frozenset(
    {"requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.cfg"}
)

# Section 6.7's default permission -- the only agent that gets it.
DEFAULT_PERMISSION_MODE = "autonomous"


def source_files_changed(changed: list[str]) -> list[str]:
    """The changed files that count against the auto-fix budget."""
    return [p for p in changed if Path(p).name not in MANIFEST_FILENAMES]


@dataclass
class DependencyUpdateResult:
    task: Task | None
    report: DependencyUpdateReport
    changed_files: list[str] = field(default_factory=list)
    reverted: bool = False
    diff: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)


def _noop(_message: str) -> None:
    pass


async def run_dependency_update(
    session: AsyncSession,
    task: Task,
    *,
    repo_path: Path,
    model,
    manifest: str = "requirements.txt",
    scratch_root: Path | None = None,
    mode: str = DEFAULT_PERMISSION_MODE,
    emit: Callable[[str], None] = _noop,
) -> DependencyUpdateResult:
    """Run one dependency update end to end, with the blast-radius cap
    and the test verdict both enforced outside the agent."""
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

    try:
        await asyncio.to_thread(git_repo.init_baseline, sandbox)
        await asyncio.to_thread(
            git_repo.create_branch, sandbox, f"amop/deps-{uuid.UUID(task_id).hex[:8]}"
        )

        ctx = ToolContext(
            agent_name="dependency_updater",
            scratch_dir=scratch_dir,
            mode=mode,
            sandbox=sandbox,
            repo_path=str(Path(repo_path).resolve()),
            db_session=session,
        )

        agent = DependencyUpdaterAgent(model, ctx)
        stage(f"UPDATING: checking {manifest} for advisories")
        agent_result = await agent.run(
            f"The manifest to check and update is {manifest}. "
            "Find any vulnerable pinned dependencies, bump them to a "
            "fixed version, and make the full test suite pass."
        )

        tool_calls = [
            {"agent": agent.name, **call} for call in agent_result.tool_calls
        ]

        # Ground truth #1: what actually changed, from git.
        changed = await asyncio.to_thread(git_repo.changed_files, sandbox)
        source_changed = source_files_changed(changed)
        stage(f"UPDATING: files changed {changed or '[]'}")

        reported = (
            agent_result.handoff
            if isinstance(agent_result.handoff, DependencyUpdateReport)
            else DependencyUpdateReport(task_id=task_id)
        )

        # Section 6.7's escalation rule, enforced mechanically.
        if len(source_changed) > MAX_FILES_FOR_AUTO_FIX:
            await asyncio.to_thread(git_repo.revert_to_baseline, sandbox)
            stage(
                f"NEEDS_MANUAL_REVIEW: the update needed source changes across "
                f"{len(source_changed)} files (cap is {MAX_FILES_FOR_AUTO_FIX}) "
                "-- reverted, this is a migration rather than a bump"
            )
            report = reported.model_copy(
                update={
                    "task_id": task_id,
                    "status": "needs_manual_review",
                    "tests_passed": False,
                    "files_changed": changed,
                    "diagnostic": (
                        f"would touch {len(source_changed)} source files "
                        f"({', '.join(source_changed)}), over the "
                        f"max_files_for_auto_fix cap of {MAX_FILES_FOR_AUTO_FIX}"
                    ),
                }
            )
            await _land(session, task, report, emit)
            return DependencyUpdateResult(
                task=task,
                report=report,
                changed_files=changed,
                reverted=True,
                tool_calls=tool_calls,
                stages=stages,
            )

        if not changed:
            stage("NEEDS_MANUAL_REVIEW: nothing was changed")
            report = reported.model_copy(
                update={
                    "task_id": task_id,
                    "status": "needs_manual_review",
                    "tests_passed": False,
                    "files_changed": [],
                    "diagnostic": reported.diagnostic
                    or "the agent made no change to the manifest",
                }
            )
            await _land(session, task, report, emit)
            return DependencyUpdateResult(
                task=task,
                report=report,
                changed_files=[],
                tool_calls=tool_calls,
                stages=stages,
            )

        # Ground truth #2: the orchestrator runs the full suite itself.
        # Section 6.7 is explicit that it is never scoped, because a
        # dependency change's blast radius is unpredictable.
        stage("TESTING: running the full suite (never scoped, per 6.7)")
        test_result = await invoke_tool("run_tests", {}, ctx, agent_name="orchestrator")
        tests_passed = bool(
            test_result.success
            and isinstance(test_result.output, dict)
            and test_result.output.get("all_passed")
        )
        stage(f"TESTING: all_passed={tests_passed}")

        if not tests_passed:
            await asyncio.to_thread(git_repo.revert_to_baseline, sandbox)
            stage("NEEDS_MANUAL_REVIEW: tests fail after the bump -- reverted")

        diff = (
            ""
            if not tests_passed
            else await asyncio.to_thread(git_repo.diff_against_baseline, sandbox)
        )
        report = reported.model_copy(
            update={
                "task_id": task_id,
                "status": "success" if tests_passed else "needs_manual_review",
                "tests_passed": tests_passed,
                "files_changed": changed,
                "diagnostic": (
                    None
                    if tests_passed
                    else "the full test suite failed after the version bump"
                ),
            }
        )
        await _land(session, task, report, emit)
        return DependencyUpdateResult(
            task=task,
            report=report,
            changed_files=changed,
            reverted=not tests_passed,
            diff=diff,
            tool_calls=tool_calls,
            stages=stages,
        )
    finally:
        await asyncio.to_thread(manager.destroy, task_id)


async def _land(
    session: AsyncSession,
    task: Task,
    report: DependencyUpdateReport,
    emit: Callable[[str], None],
) -> None:
    """Move the task to a terminal state matching the real outcome.

    Reuses the bug_fix machine rather than adding new states: CLAUDE.md's
    Milestone 14 scope adds no TaskState members, and its existing states
    already carry the right meanings -- WAITING_FOR_APPROVAL is "a human
    must look at this diff", NEEDS_HUMAN_INPUT is "this needs a person,
    it isn't mechanically finishable".

    The intermediate hops are walked explicitly because the machine has
    no shortcut edge: CODING -> WAITING_FOR_APPROVAL is illegal (verified
    against TRANSITIONS, not assumed), so a success has to pass through
    TESTING -> REVIEWING -> PR_CREATION the same way a bug_fix does. That
    is a real cost of not having a dependency_update machine of its own,
    and it's written out here rather than hidden behind a helper so the
    borrowed path is obvious to the next reader.
    """
    success = report.status == "success"
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
        if success
        else [TaskState.NEEDS_HUMAN_INPUT]
    )

    try:
        for state in path:
            await transition(
                session,
                task,
                state,
                actor="agent:dependency_updater",
                trigger=f"dependency update {report.status}",
            )
    except Exception as exc:  # noqa: BLE001 -- a bad landing must not lose the report
        emit(f"warning: could not record terminal state ({exc})")
