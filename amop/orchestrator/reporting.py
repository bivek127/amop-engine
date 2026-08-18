"""Reporting window queries — spec Section 6.8's inputs.

Deliberately a separate module from agents/reporter.py, and deliberately
containing all the SQL: these are the numbers Reporter is NOT allowed to
decide. Keeping them here, computed from `tasks` and `task_transitions`
with no model in the loop, is what makes the "a model may describe the
window; it may not decide the counts" rule enforceable rather than
aspirational.

Section 6.8 names `agent_actions` as an input alongside `tasks`. That
table doesn't exist in this codebase -- Section 14.2 defines one, no
milestone has built it. `task_transitions` (Milestone 1's append-only
state-change trail) turns out to carry everything the ReportSummary
schema needs, because every fact in that schema is really a statement
about state changes: a PR opened IS a transition into
WAITING_FOR_APPROVAL. So the substitution is not a downgrade for this
particular schema, and it's recorded here rather than silently swapped.
"""

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from amop.agents.handoffs import ReportSummary
from amop.agents.reporter import ReporterAgent
from amop.database.models import Task, TaskTransition
from amop.orchestrator.state_machine import TERMINAL_STATES, TaskState
from amop.sandbox import tools as sandbox_tools  # noqa: F401 -- registers the sandboxed tools
from amop.tools.registry import ToolContext

# Task.task_type values that count as a dependency update for 6.8's
# `dependencies_updated`. A tuple rather than a bare string so Stage 4's
# DependencyUpdater can be wired in without changing the query.
DEPENDENCY_TASK_TYPES = ("dependency_update",)

# How many of the window's tasks to actually show the model. A reporting
# window is unbounded in principle (--since 2020-01-01), and a prompt
# that grows without limit is Milestone 10's context-overflow bug
# arriving by a different route.
MAX_TASKS_IN_PROMPT = 40


@dataclass
class TaskDigest:
    """One task, reduced to what a summarizer needs."""

    task_id: str
    task_type: str
    state: str
    created_at: datetime | None
    prompt: str


@dataclass
class ReportWindow:
    """Everything Reporter is given, and everything it may not change.

    The counts here are the report. `ReportSummary`'s equivalent fields
    are overwritten from this after the model returns.
    """

    period_start: datetime
    period_end: datetime
    tasks_resolved: int = 0
    prs_opened: int = 0
    prs_merged: int = 0
    dependencies_updated: int = 0
    tasks_created: int = 0
    tasks_failed: int = 0
    digests: list[TaskDigest] = field(default_factory=list)


async def _count_transitions_into(
    session: AsyncSession, state: TaskState, start: datetime, end: datetime
) -> int:
    """Transitions INTO a state within the window.

    Counts transitions, not tasks in that state now: a PR opened during
    the window is a real event even if the task has since moved on, and
    a task sitting in WAITING_FOR_APPROVAL from before the window did not
    have its PR opened during it. `tasks.state` alone answers neither.
    """
    stmt = (
        select(func.count())
        .select_from(TaskTransition)
        .where(
            TaskTransition.to_state == state.value,
            TaskTransition.timestamp >= start,
            TaskTransition.timestamp <= end,
        )
    )
    return int((await session.execute(stmt)).scalar_one())


async def collect_report_window(
    session: AsyncSession, start: datetime, end: datetime
) -> ReportWindow:
    """Compute Section 6.8's reporting window from real rows only."""
    window = ReportWindow(period_start=start, period_end=end)

    # "Resolved" here means "reached a terminal state during the window",
    # which is the honest reading given that RESOLVED itself is currently
    # unreachable (see chain.py's MEMORY_WRITE_STATES for the same
    # problem and the same reasoning). WAITING_FOR_APPROVAL is NOT
    # counted as resolved -- a PR awaiting a human is precisely not
    # finished, and it already has its own count below.
    for state in TERMINAL_STATES:
        window.tasks_resolved += await _count_transitions_into(
            session, state, start, end
        )

    window.tasks_failed = await _count_transitions_into(
        session, TaskState.FAILED, start, end
    )
    window.prs_opened = await _count_transitions_into(
        session, TaskState.WAITING_FOR_APPROVAL, start, end
    )
    # Always 0 today: nothing in this codebase reaches MERGED (no
    # auto-merge, no post-merge detection). Queried anyway rather than
    # hardcoded, so it starts reporting real numbers the day that path
    # exists without anyone remembering to come back here.
    window.prs_merged = await _count_transitions_into(
        session, TaskState.MERGED, start, end
    )

    created_stmt = (
        select(func.count())
        .select_from(Task)
        .where(Task.created_at >= start, Task.created_at <= end)
    )
    window.tasks_created = int((await session.execute(created_stmt)).scalar_one())

    deps_stmt = (
        select(func.count())
        .select_from(Task)
        .where(
            Task.task_type.in_(DEPENDENCY_TASK_TYPES),
            Task.created_at >= start,
            Task.created_at <= end,
        )
    )
    window.dependencies_updated = int((await session.execute(deps_stmt)).scalar_one())

    digest_stmt = (
        select(Task)
        .where(Task.created_at >= start, Task.created_at <= end)
        .order_by(Task.created_at.desc())
        .limit(MAX_TASKS_IN_PROMPT)
    )
    tasks = (await session.execute(digest_stmt)).scalars().all()
    window.digests = [
        TaskDigest(
            task_id=str(t.id),
            task_type=t.task_type,
            state=t.state,
            created_at=t.created_at,
            prompt=str((t.task_context or {}).get("prompt", ""))[:300],
        )
        for t in tasks
    ]
    return window


def render_window_for_prompt(window: ReportWindow) -> str:
    """The facts, laid out for a summarizer.

    The counts are shown so the model can write prose consistent with
    them -- not so it can compute them. Whatever it echoes back is
    discarded in favor of these same numbers (see ReportSummary's
    docstring).
    """
    lines = [
        f"Reporting window: {window.period_start.isoformat()} "
        f"to {window.period_end.isoformat()}",
        "",
        "Verified counts (computed from the database -- treat as fact):",
        f"  tasks created in window : {window.tasks_created}",
        f"  tasks reaching a terminal state : {window.tasks_resolved}",
        f"  of those, failed : {window.tasks_failed}",
        f"  pull requests opened : {window.prs_opened}",
        f"  pull requests merged : {window.prs_merged}",
        f"  dependency updates : {window.dependencies_updated}",
        "",
    ]
    if not window.digests:
        lines.append("No tasks were created in this window.")
        return "\n".join(lines)

    lines.append(f"Tasks in this window (up to {MAX_TASKS_IN_PROMPT} shown):")
    for d in window.digests:
        when = d.created_at.isoformat() if d.created_at else "(unknown)"
        lines.append(f"  - [{d.state}] {d.task_type} @ {when}")
        if d.prompt:
            lines.append(f"      {d.prompt}")
    return "\n".join(lines)


def enforce_verified_counts(
    summary: ReportSummary, window: ReportWindow
) -> ReportSummary:
    """Overwrite every mechanically-observable field with the real value.

    This is the whole reason Reporter is trustworthy. The model is shown
    the counts and asked to echo them; this discards its echo regardless.
    Same mechanism chain.py uses for TestReport.all_passed and
    AnomalyAlert.repo -- .model_copy(update=...) after validation.

    Only `top_issues` survives from the model, because it's the only
    field that is genuinely a judgment rather than a fact.
    """
    return summary.model_copy(
        update={
            "period_start": window.period_start.isoformat(),
            "period_end": window.period_end.isoformat(),
            "tasks_resolved": window.tasks_resolved,
            "prs_opened": window.prs_opened,
            "prs_merged": window.prs_merged,
            "dependencies_updated": window.dependencies_updated,
        }
    )


async def run_report(
    session: AsyncSession,
    *,
    start: datetime,
    end: datetime,
    model,
) -> tuple[ReportSummary, ReportWindow]:
    """Section 6.8 end to end: collect the window, have Reporter
    summarize it, then re-assert the real numbers over whatever it said.

    Returns both, so a caller can show the report and still prove the
    counts came from the database.
    """
    window = await collect_report_window(session, start, end)

    ctx = ToolContext(
        agent_name="reporter",
        scratch_dir=sandbox_tools.SCRATCH_DIR,
        # Section 6.8: observer. Reporter has no tools at all, so this is
        # belt-and-braces rather than the primary guarantee -- but a
        # future edit that gives it a tool shouldn't silently also give
        # it write permission.
        mode="observer",
        db_session=session,
    )
    agent = ReporterAgent(model, ctx)
    result = await agent.run(render_window_for_prompt(window))

    if not result.success or result.handoff is None:
        raise RuntimeError(f"reporter failed: {result.error or 'no valid handoff'}")

    return enforce_verified_counts(result.handoff, window), window
