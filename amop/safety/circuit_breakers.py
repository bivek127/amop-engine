"""Circuit Breakers — spec Section 12.3, Design Decision D-11 ("a per-task
cost cap alone is insufficient... breakers must compose across three
independent dimensions: per-task cost, cumulative cost, and failure
velocity"). Section 29.4 cross-references this as the single
highest-value finding across the spec's own review history: a per-task
cap composes to an unbounded total unless something caps the number and
velocity of tasks themselves, independent of what each one costs.

This milestone builds four of the full spec table's breakers (the ones
CLAUDE.md's Milestone 9 scope names) -- `max_tool_calls_per_task` /
`max_cost_per_task_usd` / `cost_velocity_alerts` (needs Telegram) /
`retry_backoff` / `consecutive_failure_cooldown` are real, spec'd
breakers this milestone deliberately does not build.

Pure query functions against the existing `tasks` table -- no new
tables. `Task.task_context->>'key'` JSONB queries are a new pattern in
this codebase (nothing has queried task_context by key before this
milestone; every prior write to it was CLI-layer bookkeeping only, see
cli/main.py) but a straightforward, idiomatic one:
`Task.task_context['key'].astext == value` compiles to Postgres's `->>`
operator. Unindexed at this scale, matching CodeChunk's own "no ANN
index... fixture-repo scale is fast enough" reasoning -- a GIN index on
task_context is a trivial future add if task volume ever makes this
matter, not needed for a demo/test-scale deployment.

Cost is a documented placeholder, not real token-based accounting: Ollama
(the only real provider) has zero marginal dollar cost, and
`OllamaProvider.complete()`'s real captured token counts
(`input_tokens`/`output_tokens`) are discarded by agents/base.py's
_run_loop today -- wiring real costing through would mean modifying a
Milestone-0 "completed" file for something this milestone's brief
doesn't ask for. AMOP_COST_ESTIMATE_PER_TASK_USD is a configurable flat
estimate, stored into task_context at task-creation time and summed from
there (not recomputed), so a future variable-cost-per-task-type change
wouldn't require touching this module again.

Matches safety/engine.py's Decision dataclass shape exactly (allow_/deny
classmethods, no exceptions) -- same "software decides, plainly, no
exceptions to catch" discipline as every other safety check in this
codebase.
"""

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from amop.database.models import Task
from amop.orchestrator.state_machine import TaskState

DEFAULT_COST_ESTIMATE_PER_TASK_USD = 0.05
DEFAULT_MAX_COST_PER_DAY_USD = 25.00
DEFAULT_MAX_TASKS_PER_HOUR = 20
DEFAULT_MAX_ANOMALIES_PER_HOUR_PER_REPO = 10
DEFAULT_FAILURE_STREAK_THRESHOLD = 3

# AMOP_* env var convention (AMOP_SCRATCH_DIR, AMOP_MAX_LOC_PER_TASK, ...).
COST_ESTIMATE_PER_TASK_USD = float(
    os.environ.get(
        "AMOP_COST_ESTIMATE_PER_TASK_USD", str(DEFAULT_COST_ESTIMATE_PER_TASK_USD)
    )
)
MAX_COST_PER_DAY_USD = float(
    os.environ.get("AMOP_MAX_COST_PER_DAY_USD", str(DEFAULT_MAX_COST_PER_DAY_USD))
)
MAX_TASKS_PER_HOUR = int(
    os.environ.get("AMOP_MAX_TASKS_PER_HOUR", str(DEFAULT_MAX_TASKS_PER_HOUR))
)
MAX_ANOMALIES_PER_HOUR_PER_REPO = int(
    os.environ.get(
        "AMOP_MAX_ANOMALIES_PER_HOUR_PER_REPO",
        str(DEFAULT_MAX_ANOMALIES_PER_HOUR_PER_REPO),
    )
)
FAILURE_STREAK_THRESHOLD = int(
    os.environ.get(
        "AMOP_FAILURE_STREAK_THRESHOLD", str(DEFAULT_FAILURE_STREAK_THRESHOLD)
    )
)


@dataclass
class BreakerResult:
    allow: bool
    reason: str | None = None

    @classmethod
    def allow_(cls) -> "BreakerResult":
        return cls(allow=True)

    @classmethod
    def deny(cls, reason: str) -> "BreakerResult":
        return cls(allow=False, reason=reason)


def _utc_day_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


async def check_cost_breaker(
    session: AsyncSession,
    estimate_usd: float = COST_ESTIMATE_PER_TASK_USD,
    cap_usd: float = MAX_COST_PER_DAY_USD,
) -> BreakerResult:
    """Section 12.3: max_cost_per_day_usd -- global, cumulative across all
    tasks/repos. Check-before-dispatch (Section 11.3.2's pattern): reject
    if spent-so-far + this task's estimate would exceed the cap, don't
    cancel in-flight work. Trip action per spec: "orchestrator drops into
    global observer mode... until UTC day rollover" -- this function is
    the check a caller makes before creating a task; it doesn't itself
    flip a global mode (no such global-mode concept exists in this
    codebase's Milestone 2 permission-mode design, which is per-ToolContext,
    not orchestrator-wide) -- the caller (orchestrator/watch.py,
    cli/main.py) is responsible for refusing task creation on a `deny`.
    """
    day_start = _utc_day_start()
    result = await session.execute(
        select(Task.task_context["cost_estimate_usd"].astext).where(
            Task.created_at >= day_start
        )
    )
    spent = sum(float(v) for v in result.scalars().all() if v is not None)
    if spent + estimate_usd > cap_usd:
        return BreakerResult.deny(
            f"max_cost_per_day_usd cap (${cap_usd:.2f}) would be exceeded: "
            f"${spent:.2f} already spent today + ${estimate_usd:.2f} estimate "
            f"for this task"
        )
    return BreakerResult.allow_()


async def check_tasks_per_hour(
    session: AsyncSession, cap: int = MAX_TASKS_PER_HOUR
) -> BreakerResult:
    """Section 12.3: max_tasks_per_hour -- global, orchestrator-wide.
    Trip action per spec: "new tasks queue instead of starting." For
    Watcher's poll loop this is naturally satisfied without a real queue
    data structure: a refused candidate simply gets re-examined on the
    next poll cycle (dedup layer 1a makes that re-examination nearly
    free, see orchestrator/watch.py) -- effectively a queue with the poll
    interval as its tick. cli/main.py's one-shot `fix` command has no
    such natural retry and must decide explicitly how to surface a
    refusal (see that module).
    """
    cutoff = datetime.now(UTC) - timedelta(hours=1)
    result = await session.execute(
        select(func.count()).select_from(Task).where(Task.created_at >= cutoff)
    )
    count = result.scalar_one()
    if count >= cap:
        return BreakerResult.deny(
            f"max_tasks_per_hour cap ({cap}) reached: {count} tasks created "
            "in the last hour"
        )
    return BreakerResult.allow_()


async def check_anomaly_rate(
    session: AsyncSession, repo: str, cap: int = MAX_ANOMALIES_PER_HOUR_PER_REPO
) -> BreakerResult:
    """Section 12.3: anomaly_rate_breaker -- per-repo, evaluated inside
    Watcher before task creation. Trip action per spec: stop spawning new
    tasks for that repo and raise ONE meta-AnomalyAlert instead
    ("detection rate abnormal -- investigate the detector, not the app")
    -- this is what actually stops a false-positive storm at the source,
    rather than merely capping its cost. The meta-alert itself is built
    by the caller (orchestrator/watch.py); this function only answers
    allow/deny.
    """
    cutoff = datetime.now(UTC) - timedelta(hours=1)
    result = await session.execute(
        select(func.count())
        .select_from(Task)
        .where(
            Task.created_at >= cutoff,
            Task.task_context["source"].astext == "github_watcher",
            Task.task_context["repo"].astext == repo,
        )
    )
    count = result.scalar_one()
    if count >= cap:
        return BreakerResult.deny(
            f"anomaly_rate_breaker cap ({cap}/hour) reached for {repo}: "
            f"{count} watcher-created tasks in the last hour -- detection "
            "rate abnormal, investigate the detector, not the app"
        )
    return BreakerResult.allow_()


async def check_failure_streak(
    session: AsyncSession,
    repo: str,
    github_issue_number: int | None = None,
    streak: int = FAILURE_STREAK_THRESHOLD,
    exclude_task_id=None,
) -> BreakerResult:
    """Section 12.3: failure_streak_breaker -- spec scopes this
    PER-INCIDENT ("N consecutive FAILED outcomes on the same incident_id"),
    a different, narrower breaker than consecutive_failure_cooldown
    (per-repo, not built this milestone). With no `incidents` table, the
    closest faithful proxy is (repo, github_issue_number), not repo
    alone -- a repo-only scope would block ALL future auto-tasks on a
    repo after 3 unrelated failures, when spec intends only blocking
    retries of the SAME underlying bug. Falls back to repo-only for
    non-GitHub-sourced tasks (e.g. a CLI `amop fix` run) that have no
    issue number to key on.

    `exclude_task_id`: the same self-inclusion bug found and fixed in
    orchestrator/watch.py's find_similar_open_task applies here too --
    orchestrator/watch.py's triage_anomaly() calls this AFTER the
    candidate task is already created and committed (in TRIAGING, so it
    matches this query's (repo, issue_number) filter). Left unexcluded,
    the most-recent-N window includes the candidate itself, displacing a
    real FAILED entry and silently masking a real streak. Caught live
    during implementation, not theoretical -- same class of bug as the
    dedup one, same fix shape.

    Trip action per spec: further auto-retry is blocked "regardless of
    remaining cost budget" -- deliberately independent of the cost
    breaker, because repeated failure is a signal that more attempts
    won't help, not just that they're expensive. Routes to
    NEEDS_HUMAN_INPUT; the caller (orchestrator/watch.py) performs that
    transition, this function only answers allow/deny.
    """
    filters = [Task.task_context["repo"].astext == repo]
    if github_issue_number is not None:
        filters.append(
            Task.task_context["github_issue_number"].astext == str(github_issue_number)
        )
    if exclude_task_id is not None:
        filters.append(Task.id != exclude_task_id)

    result = await session.execute(
        select(Task.state)
        .where(*filters)
        .order_by(Task.created_at.desc())
        .limit(streak)
    )
    recent_states = result.scalars().all()
    if len(recent_states) >= streak and all(
        s == TaskState.FAILED.value for s in recent_states
    ):
        target = f"{repo}" + (
            f" issue #{github_issue_number}" if github_issue_number is not None else ""
        )
        return BreakerResult.deny(
            f"failure_streak_breaker: {streak} consecutive FAILED outcomes on "
            f"{target} -- further auto-retry blocked regardless of remaining "
            "cost budget"
        )
    return BreakerResult.allow_()
