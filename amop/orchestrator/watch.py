"""Watcher's per-anomaly triage flow — spec Section 6.1's dedup + Section
4.2's TRIAGING state, wired together for real for the first time.

Deliberately NOT part of orchestrator/chain.py: `triage_anomaly()` below
is the glue between "Watcher produced an AnomalyAlert" and "a real Task
exists, correctly triaged" -- chain.py's `run_chain()` is only ever
entered once a task is already known-novel and past the floor, and stays
that way (its own TRIAGING step is deliberately minimal, see the
Milestone 9 comment there). Keeping this in its own module means
chain.py's core chain logic isn't touched beyond the one necessary
conditional fix noted there.

Dedup — two layers, different timing (see cli/main.py's `watch` command
for layer 1a, the cheap exact-match filter applied BEFORE Watcher's
prompt is even built):

  Layer 1a (cheap, exact, upstream of this module): does a non-terminal
  task already exist for this exact (repo, github_issue_number)? Filters
  the issue list before Watcher ever sees it -- a filtered issue was
  never a real candidate, so it shouldn't count against any breaker.

  Layer 1b (spec's real semantic check, here): embed the AnomalyAlert's
  summary and compare against every currently-open bug_fix task's stored
  description, cosine similarity. Spec has two different thresholds for
  this -- Section 4.2's transition-table comment says >= 0.92, Section
  6.1's own prose says >= 0.9. Anchored on 0.92 since it's already
  committed to code in state_machine.py's comment for the
  (TRIAGING, MERGED_INTO_EXISTING) edge this function is the first real
  caller of.

No persisted embedding column / no `incidents` table this milestone
(CLAUDE.md's explicit authorization: "reuse what exists where
reasonable") -- layer 1b re-embeds every open task's description fresh
on every call. Fine at demo/test scale; a real production deployment
with hundreds of concurrently-open tasks would want a persisted
embedding column and a proper ANN index, same story as CodeChunk's own
documented "not needed at this scale" reasoning.
"""

import math
import os
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from amop.agents.handoffs import AnomalyAlert
from amop.codebase_intel.embeddings import embed_texts
from amop.database.models import Task
from amop.memory import store as memory_store
from amop.orchestrator.state_machine import TERMINAL_STATES, TaskState
from amop.orchestrator.task import create_task, transition
from amop.safety import circuit_breakers

# Section 4.2's transition-table comment; see module docstring for why
# this number was chosen over Section 6.1's own "0.9" prose.
DEDUP_SIMILARITY_THRESHOLD = 0.92

_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
DEFAULT_SEVERITY_FLOOR = "medium"
SEVERITY_FLOOR = os.environ.get("AMOP_WATCHER_SEVERITY_FLOOR", DEFAULT_SEVERITY_FLOOR)


def _noop(_message: str) -> None:
    pass


def _non_terminal_states() -> list[str]:
    return [s.value for s in TaskState if s not in TERMINAL_STATES]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def find_existing_task_for_issue(
    session: AsyncSession, repo: str, issue_number: int
) -> Task | None:
    """Dedup layer 1a: cheap, exact match against a non-terminal task's
    stored (repo, github_issue_number). No LLM call, no embedding call --
    the filter cli/main.py's watch loop applies before an issue ever
    reaches Watcher's prompt."""
    result = await session.execute(
        select(Task).where(
            Task.task_context["repo"].astext == repo,
            Task.task_context["github_issue_number"].astext == str(issue_number),
            Task.state.in_(_non_terminal_states()),
        )
    )
    return result.scalars().first()


async def find_similar_open_task(
    session: AsyncSession,
    summary: str,
    *,
    exclude_task_id,
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> Task | None:
    """Dedup layer 1b -- see module docstring.

    `exclude_task_id` is required, not optional: by the time this runs,
    the candidate task has already been created and committed in
    TRIAGING (a non-terminal state) -- so it would otherwise appear in
    its own comparison set and match itself at similarity 1.0 every
    time, marking every new task a "duplicate" of itself. Caught live
    during implementation, not theoretical.
    """
    result = await session.execute(
        select(Task).where(
            Task.task_type == "bug_fix",
            Task.state.in_(_non_terminal_states()),
            Task.id != exclude_task_id,
        )
    )
    open_tasks = list(result.scalars().all())
    if not open_tasks:
        return None

    descriptions = [(t.task_context or {}).get("prompt", "") for t in open_tasks]
    all_embeddings = await embed_texts([summary, *descriptions])
    summary_vec, existing_vecs = all_embeddings[0], all_embeddings[1:]

    for task, vec in zip(open_tasks, existing_vecs, strict=True):
        if _cosine_similarity(summary_vec, vec) >= threshold:
            return task
    return None


async def _find_prior_resolved_incident(
    session: AsyncSession, summary: str, local_path: Path | None
):
    """Milestone 14: the resolved-history half of dedup, backed by
    long-term memory (Section 10.2's first consumer).

    Returns a MemoryMatch or None. Best-effort -- an unavailable
    embedding model must not stop triage, since this is advisory
    context, not a gate.
    """
    if local_path is None:
        return None
    try:
        matches = await memory_store.search_memory(
            session,
            summary,
            repo_path=str(Path(local_path).resolve()),
            top_k=1,
            min_similarity=DEDUP_SIMILARITY_THRESHOLD,
        )
    except Exception:  # noqa: BLE001 -- advisory only, never blocks triage
        return None
    return matches[0] if matches else None


async def triage_anomaly(
    session: AsyncSession,
    alert: AnomalyAlert,
    *,
    local_path: Path | None,
    model,
    scratch_root: Path | None = None,
    emit: Callable[[str], None] = _noop,
) -> Task | None:
    """The full per-anomaly flow:

        check_cost_breaker + check_tasks_per_hour (before ANY task row
        exists -- "check-before-dispatch", Section 11.3.2)
          -> create_task(CREATED) -> transition(TRIAGING)
          -> dedup layer 1b
               -> match? transition(MERGED_INTO_EXISTING), done
          -> check_failure_streak (independent of cost, per spec)
               -> tripped? transition(NEEDS_HUMAN_INPUT), done
          -> severity < floor?
               -> transition(CANCELLED), done
          -> novel, above floor: hand off to run_fix() against
             local_path if given, else stop at TRIAGING (detection-only
             mode when no local clone is configured for this repo).

    `alert.repo`/`alert.github_issue_number` are assumed already stamped
    by the caller (cli/main.py's watch loop, using alert.issue_index to
    look them up from its own ground-truth pre-fetched issue list) --
    see agents/handoffs.py's AnomalyAlert docstring for why the model's
    own echo is never trusted for these two fields.

    Returns None (no Task created) when a breaker blocks creation this
    cycle -- not an error. The underlying GitHub issue is still open, so
    the next poll cycle naturally reconsiders it (dedup layer 1a is a
    no-op until a task actually exists), which is what "queue" means for
    max_tasks_per_hour's spec'd trip action here (see
    circuit_breakers.check_tasks_per_hour's docstring).
    """
    cost_check = await circuit_breakers.check_cost_breaker(session)
    if not cost_check.allow:
        emit(f"BLOCKED (max_cost_per_day_usd): {cost_check.reason}")
        return None

    rate_check = await circuit_breakers.check_tasks_per_hour(session)
    if not rate_check.allow:
        emit(f"BLOCKED (max_tasks_per_hour): {rate_check.reason}")
        return None

    task = await create_task(
        session,
        task_type="bug_fix",
        task_context={
            "prompt": alert.summary,
            "repo": alert.repo,
            "source": "github_watcher",
            "github_issue_number": alert.github_issue_number,
            "severity": alert.severity,
            "confidence": alert.confidence,
            "cost_estimate_usd": circuit_breakers.COST_ESTIMATE_PER_TASK_USD,
        },
        severity=alert.severity,
    )
    task = await transition(
        session,
        task,
        TaskState.TRIAGING,
        actor="agent:watcher",
        trigger="AnomalyAlert accepted for triage",
    )
    emit(f"TRIAGING: task {task.id} for {alert.repo}#{alert.github_issue_number}")

    dup = await find_similar_open_task(session, alert.summary, exclude_task_id=task.id)
    if dup is not None:
        task = await transition(
            session,
            task,
            TaskState.MERGED_INTO_EXISTING,
            actor="system",
            trigger=(
                f"dedup match against open task {dup.id} "
                f"(similarity >= {DEDUP_SIMILARITY_THRESHOLD})"
            ),
        )
        emit(f"MERGED_INTO_EXISTING: duplicate of task {dup.id}")
        return task

    # Milestone 14 / Section 10.2's second dedup consumer: layer 1b above
    # can only ever see NON-terminal tasks, so a bug that was already
    # diagnosed and closed months ago is invisible to it. Long-term
    # memory is exactly the record of those.
    #
    # Deliberately advisory, not a MERGED_INTO_EXISTING transition: that
    # transition's own spec trigger is "dedup match against open
    # incident", and a recurring bug whose earlier fix did not hold is a
    # genuinely NEW task that happens to rhyme with an old one -- not a
    # duplicate of something a human could go look at. Silently closing
    # it would suppress precisely the signal (this regressed) that makes
    # incident memory worth keeping. So: surfaced to the operator, and
    # the task proceeds.
    prior = await _find_prior_resolved_incident(session, alert.summary, local_path)
    if prior is not None:
        emit(
            f"NOTE: similar past incident on record ({prior.similarity:.2f} "
            f"similarity, outcome={prior.item.content.get('outcome')}): "
            f"{str(prior.item.content.get('root_cause'))[:120]} "
            "-- proceeding anyway (possible recurrence, not a duplicate)"
        )

    streak_check = await circuit_breakers.check_failure_streak(
        session, alert.repo, alert.github_issue_number, exclude_task_id=task.id
    )
    if not streak_check.allow:
        task = await transition(
            session,
            task,
            TaskState.NEEDS_HUMAN_INPUT,
            actor="system",
            trigger=streak_check.reason,
        )
        emit(f"NEEDS_HUMAN_INPUT: {streak_check.reason}")
        return task

    if _SEVERITY_ORDER.get(alert.severity, 0) < _SEVERITY_ORDER.get(SEVERITY_FLOOR, 1):
        task = await transition(
            session,
            task,
            TaskState.CANCELLED,
            actor="system",
            trigger=f"severity {alert.severity!r} below configured floor {SEVERITY_FLOOR!r}",
        )
        emit(f"CANCELLED: severity {alert.severity} below floor {SEVERITY_FLOOR}")
        return task

    if local_path is None:
        emit(
            f"TRIAGING: novel, above floor, but no --local-path configured for "
            f"{alert.repo} -- stopping at task creation this milestone"
        )
        return task

    # Local import: chain.py doesn't import this module, so there's no
    # real cycle, but keeping run_fix's import scoped to where it's used
    # avoids ever creating one as this file grows.
    from amop.orchestrator.chain import run_fix

    result = await run_fix(
        session,
        task,
        description=alert.summary,
        repo_path=local_path,
        model=model,
        scratch_root=scratch_root,
        mode="operator",
        emit=emit,
    )
    return result.task
