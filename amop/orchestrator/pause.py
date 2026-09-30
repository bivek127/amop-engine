"""Emergency shutdown -- spec Section 12.7 / ADR-10.

`amop pause` (CLI) / `/pause` (Telegram) / `POST /system/pause` (API,
what both of those actually call) set a durable, global flag.
`orchestrator/task.py::transition()` -- the one function every real
state change in this codebase already funnels through, directly or via
`transition_with_retry()` (chain.py's CODING/TESTING/REVIEWING loop,
watch.py's triage_anomaly, cli/main.py's approve/reject/cancel flows,
the API's task routes) -- blocks there, once, before starting any NEW
transition, until the flag clears. Gating this single choke point
covers every caller above without touching any of them individually.

This is a soft stop by design, matching ADR-10's own reasoning: a
second, less-tested hard-kill code path is a liability, not a safety
feature. The real emergency stop remains `docker compose down`,
documented, not a new mechanism -- deliberately not built here.

An in-flight agent turn is never interrupted by this. `transition()`
is only ever called BETWEEN turns/attempts (e.g. after Coder's whole
turn produces a diff, before the chain moves to TESTING) -- never
mid-tool-call, since tool calls happen inside an agent's own reasoning
loop (agents/base.py), which never calls transition() itself. So
"in-flight tool calls complete, no new [transitions] start" holds
without this module (or transition()) needing any awareness of the
agent loop, the tool registry, or any of chain.py's many call sites.

Durable, not in-memory: a real Postgres row (database/models.py's
OrchestratorPause), read fresh on every check (never cached via the
ORM identity map -- see _is_paused_raw's own comment for why that
matters). Survives a process restart the same way every other durable-
state pattern in this codebase does (budgets, processed_events) --
no new daemon, no new process; whichever real process calls
transition() next (amop fix, amop watch, amop resume, the API's
background task runner) picks up the current flag automatically.
"""

import asyncio
import logging
import os
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from amop.database.models import OrchestratorPause

logger = logging.getLogger("amop.orchestrator.pause")

# The one real row this table ever holds.
PAUSE_ROW_ID = 1

DEFAULT_POLL_SECONDS = 2.0
POLL_SECONDS = float(
    os.environ.get("AMOP_PAUSE_POLL_SECONDS", str(DEFAULT_POLL_SECONDS))
)


async def _get_or_create_row(session: AsyncSession) -> OrchestratorPause:
    row = await session.get(OrchestratorPause, PAUSE_ROW_ID)
    if row is None:
        row = OrchestratorPause(id=PAUSE_ROW_ID, paused=False)
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def _is_paused_raw(session: AsyncSession) -> bool:
    """The actual polled check -- selects the scalar column directly
    rather than session.get()'ing the ORM row. That distinction is
    load-bearing: Session.get() serves an already-loaded row straight
    from the identity map without a new query, which would make a
    polling loop built on it check the SAME cached value forever and
    never notice another process's/session's commit. A raw column
    SELECT has no identity-map entry to go stale."""
    result = await session.execute(
        select(OrchestratorPause.paused).where(OrchestratorPause.id == PAUSE_ROW_ID)
    )
    value = result.scalar_one_or_none()
    if value is None:
        # No row yet -- default open (not paused), and create the row
        # so a later writer has something to UPDATE rather than having
        # to upsert blindly on every call.
        await _get_or_create_row(session)
        return False
    return bool(value)


async def is_paused(session: AsyncSession) -> bool:
    return await _is_paused_raw(session)


async def get_state(session: AsyncSession) -> OrchestratorPause:
    return await _get_or_create_row(session)


async def set_paused(
    session: AsyncSession, *, paused: bool, by: str
) -> OrchestratorPause:
    """Idempotent: pausing an already-paused system (or resuming an
    already-running one) just refreshes the timestamp/actor rather than
    erroring -- same "triggering an already-applied mode change isn't
    special-cased" shape as approving an already-approved task."""
    row = await _get_or_create_row(session)
    now = datetime.now(UTC)
    row.paused = paused
    row.paused_by = by
    if paused:
        row.paused_at = now
    else:
        row.resumed_at = now
    row.updated_at = now
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def wait_while_paused(session: AsyncSession) -> None:
    """Blocks the CALLING COROUTINE (not the process -- every other
    asyncio task in the same process keeps running normally) until the
    system is unpaused. Called from the very top of
    orchestrator/task.py's transition(), before anything else happens,
    matching Section 12.7's own "the orchestrator checks before
    starting any new task-state transition."

    A task already mid-agent-turn when pause hits is never interrupted
    -- this function isn't reachable from inside that turn at all (see
    module docstring). What actually happens live: the turn finishes
    naturally, the orchestrator's next transition() call for that same
    task lands here and waits, right alongside a brand-new task's very
    first CREATED -> TRIAGING transition -- both held by the exact same
    gate, which is what makes "a second task submitted during the
    pause does not start" and "an in-flight task's next step waits"
    the same mechanism, not two.
    """
    logged = False
    while await _is_paused_raw(session):
        if not logged:
            logger.info(
                "orchestrator paused -- holding new task-state transitions "
                "until resumed"
            )
            logged = True
        await asyncio.sleep(POLL_SECONDS)
    if logged:
        logger.info("orchestrator resumed -- proceeding with pending transitions")
