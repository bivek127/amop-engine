"""Milestone 16 — Concurrency Safety.

Adversarial, not illustrative. Every test here was run against the
UNFIXED code first and observed to fail; a concurrency test that has
never been seen failing is not evidence that it can detect anything.
(Milestone 14's lesson, applied deliberately: a profiler test asserted
the hot function was *present* and passed for weeks while its *rank* —
the property that actually mattered — was broken.)

Genuine concurrency, not simulated: every racing coroutine gets its own
AsyncSession, which means its own pooled connection, and an
asyncio.Barrier ensures both writers have LOADED the row before either
is allowed to write. Without that barrier the two coroutines serialize
naturally and the test passes against broken code, proving nothing.

Requires: Postgres at TEST_DATABASE_URL.
"""

import asyncio

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text

from amop.database.models import Task, TaskTransition
from amop.database.session import init_db, make_engine, make_session_factory
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import create_task, transition

TEST_DATABASE_URL = "postgresql+asyncpg://localhost/amop_test"

# CREATED -> ... -> WAITING_FOR_APPROVAL, the real chain order.
_TO_APPROVAL = (
    TaskState.TRIAGING,
    TaskState.INVESTIGATING,
    TaskState.PLANNING_FIX,
    TaskState.CODING,
    TaskState.TESTING,
    TaskState.REVIEWING,
    TaskState.PR_CREATION,
    TaskState.WAITING_FOR_APPROVAL,
)


@pytest_asyncio.fixture
async def engine():
    eng = make_engine(TEST_DATABASE_URL)
    await init_db(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    yield make_session_factory(engine)
    async with engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE task_transitions, tasks RESTART IDENTITY CASCADE")
        )


@pytest_asyncio.fixture
async def session(session_factory):
    async with session_factory() as s:
        yield s


async def _task_at_approval(session):
    task = await create_task(session, task_type="bug_fix", task_context={})
    for state in _TO_APPROVAL:
        await transition(session, task, state)
    return task


async def _count_transitions_from(session, task_id, from_state):
    return await session.scalar(
        select(func.count())
        .select_from(TaskTransition)
        .where(
            TaskTransition.task_id == task_id,
            TaskTransition.from_state == from_state.value,
        )
    )


# ---------------------------------------------------------------------
# A. OCC — two concurrent writers on the SAME task row.
#
# This is not a hypothetical race. Milestone 15 built the second writer:
# `POST /tasks/{id}/approve` and `/reject` (API, Telegram bot, and web
# dashboard all reach them) both call transition() on a task that
# `run_fix()` may still be holding in memory. Two humans -- or one human
# on two surfaces -- approving and rejecting at once is the literal
# scenario reproduced here.
# ---------------------------------------------------------------------


async def test_occ_two_concurrent_writers_leave_exactly_one_transition(
    session, session_factory
):
    """The assertion with teeth is the audit-row count, not the state.

    A lost update writes TWO task_transitions rows (both writers believe
    they moved the task) while `tasks.state` reflects only whichever
    committed last. The audit trail and the row then disagree
    permanently, and nothing in the system ever notices -- which is
    precisely the silent corruption this milestone exists to stop.
    """
    task = await _task_at_approval(session)
    task_id, before = task.id, task.version

    barrier = asyncio.Barrier(2)

    async def writer(to_state: TaskState):
        async with session_factory() as s:
            row = await s.get(Task, task_id)
            # Both writers now hold the same version. Nothing has been
            # written yet -- this is the exact interleaving that makes
            # the race real rather than accidental.
            await barrier.wait()
            return await transition(s, row, to_state, actor="human:test")

    outcomes = await asyncio.gather(
        writer(TaskState.MERGED),
        writer(TaskState.CANCELLED),
        return_exceptions=True,
    )

    failures = [o for o in outcomes if isinstance(o, BaseException)]
    successes = [o for o in outcomes if not isinstance(o, BaseException)]

    assert len(successes) == 1, (
        f"expected exactly one writer to win, got {len(successes)} "
        f"-- both writes landed, this is a lost update"
    )
    assert len(failures) == 1, "expected the losing writer to detect the conflict"

    await session.commit()  # drop this session's snapshot, re-read fresh
    rows = await _count_transitions_from(
        session, task_id, TaskState.WAITING_FOR_APPROVAL
    )
    assert rows == 1, (
        f"{rows} transitions recorded out of WAITING_FOR_APPROVAL -- the audit "
        f"trail claims the task left that state more than once"
    )

    fresh = await session.get(Task, task_id, populate_existing=True)
    assert fresh.state in {TaskState.MERGED.value, TaskState.CANCELLED.value}
    assert fresh.version == before + 1, (
        f"version went {before} -> {fresh.version}; exactly one write landed, "
        f"so exactly one increment should have"
    )


async def test_a_stale_object_cannot_bypass_the_state_machine(session, session_factory):
    """The sharpest consequence of the missing guard, and the reason this
    is a safety bug rather than a tidiness one.

    `transition()` validates against `TaskState(task.state)` -- the value
    on the in-memory object. When another writer has already moved the row,
    that value is stale, so validate_transition() passes judgement on a
    state the task is no longer in. The result is an ILLEGAL transition
    landing in the database: a task a human explicitly CANCELLED comes
    back as MERGED. Milestone 1's state machine is the thing every later
    milestone leans on for "this cannot happen", and a stale object walks
    straight around it.

    No barrier and no race here -- the writes are strictly ordered. That
    is deliberate: this failure needs no unlucky timing, only a second
    writer somewhere in the system, which Milestone 15 shipped.
    """
    task = await _task_at_approval(session)
    task_id = task.id

    # Writer A loads the task and holds it (a chain mid-run does exactly
    # this for minutes at a time).
    stale_session = session_factory()
    stale_task = await stale_session.get(Task, task_id)

    # Writer B rejects it. Committed, done, final.
    async with session_factory() as s:
        row = await s.get(Task, task_id)
        await transition(s, row, TaskState.CANCELLED, actor="human:telegram")

    # Writer A now acts on what it still believes the state to be.
    outcome = None
    try:
        await transition(
            stale_session, stale_task, TaskState.MERGED, actor="human:web"
        )
    except Exception as exc:  # noqa: BLE001 -- any refusal is a pass here
        outcome = exc
    finally:
        await stale_session.close()

    await session.commit()
    fresh = await session.get(Task, task_id, populate_existing=True)

    assert outcome is not None, (
        "a stale writer was allowed to transition a CANCELLED task -- "
        "validate_transition() judged the move against a state the task "
        "had already left"
    )
    assert fresh.state == TaskState.CANCELLED.value, (
        f"human-issued CANCELLED was overwritten to {fresh.state} by a "
        f"writer holding a stale snapshot"
    )
