"""Section 12.7 / ADR-10's emergency shutdown: a durable pause flag
that `orchestrator/task.py::transition()` -- the one function every
real state change in this codebase funnels through -- blocks on before
starting any new transition.

Real Postgres throughout (no mocked session): the actual property under
test is "does a second, independent session/connection see a pause
committed by a different session," which a mocked session can't
exercise honestly.

Requires: Postgres at TEST_DATABASE_URL.
"""

import asyncio
import os

import pytest
import pytest_asyncio
from sqlalchemy import text

from amop.database.session import init_db, make_engine, make_session_factory
from amop.orchestrator import pause as pause_mod
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import create_task, transition

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://localhost/amop_test"
)


@pytest_asyncio.fixture
async def engine():
    eng = make_engine(TEST_DATABASE_URL)
    await init_db(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return make_session_factory(engine)


@pytest_asyncio.fixture(autouse=True)
async def _cleanup(engine):
    yield
    async with engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE task_transitions, tasks, orchestrator_pause RESTART IDENTITY CASCADE")
        )


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    """Real polling loop, real sleeps -- just short ones, so these tests
    run in under a second instead of matching the real default
    (AMOP_PAUSE_POLL_SECONDS, 2s) that a live demo uses."""
    monkeypatch.setattr(pause_mod, "POLL_SECONDS", 0.02)


async def test_defaults_to_not_paused_with_no_row_ever_written(session_factory):
    async with session_factory() as session:
        assert await pause_mod.is_paused(session) is False


async def test_set_paused_then_read_back_from_a_fresh_session(session_factory):
    """Not the in-memory-flag shape this replaces: a SECOND, independent
    session must see what a first session committed."""
    async with session_factory() as writer:
        await pause_mod.set_paused(writer, paused=True, by="operator-x")

    async with session_factory() as reader:
        assert await pause_mod.is_paused(reader) is True
        state = await pause_mod.get_state(reader)
        assert state.paused_by == "operator-x"
        assert state.paused_at is not None


async def test_set_paused_is_idempotent_not_an_error(session_factory):
    async with session_factory() as session:
        await pause_mod.set_paused(session, paused=True, by="a")
        row = await pause_mod.set_paused(session, paused=True, by="b")
        assert row.paused is True
        assert row.paused_by == "b"  # last writer wins, no error


async def test_durability_survives_a_fresh_engine_and_session_factory(session_factory):
    """The actual durability property Section 12.7 asks for: a brand
    new engine (standing in for a genuinely restarted process) must
    still see the paused flag -- not just a second session on the same
    already-open connection pool."""
    async with session_factory() as session:
        await pause_mod.set_paused(session, paused=True, by="operator")

    fresh_engine = make_engine(TEST_DATABASE_URL)
    try:
        fresh_factory = make_session_factory(fresh_engine)
        async with fresh_factory() as session:
            assert await pause_mod.is_paused(session) is True
    finally:
        await fresh_engine.dispose()


async def test_transition_blocks_while_paused_and_proceeds_on_resume(session_factory):
    """The real, end-to-end property: a genuinely in-progress
    transition() call, on ITS OWN session, is held by a pause set from
    a completely different session -- then released within one poll
    tick of a resume from yet another session. No mocking of the wait
    loop itself; this is the real coroutine, really suspended."""
    async with session_factory() as setup_session:
        task = await create_task(setup_session, task_type="bug_fix")
        await pause_mod.set_paused(setup_session, paused=True, by="operator")

    async with session_factory() as transition_session:
        # Re-attach the task to this session's identity map.
        from amop.orchestrator.task import get_task

        task = await get_task(transition_session, task.id)

        transition_task = asyncio.create_task(
            transition(
                transition_session, task, TaskState.TRIAGING, actor="test"
            )
        )
        await asyncio.sleep(0.1)
        assert not transition_task.done(), (
            "transition() must not proceed while paused"
        )

        async with session_factory() as resume_session:
            await pause_mod.set_paused(resume_session, paused=False, by="operator")

        result = await asyncio.wait_for(transition_task, timeout=2.0)
        assert result.state == TaskState.TRIAGING.value


async def test_a_fresh_task_transition_is_also_held_not_just_a_resumed_one(
    session_factory,
):
    """Section 12.7's own "no new [transitions] start" -- proven against
    a task's very FIRST transition (CREATED -> TRIAGING), the same shape
    a brand-new `amop fix` submission goes through, not just a
    mid-chain one."""
    async with session_factory() as setup_session:
        await pause_mod.set_paused(setup_session, paused=True, by="operator")
        task = await create_task(setup_session, task_type="bug_fix")

    async with session_factory() as transition_session:
        from amop.orchestrator.task import get_task

        task = await get_task(transition_session, task.id)
        held = asyncio.create_task(
            transition(transition_session, task, TaskState.TRIAGING, actor="test")
        )
        await asyncio.sleep(0.1)
        assert not held.done()
        held.cancel()
        try:
            await held
        except asyncio.CancelledError:
            pass


async def test_not_paused_never_blocks(session_factory):
    """No regression on the ordinary, unpaused path -- every existing
    caller of transition() must be completely unaffected when nobody
    has ever called set_paused()."""
    async with session_factory() as session:
        task = await create_task(session, task_type="bug_fix")
        result = await asyncio.wait_for(
            transition(session, task, TaskState.TRIAGING, actor="test"), timeout=1.0
        )
        assert result.state == TaskState.TRIAGING.value
