import os

import pytest
import pytest_asyncio
from sqlalchemy import text

from amop.database.session import init_db, make_engine, make_session_factory
from amop.orchestrator.state_machine import (
    TRANSITIONS,
    IllegalTransitionError,
    TaskState,
)
from amop.orchestrator.task import create_task, get_transitions, transition

# Separate database from amop_dev — dev and test data never mix. Override
# with TEST_DATABASE_URL if the human's local test db is named differently.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://localhost/amop_test"
)

# The full legal bug_fix path, Section 4.1: CREATED -> ... -> RESOLVED.
FULL_LEGAL_PATH = [
    TaskState.TRIAGING,
    TaskState.INVESTIGATING,
    TaskState.PLANNING_FIX,
    TaskState.CODING,
    TaskState.TESTING,
    TaskState.REVIEWING,
    TaskState.PR_CREATION,
    TaskState.WAITING_FOR_APPROVAL,
    TaskState.MERGED,
    TaskState.RESOLVED,
]

ILLEGAL_TRANSITIONS = [
    (TaskState.CREATED, TaskState.RESOLVED),  # skips the entire machine
    (TaskState.CREATED, TaskState.CODING),  # skips triage/investigation/planning
    (TaskState.RESOLVED, TaskState.CODING),  # RESOLVED is terminal
    (TaskState.TESTING, TaskState.MERGED),  # skips review/PR/approval
]


@pytest_asyncio.fixture
async def engine():
    # Function-scoped, not session-scoped: pytest-asyncio gives each test
    # its own event loop, and an asyncpg connection pool created in one
    # loop can't be reused from another ("another operation is in
    # progress" is what that looks like). A fresh engine per test avoids
    # crossing that boundary. init_db()'s create_all is idempotent, so
    # re-running it every test is just a no-op after the first.
    eng = make_engine(TEST_DATABASE_URL)
    await init_db(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    session_factory = make_session_factory(engine)
    async with session_factory() as s:
        yield s
    # Keep amop_test empty between tests so runs don't accumulate rows.
    async with engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE task_transitions, tasks RESTART IDENTITY CASCADE")
        )


@pytest.mark.parametrize("from_state,to_state", list(TRANSITIONS.keys()))
async def test_every_legal_transition_succeeds(session, from_state, to_state):
    task = await create_task(session, task_type="bug_fix")
    task.state = from_state.value
    session.add(task)
    await session.commit()

    updated = await transition(session, task, to_state, actor="test")

    assert updated.state == to_state.value
    history = await get_transitions(session, task.id)
    assert len(history) == 1
    assert history[0].from_state == from_state.value
    assert history[0].to_state == to_state.value
    assert history[0].actor == "test"


@pytest.mark.parametrize("from_state,to_state", ILLEGAL_TRANSITIONS)
async def test_illegal_transition_raises_and_changes_nothing(
    session, from_state, to_state
):
    task = await create_task(session, task_type="bug_fix")
    task.state = from_state.value
    session.add(task)
    await session.commit()

    with pytest.raises(IllegalTransitionError):
        await transition(session, task, to_state, actor="test")

    # Rejected transition must not be recorded, and the task's state must
    # not have moved.
    history = await get_transitions(session, task.id)
    assert history == []
    await session.refresh(task)
    assert task.state == from_state.value


async def test_full_legal_path_created_to_resolved(session):
    task = await create_task(session, task_type="bug_fix")
    assert task.state == TaskState.CREATED.value

    for to_state in FULL_LEGAL_PATH:
        task = await transition(session, task, to_state, actor="test")

    assert task.state == TaskState.RESOLVED.value
    assert task.resolved_at is not None


async def test_transition_history_is_queryable_and_in_order(session):
    task = await create_task(session, task_type="bug_fix")
    for to_state in FULL_LEGAL_PATH:
        task = await transition(session, task, to_state, actor="test")

    history = await get_transitions(session, task.id)

    assert len(history) == len(FULL_LEGAL_PATH)
    expected_from = [TaskState.CREATED.value] + [
        s.value for s in FULL_LEGAL_PATH[:-1]
    ]
    assert [h.from_state for h in history] == expected_from
    assert [h.to_state for h in history] == [s.value for s in FULL_LEGAL_PATH]
    # Rows must come back in the order they actually happened.
    ids = [h.id for h in history]
    assert ids == sorted(ids)
