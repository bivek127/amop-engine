"""Milestone 18 (Resumed) — Dashboard: Deploy-and-Watch, Not Just a Viewer.

Step 0 found, with real code evidence, that `POST /tasks` never executed
the chain: `create_task_endpoint`'s own docstring said so, and nothing
else in the codebase picks up an API-created `CREATED` row (`amop
watch`'s poll loop only ever creates its own tasks from GitHub issues;
`amop serve-api` is bare uvicorn with no worker). This file covers the
two things that close that gap:

  1. `POST /tasks` now schedules `run_fix_and_persist` via FastAPI
     `BackgroundTasks` for a well-formed `bug_fix` submission (real repo
     + description) -- and, just as importantly, does NOT for anything
     else (no repo/description, or a non-bug_fix task_type), preserving
     Milestone 15's original "create the row only" contract for those.
  2. `run_fix_and_persist` itself: a successful run persists the same
     task_context shape every other run_fix() caller already does
     (`persist_chain_result`), and a failure -- before run_chain() ever
     starts, or after it's already moved the task further -- lands on a
     real, legal, observable state instead of leaving the task stuck at
     CREATED forever with nothing to explain why.

`run_fix()` itself is never called for real here (needs a live Docker
daemon + Ollama, same as every other chain test in this project) --
`chain.run_fix` is monkeypatched to a scripted stub, same pattern
test_milestone29.py already uses for its own sandbox-free chain-glue
tests.

Requires: Postgres at TEST_DATABASE_URL. No Docker, no Ollama.
"""

import uuid
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from amop.api.app import app
from amop.api.deps import configure_session_factory, get_session
from amop.database.session import init_db, make_engine, make_session_factory
from amop.orchestrator import chain as chain_module
from amop.orchestrator.chain import ChainResult, run_fix_and_persist
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import create_task, get_task, transition

TEST_DATABASE_URL = "postgresql+asyncpg://localhost/amop_test"
TEST_TOKEN = "test-token-do-not-use-in-prod"


@pytest_asyncio.fixture
async def engine():
    eng = make_engine(TEST_DATABASE_URL)
    await init_db(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    session_factory = make_session_factory(engine)
    async with session_factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE task_transitions, tasks, repositories RESTART IDENTITY CASCADE")
        )


@pytest.fixture(autouse=True)
def _api_token(monkeypatch):
    monkeypatch.setenv("AMOP_API_TOKEN", TEST_TOKEN)


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


@pytest_asyncio.fixture
async def client(engine):
    """Same pattern as test_milestone15.py/test_milestone21.py's own
    `client` fixtures: overrides the request-scoped `get_session`
    dependency AND configures deps.py's module-level session factory,
    since `POST /tasks`'s background job (like the webhook's) reads that
    factory directly, outside FastAPI's per-request DI cycle."""
    session_factory = make_session_factory(engine)

    async def _override_get_session():
        async with session_factory() as s:
            yield s

    app.dependency_overrides[get_session] = _override_get_session
    configure_session_factory(session_factory)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------
# POST /tasks -- when background execution is (and isn't) scheduled
# ---------------------------------------------------------------------


@pytest.fixture
def scheduled_calls(monkeypatch):
    """Replaces the real run_fix_and_persist the route would schedule
    with a recorder -- proves the route decided to schedule it, and with
    what arguments, without ever touching run_fix()/Docker/Ollama."""
    calls = []

    async def _fake(session_factory, task_id, **kwargs):
        calls.append({"task_id": task_id, **kwargs})

    monkeypatch.setattr("amop.api.routes.tasks.run_fix_and_persist", _fake)
    return calls


async def test_bug_fix_with_repo_and_description_schedules_background_execution(
    client, auth_headers, scheduled_calls
):
    r = await client.post(
        "/tasks",
        json={"task_type": "bug_fix", "repo": "/repo/real", "description": "500 on checkout"},
        headers=auth_headers,
    )
    assert r.status_code == 201, r.text
    task_id = uuid.UUID(r.json()["id"])

    assert len(scheduled_calls) == 1
    call = scheduled_calls[0]
    assert call["task_id"] == task_id
    assert call["repo_path"] == Path("/repo/real")
    assert call["description"] == "500 on checkout"


async def test_bug_fix_without_a_repo_schedules_nothing(client, auth_headers, scheduled_calls):
    """Preserves Milestone 15's original contract for this shape: no repo
    means there is nothing real to run yet (a task a human will populate
    or dispatch some other way), same as before this milestone."""
    r = await client.post(
        "/tasks",
        json={"task_type": "bug_fix", "description": "no repo given"},
        headers=auth_headers,
    )
    assert r.status_code == 201
    assert scheduled_calls == []


async def test_bug_fix_without_a_description_schedules_nothing(
    client, auth_headers, scheduled_calls
):
    r = await client.post(
        "/tasks",
        json={"task_type": "bug_fix", "repo": "/repo/real"},
        headers=auth_headers,
    )
    assert r.status_code == 201
    assert scheduled_calls == []


async def test_non_bug_fix_task_types_never_schedule_background_execution(
    client, auth_headers, scheduled_calls
):
    """optimization/dependency_update are explicitly out of this
    milestone's scope -- they still only ever run via `amop
    optimize`/`amop update-deps`, not the dashboard's bug-report form."""
    r = await client.post(
        "/tasks",
        json={
            "task_type": "optimization",
            "repo": "/repo/real",
            "description": "make it faster",
        },
        headers=auth_headers,
    )
    assert r.status_code == 201
    assert scheduled_calls == []


# ---------------------------------------------------------------------
# run_fix_and_persist -- success and failure both land somewhere real
# ---------------------------------------------------------------------


async def test_run_fix_and_persist_persists_a_successful_result_like_every_other_caller(
    session, engine, monkeypatch
):
    task = await create_task(
        session, task_type="bug_fix", task_context={"repo": "/repo/real", "prompt": "x"}
    )

    async def _fake_run_fix(session, task, **kwargs):
        return ChainResult(
            task=task,
            final_state=TaskState.RESOLVED,
            diff="--- a/x.py\n+++ b/x.py\n",
            stages=["CODING: done"],
            tool_calls=[{"name": "patch_file", "success": True}],
        )

    monkeypatch.setattr(chain_module, "run_fix", _fake_run_fix)

    session_factory = make_session_factory(engine)
    await run_fix_and_persist(
        session_factory,
        task.id,
        description="x",
        repo_path=Path("/repo/real"),
        model=object(),
    )

    async with session_factory() as verify_session:
        reloaded = await get_task(verify_session, task.id)
        assert reloaded.task_context["final_state"] == "RESOLVED"
        assert reloaded.task_context["diff"] == "--- a/x.py\n+++ b/x.py\n"
        assert reloaded.task_context["tool_calls"] == [
            {"name": "patch_file", "success": True}
        ]


async def test_run_fix_and_persist_lands_a_pre_chain_failure_on_cancelled_not_stuck_at_created(
    session, engine, monkeypatch
):
    """The real, previously-unasserted failure this milestone closes: a
    bad repo path fails inside run_fix()'s own pre-chain setup
    (materialize(), before run_chain() ever runs), which has no internal
    error handling of its own. Without this, the task sits at CREATED
    forever with nothing on the dashboard to explain why."""
    task = await create_task(
        session, task_type="bug_fix", task_context={"repo": "/nope", "prompt": "x"}
    )

    async def _fake_run_fix(session, task, **kwargs):
        raise FileNotFoundError("repo not found: /nope")

    monkeypatch.setattr(chain_module, "run_fix", _fake_run_fix)

    session_factory = make_session_factory(engine)
    await run_fix_and_persist(
        session_factory,
        task.id,
        description="x",
        repo_path=Path("/nope"),
        model=object(),
    )

    async with session_factory() as verify_session:
        reloaded = await get_task(verify_session, task.id)
        assert reloaded.state == TaskState.CANCELLED.value
        assert "repo not found" in reloaded.task_context["background_execution_error"]


async def test_run_fix_and_persist_reads_the_tasks_current_state_not_always_created(
    session, engine, monkeypatch
):
    """If run_chain() already moved the task past CREATED before
    something later broke, the fallback must land on the best state legal
    from THAT state -- read fresh from the DB, not assumed to still be
    CREATED. TRIAGING -> NEEDS_HUMAN_INPUT is a real edge (Milestone 9's
    failure_streak_breaker one); FAILED isn't legal from TRIAGING, so
    NEEDS_HUMAN_INPUT is the one _failure_state's own preference table
    would pick -- asserted by calling that same real function, not by
    hardcoding an assumption about it here."""
    task = await create_task(
        session, task_type="bug_fix", task_context={"repo": "/repo/real", "prompt": "x"}
    )

    async def _fake_run_fix(session, task, **kwargs):
        await transition(session, task, TaskState.TRIAGING, actor="test", trigger="test")
        raise RuntimeError("boom after triage")

    monkeypatch.setattr(chain_module, "run_fix", _fake_run_fix)

    session_factory = make_session_factory(engine)
    await run_fix_and_persist(
        session_factory,
        task.id,
        description="x",
        repo_path=Path("/repo/real"),
        model=object(),
    )

    expected = chain_module._failure_state(TaskState.TRIAGING)
    async with session_factory() as verify_session:
        reloaded = await get_task(verify_session, task.id)
        assert reloaded.state == expected.value
        assert "boom after triage" in reloaded.task_context["background_execution_error"]


async def test_run_fix_and_persist_never_touches_an_already_terminal_task(
    session, engine, monkeypatch
):
    """run_chain() reaching a real terminal state on its own (e.g. FAILED
    after exhausted retries) and then something raises during cleanup
    afterward must not get silently overwritten to CANCELLED -- the
    chain's own honest outcome wins."""
    task = await create_task(
        session, task_type="bug_fix", task_context={"repo": "/repo/real", "prompt": "x"}
    )
    await transition(session, task, TaskState.TRIAGING, actor="test", trigger="test")
    await transition(session, task, TaskState.INVESTIGATING, actor="test", trigger="test")
    await transition(session, task, TaskState.PLANNING_FIX, actor="test", trigger="test")
    await transition(session, task, TaskState.CODING, actor="test", trigger="test")
    await transition(session, task, TaskState.TESTING, actor="test", trigger="test")
    await transition(session, task, TaskState.FAILED, actor="test", trigger="test")

    async def _fake_run_fix(session, task, **kwargs):
        raise RuntimeError("teardown blew up after the chain already finished")

    monkeypatch.setattr(chain_module, "run_fix", _fake_run_fix)

    session_factory = make_session_factory(engine)
    await run_fix_and_persist(
        session_factory,
        task.id,
        description="x",
        repo_path=Path("/repo/real"),
        model=object(),
    )

    async with session_factory() as verify_session:
        reloaded = await get_task(verify_session, task.id)
        assert reloaded.state == TaskState.FAILED.value
        assert "background_execution_error" not in (reloaded.task_context or {})


async def test_run_fix_and_persist_never_raises_even_when_the_task_no_longer_exists(engine):
    """Defensive: the background job runs after the HTTP response is
    already gone, so nothing is listening for an exception -- an
    uncaught one only produces noisy server logs (see
    api/routes/webhooks.py's own documented finding of exactly this
    class of bug). A task_id that doesn't exist must not raise."""
    session_factory = make_session_factory(engine)
    await run_fix_and_persist(
        session_factory,
        uuid.uuid4(),
        description="x",
        repo_path=Path("/repo/real"),
        model=object(),
    )
