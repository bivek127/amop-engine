"""Milestone 15 — API Layer + Telegram Bot + Web Dashboard.

Stage 1: the API layer, Section 15.1-15.3. Tiers:
  * (a) real Postgres (TEST_DATABASE_URL) + a real HTTP request cycle
    (httpx.AsyncClient over ASGITransport, in-process), with
    `get_session` overridden to the test database -- the app's own
    lifespan would otherwise connect to the real DATABASE_URL, but the
    override means no route ever actually uses that connection.

Requires: Postgres at TEST_DATABASE_URL.
"""

import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from amop.api.app import app
from amop.api.deps import get_session
from amop.database.models import MemoryItem, PullRequest, Repository, Task
from amop.database.session import init_db, make_engine, make_session_factory
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import create_task, transition

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
            text(
                "TRUNCATE memory_items, task_transitions, tasks, "
                "repositories, pull_requests RESTART IDENTITY CASCADE"
            )
        )


@pytest.fixture(autouse=True)
def _api_token(monkeypatch):
    monkeypatch.setenv("AMOP_API_TOKEN", TEST_TOKEN)


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


@pytest_asyncio.fixture
async def client(engine):
    """A real HTTP client (via ASGI transport, in-process) whose
    `get_session` dependency is overridden to the TEST database -- so
    route handlers read/write TEST_DATABASE_URL, never DATABASE_URL,
    regardless of what the app's own lifespan would otherwise connect
    to. httpx.AsyncClient + ASGITransport, not Starlette's TestClient:
    TestClient bridges to its own background thread with its own event
    loop, and an asyncpg connection created in *this* test's loop can't
    be used from that one ("attached to a different loop") -- the same
    class of pitfall Milestone 1's own `engine` fixture docstring
    already warns about, here via a different mechanism. Staying on
    httpx.AsyncClient keeps everything in one loop.
    """
    session_factory = make_session_factory(engine)

    async def _override_get_session():
        async with session_factory() as s:
            yield s

    app.dependency_overrides[get_session] = _override_get_session
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------
# Auth -- Done-When #4: every mutating endpoint rejects an
# unauthenticated/unauthorized caller
# ---------------------------------------------------------------------


async def test_get_endpoints_never_require_auth(client, session):
    """The other half of the same rule: reads stay open (CLAUDE.md's own
    "auth on all MUTATING endpoints" -- not a blanket lock)."""
    task = await create_task(session, task_type="bug_fix", task_context={})
    for path in (
        "/tasks",
        f"/tasks/{task.id}",
        f"/tasks/{task.id}/actions",
        f"/tasks/{task.id}/diff",
        "/repositories",
        "/pull-requests",
        "/memory",
    ):
        r = await client.get(path)
        assert r.status_code != 401, f"{path} unexpectedly required auth"


@pytest.mark.parametrize(
    "method,path_template,body",
    [
        ("POST", "/tasks", {"task_type": "bug_fix", "description": "x"}),
        ("POST", "/tasks/{task_id}/cancel", None),
        ("POST", "/tasks/{task_id}/approve", None),
        ("POST", "/tasks/{task_id}/reject", None),
        ("POST", "/repositories", {"repo_path": "/tmp/x"}),
        ("PATCH", "/memory/{memory_id}", {"disputed": True}),
    ],
)
async def test_every_mutating_endpoint_rejects_unauthenticated(
    client, session, method, path_template, body
):
    task = await create_task(session, task_type="bug_fix", task_context={})
    memory_row = MemoryItem(
        repo_path="/repo",
        memory_type="incident_resolution",
        content={"root_cause": "x"},
        embedding=[0.0] * 768,
    )
    session.add(memory_row)
    await session.commit()

    path = path_template.format(task_id=task.id, memory_id=memory_row.id)
    r = await client.request(method, path, json=body)

    assert r.status_code == 401, f"{method} {path} did not reject unauthenticated"
    assert r.json()["error_code"] == "UNAUTHORIZED"


async def test_mutating_endpoint_rejects_wrong_token(client, session):
    task = await create_task(session, task_type="bug_fix", task_context={})
    r = await client.post(
        f"/tasks/{task.id}/cancel", headers={"Authorization": "Bearer wrong-token"}
    )
    assert r.status_code == 401


# ---------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------


async def test_create_task_round_trips_via_get(client, auth_headers):
    r = await client.post(
        "/tasks",
        json={"task_type": "bug_fix", "description": "the login page 500s"},
        headers=auth_headers,
    )
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["state"] == "CREATED"
    assert created["task_context"]["prompt"] == "the login page 500s"

    r2 = await client.get(f"/tasks/{created['id']}")
    assert r2.status_code == 200
    assert r2.json()["id"] == created["id"]


async def test_get_task_404_uses_the_standard_envelope(client):
    r = await client.get(f"/tasks/{uuid.uuid4()}")
    assert r.status_code == 404
    body = r.json()
    assert set(body.keys()) == {"error_code", "message", "detail"}
    assert body["error_code"] == "TASK_NOT_FOUND"


async def test_list_tasks_filters_by_state(client, session):
    await create_task(session, task_type="bug_fix", task_context={})
    resolved = await create_task(session, task_type="bug_fix", task_context={})
    await transition(session, resolved, TaskState.CANCELLED)

    r = await client.get("/tasks", params={"state": "CANCELLED"})
    assert r.status_code == 200
    states = {t["state"] for t in r.json()}
    assert states == {"CANCELLED"}


async def test_cancel_drives_a_real_transition(client, session, auth_headers):
    task = await create_task(session, task_type="bug_fix", task_context={})
    r = await client.post(f"/tasks/{task.id}/cancel", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["state"] == "CANCELLED"

    # Not just the response -- the row itself, re-fetched independently.
    r2 = await client.get(f"/tasks/{task.id}")
    assert r2.json()["state"] == "CANCELLED"


async def test_approve_and_reject_drive_the_real_state_machine(
    client, session, auth_headers
):
    """The first real caller of WAITING_FOR_APPROVAL -> MERGED /
    -> CANCELLED in this codebase's history -- these edges have existed
    in TRANSITIONS since Milestone 1 but nothing ever exercised them."""
    approved = await create_task(session, task_type="bug_fix", task_context={})
    for state in (
        TaskState.TRIAGING,
        TaskState.INVESTIGATING,
        TaskState.PLANNING_FIX,
        TaskState.CODING,
        TaskState.TESTING,
        TaskState.REVIEWING,
        TaskState.PR_CREATION,
        TaskState.WAITING_FOR_APPROVAL,
    ):
        await transition(session, approved, state)

    r = await client.post(f"/tasks/{approved.id}/approve", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["state"] == "MERGED"

    rejected = await create_task(session, task_type="bug_fix", task_context={})
    for state in (
        TaskState.TRIAGING,
        TaskState.INVESTIGATING,
        TaskState.PLANNING_FIX,
        TaskState.CODING,
        TaskState.TESTING,
        TaskState.REVIEWING,
        TaskState.PR_CREATION,
        TaskState.WAITING_FOR_APPROVAL,
    ):
        await transition(session, rejected, state)

    r2 = await client.post(f"/tasks/{rejected.id}/reject", headers=auth_headers)
    assert r2.status_code == 200
    assert r2.json()["state"] == "CANCELLED"


async def test_approve_on_the_wrong_state_is_a_409_not_a_500(
    client, session, auth_headers
):
    task = await create_task(session, task_type="bug_fix", task_context={})  # CREATED
    r = await client.post(f"/tasks/{task.id}/approve", headers=auth_headers)
    assert r.status_code == 409
    assert r.json()["error_code"] == "ILLEGAL_TRANSITION"


async def test_task_actions_endpoint_returns_transitions(client, session):
    task = await create_task(session, task_type="bug_fix", task_context={})
    await transition(session, task, TaskState.TRIAGING)
    await transition(session, task, TaskState.CANCELLED)

    r = await client.get(f"/tasks/{task.id}/actions")
    assert r.status_code == 200
    to_states = [row["to_state"] for row in r.json()]
    assert to_states == ["TRIAGING", "CANCELLED"]


async def test_task_diff_endpoint_reads_persisted_diff(client, session):
    task = await create_task(session, task_type="bug_fix", task_context={})
    task.task_context = {**(task.task_context or {}), "diff": "--- a\n+++ b\n"}
    session.add(task)
    await session.commit()

    r = await client.get(f"/tasks/{task.id}/diff")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert "+++ b" in body["diff"]


async def test_task_diff_endpoint_reports_unavailable_honestly(client, session):
    task = await create_task(session, task_type="bug_fix", task_context={})
    r = await client.get(f"/tasks/{task.id}/diff")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["diff"] == ""


# ---------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------


async def test_repositories_round_trip_and_reject_duplicates(client, auth_headers):
    r = await client.post(
        "/repositories",
        json={"repo_path": "/repo/one", "display_name": "One"},
        headers=auth_headers,
    )
    assert r.status_code == 201
    repo_id = r.json()["id"]

    r2 = await client.get("/repositories")
    assert any(row["id"] == repo_id for row in r2.json())

    r3 = await client.post(
        "/repositories", json={"repo_path": "/repo/one"}, headers=auth_headers
    )
    assert r3.status_code == 409
    assert r3.json()["error_code"] == "REPOSITORY_EXISTS"


async def test_create_task_with_repo_id_resolves_to_the_repo_path(
    client, session, auth_headers
):
    repo = Repository(repo_path="/repo/known")
    session.add(repo)
    await session.commit()
    await session.refresh(repo)

    r = await client.post(
        "/tasks",
        json={"task_type": "bug_fix", "repo_id": str(repo.id), "description": "x"},
        headers=auth_headers,
    )
    assert r.status_code == 201
    assert r.json()["task_context"]["repo"] == "/repo/known"


async def test_create_task_with_unknown_repo_id_is_404(client, auth_headers):
    r = await client.post(
        "/tasks",
        json={"task_type": "bug_fix", "repo_id": str(uuid.uuid4())},
        headers=auth_headers,
    )
    assert r.status_code == 404
    assert r.json()["error_code"] == "REPOSITORY_NOT_FOUND"


async def test_list_tasks_by_repo_id(client, session, auth_headers):
    repo = Repository(repo_path="/repo/scoped")
    session.add(repo)
    await session.commit()
    await session.refresh(repo)

    r = await client.post(
        "/tasks",
        json={"task_type": "bug_fix", "repo_id": str(repo.id)},
        headers=auth_headers,
    )
    scoped_id = r.json()["id"]
    await create_task(session, task_type="bug_fix", task_context={"repo": "/other"})

    r2 = await client.get("/tasks", params={"repo_id": str(repo.id)})
    ids = [t["id"] for t in r2.json()]
    assert ids == [scoped_id]


# ---------------------------------------------------------------------
# Pull requests
# ---------------------------------------------------------------------


async def test_pull_requests_list_and_filter_by_status(client, session):
    session.add_all(
        [
            PullRequest(repo_path="/r", url="https://x/1", number=1, status="open"),
            PullRequest(repo_path="/r", url="https://x/2", number=2, status="merged"),
        ]
    )
    await session.commit()

    r = await client.get("/pull-requests")
    assert len(r.json()) == 2

    r2 = await client.get("/pull-requests", params={"status": "merged"})
    assert len(r2.json()) == 1
    assert r2.json()[0]["number"] == 2


# ---------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------


async def test_memory_list_and_dispute_toggle(client, session, auth_headers):
    item = MemoryItem(
        repo_path="/repo",
        memory_type="incident_resolution",
        content={"root_cause": "off by one"},
        embedding=[0.0] * 768,
    )
    session.add(item)
    await session.commit()
    await session.refresh(item)

    r = await client.get("/memory")
    assert any(row["id"] == str(item.id) for row in r.json())

    r2 = await client.patch(f"/memory/{item.id}", json={"disputed": True}, headers=auth_headers)
    assert r2.status_code == 200
    assert r2.json()["disputed"] is True

    # Section 10.4: excluded from the memory *retrieval* consumer, but
    # never removed from this listing -- it's still there, still marked.
    r3 = await client.get("/memory")
    row = next(row for row in r3.json() if row["id"] == str(item.id))
    assert row["disputed"] is True


async def test_memory_dispute_on_unknown_id_is_404(client, auth_headers):
    r = await client.patch(
        f"/memory/{uuid.uuid4()}", json={"disputed": True}, headers=auth_headers
    )
    assert r.status_code == 404
    assert r.json()["error_code"] == "MEMORY_NOT_FOUND"


async def test_memory_filters_by_type(client, session):
    session.add_all(
        [
            MemoryItem(
                repo_path="/repo",
                memory_type="incident_resolution",
                content={},
                embedding=[0.0] * 768,
            ),
            MemoryItem(
                repo_path="/repo",
                memory_type="codebase_note",
                content={},
                embedding=[0.0] * 768,
            ),
        ]
    )
    await session.commit()

    r = await client.get("/memory", params={"type": "codebase_note"})
    assert len(r.json()) == 1
    assert r.json()[0]["memory_type"] == "codebase_note"
