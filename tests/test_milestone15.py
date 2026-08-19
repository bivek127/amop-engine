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
from sqlalchemy import select, text

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


# ---------------------------------------------------------------------
# Stage 2 — Telegram Bot (Section 16.1)
#
# The bot is an API client (see bot.py's own docstring), so these tests
# route its `_api_get`/`_api_post` through the SAME in-process ASGI
# client already built for Stage 1 -- a real request cycle through the
# real route handlers, just without an actual TCP socket, reusing the
# `client`/`engine` fixtures rather than standing up a second app.
# ---------------------------------------------------------------------

from unittest.mock import AsyncMock, MagicMock

from amop.interfaces.telegram_bot import bot as bot_module

_APPROVE_PATH = (
    TaskState.TRIAGING,
    TaskState.INVESTIGATING,
    TaskState.PLANNING_FIX,
    TaskState.CODING,
    TaskState.TESTING,
    TaskState.REVIEWING,
    TaskState.PR_CREATION,
    TaskState.WAITING_FOR_APPROVAL,
)


@pytest.fixture
def patch_bot_api(monkeypatch, client):
    """Route the bot's API calls through the in-process test client
    instead of a real socket to AMOP_API_BASE_URL."""

    async def _get(path, params=None):
        return await client.get(path, params=params)

    async def _post(path, json=None):
        return await client.post(path, json=json, headers=bot_module._auth_headers())

    monkeypatch.setattr(bot_module, "_api_get", _get)
    monkeypatch.setattr(bot_module, "_api_post", _post)
    monkeypatch.setattr(bot_module, "API_TOKEN", TEST_TOKEN)
    monkeypatch.setenv("AMOP_TELEGRAM_ALLOWED_USER_IDS", "42")
    return bot_module


def _fake_update(user_id: int | None):
    update = MagicMock()
    if user_id is None:
        update.effective_user = None
    else:
        update.effective_user.id = user_id
    update.message.reply_text = AsyncMock()
    return update


def _fake_context(args=None):
    context = MagicMock()
    context.args = args or []
    context.bot.send_message = AsyncMock()
    context.application.bot_data = {}
    return context


def _reply_texts(update) -> list[str]:
    return [c.args[0] for c in update.message.reply_text.call_args_list]


# --- classify_intent: pure, no I/O -------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("the login page returns a 500 error on submit", "bug_fix"),
        ("please update the vulnerable requests dependency", "dependency_update"),
        ("this endpoint is really slow, please optimize it", "optimization"),
        ("bump the outdated CVE package", "dependency_update"),
    ],
)
def test_classify_intent_routes_to_the_right_task_type(text, expected):
    assert bot_module.classify_intent(text) == expected


@pytest.mark.parametrize("text", ["help", "fix it", "please"])
def test_classify_intent_returns_none_for_too_little_signal(text):
    assert bot_module.classify_intent(text) is None


def test_classify_intent_returns_none_on_conflicting_signals():
    # A genuine conflict -- can't be a dependency bump AND a performance
    # fix at once -- must ask, not guess between them.
    assert (
        bot_module.classify_intent("update the outdated dependency to optimize speed")
        is None
    )


def test_allowed_user_ids_parses_the_env_var(monkeypatch):
    monkeypatch.setenv("AMOP_TELEGRAM_ALLOWED_USER_IDS", "1, 2,3")
    assert bot_module.allowed_user_ids() == {1, 2, 3}


def test_allowed_user_ids_empty_when_unset(monkeypatch):
    monkeypatch.delenv("AMOP_TELEGRAM_ALLOWED_USER_IDS", raising=False)
    assert bot_module.allowed_user_ids() == set()


# --- /status -------------------------------------------------------------


async def test_status_command_rejects_an_unlisted_user(patch_bot_api):
    update = _fake_update(user_id=999)  # not in the "42" allowlist
    await bot_module.status_command(update, _fake_context())
    assert "not authorized" in _reply_texts(update)[0].lower()


async def test_status_command_lists_active_tasks(patch_bot_api, session):
    task = await create_task(session, task_type="bug_fix", task_context={})
    await transition(session, task, TaskState.TRIAGING)

    update = _fake_update(user_id=42)
    await bot_module.status_command(update, _fake_context())

    text = _reply_texts(update)[0]
    assert str(task.id)[:8] in text
    assert "TRIAGING" in text
    assert "In progress" in text  # grouped under the right header
    # Regression test for a second real live bug: the group header's own
    # "(N)" count used literal, un-escaped parens -- MarkdownV2 rejects
    # a bare '(' the same as the no-description case, so this shipped
    # broken even after that first fix landed.
    assert "\\(1\\)" in text
    assert "(1)" not in text


async def test_status_command_reports_when_there_are_no_tasks_at_all(patch_bot_api, session):
    update = _fake_update(user_id=42)
    await bot_module.status_command(update, _fake_context())

    assert "no tasks yet" in _reply_texts(update)[0].lower()


async def test_status_command_groups_a_terminal_task_under_done(patch_bot_api, session):
    task = await create_task(session, task_type="bug_fix", task_context={})
    await transition(session, task, TaskState.CANCELLED)

    update = _fake_update(user_id=42)
    await bot_module.status_command(update, _fake_context())

    text = _reply_texts(update)[0]
    assert "Done" in text
    assert "CANCELLED" in text
    assert "In progress" not in text  # empty groups are omitted, not shown empty


async def test_status_command_escapes_markdown_for_a_task_with_no_description(
    patch_bot_api, session
):
    task = await create_task(session, task_type="bug_fix", task_context={})
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
        await transition(session, task, state)

    update = _fake_update(user_id=42)
    await bot_module.status_command(update, _fake_context())

    text = _reply_texts(update)[0]
    # Regression test for a real live bug: an earlier version emitted a
    # literal, un-escaped "(no description)" -- MarkdownV2 rejects a
    # bare '(', so Telegram's sendMessage call itself failed with a 400
    # and the message never arrived at all (silence, not a bad render).
    assert "(no description)" not in text
    assert "no description" in text
    kwargs = update.message.reply_text.call_args.kwargs
    assert kwargs.get("parse_mode") == bot_module.ParseMode.MARKDOWN_V2


async def test_status_command_escapes_the_overflow_line_past_the_cap(patch_bot_api, session):
    for _ in range(12):
        task = await create_task(session, task_type="bug_fix", task_context={})
        await transition(session, task, TaskState.TRIAGING)

    update = _fake_update(user_id=42)
    await bot_module.status_command(update, _fake_context())

    text = _reply_texts(update)[0]
    # "..." is three literal '.' characters -- also MarkdownV2-reserved,
    # same class of bug as the no-description case above.
    assert bot_module._escape_markdown_v2("...and 2 more") in text


async def test_status_command_shows_a_description_for_waiting_for_approval_tasks(
    patch_bot_api, session
):
    task = await create_task(
        session,
        task_type="bug_fix",
        task_context={"prompt": "the login page 500s when submitting"},
    )
    await transition(session, task, TaskState.TRIAGING)
    await transition(session, task, TaskState.INVESTIGATING)
    await transition(session, task, TaskState.PLANNING_FIX)
    await transition(session, task, TaskState.CODING)
    await transition(session, task, TaskState.TESTING)
    await transition(session, task, TaskState.REVIEWING)
    await transition(session, task, TaskState.PR_CREATION)
    await transition(session, task, TaskState.WAITING_FOR_APPROVAL)

    update = _fake_update(user_id=42)
    await bot_module.status_command(update, _fake_context())

    text = _reply_texts(update)[0]
    assert "Waiting for your approval" in text
    assert "the login page 500s when submitting" in text


# --- /ask ------------------------------------------------------------------


async def test_ask_command_creates_a_real_task(patch_bot_api, session):
    update = _fake_update(user_id=42)
    context = _fake_context(args=["the", "checkout", "page", "500s", "on", "submit"])

    await bot_module.ask_command(update, context)

    text = _reply_texts(update)[0]
    assert "bug_fix" in text
    rows = (await session.execute(select(Task))).scalars().all()
    assert len(rows) == 1
    assert rows[0].task_context["prompt"] == "the checkout page 500s on submit"


async def test_ask_command_asks_for_clarification_instead_of_guessing(
    patch_bot_api, session
):
    update = _fake_update(user_id=42)
    context = _fake_context(args=["please", "help"])

    await bot_module.ask_command(update, context)

    assert "not sure" in _reply_texts(update)[0].lower()
    rows = (await session.execute(select(Task))).scalars().all()
    assert rows == []  # no guessed task created


async def test_ask_command_rejects_an_unlisted_user(patch_bot_api, session):
    update = _fake_update(user_id=999)
    context = _fake_context(args=["fix", "the", "thing"])

    await bot_module.ask_command(update, context)

    assert "not authorized" in _reply_texts(update)[0].lower()
    rows = (await session.execute(select(Task))).scalars().all()
    assert rows == []


# --- /approve, /reject -------------------------------------------------


async def test_approve_command_drives_a_real_transition(patch_bot_api, session):
    task = await create_task(session, task_type="bug_fix", task_context={})
    for state in _APPROVE_PATH:
        await transition(session, task, state)

    update = _fake_update(user_id=42)
    context = _fake_context(args=[str(task.id)])
    await bot_module.approve_command(update, context)

    assert "MERGED" in _reply_texts(update)[0]
    await session.refresh(task)
    assert task.state == "MERGED"


async def test_reject_command_drives_a_real_transition(patch_bot_api, session):
    task = await create_task(session, task_type="bug_fix", task_context={})
    for state in _APPROVE_PATH:
        await transition(session, task, state)

    update = _fake_update(user_id=42)
    context = _fake_context(args=[str(task.id)])
    await bot_module.reject_command(update, context)

    assert "CANCELLED" in _reply_texts(update)[0]


async def test_approve_command_on_wrong_state_replies_clearly(patch_bot_api, session):
    task = await create_task(session, task_type="bug_fix", task_context={})  # CREATED
    update = _fake_update(user_id=42)
    context = _fake_context(args=[str(task.id)])

    await bot_module.approve_command(update, context)

    assert "can't approve" in _reply_texts(update)[0].lower()


async def test_approve_command_with_no_task_id_shows_usage(patch_bot_api):
    update = _fake_update(user_id=42)
    await bot_module.approve_command(update, _fake_context(args=[]))
    assert "usage" in _reply_texts(update)[0].lower()


# --- /report -- formatting only, no real Ollama call in the default suite --


async def test_report_command_formats_a_successful_report(monkeypatch, patch_bot_api):
    fake_response = httpx.Response(
        200,
        json={
            "period_start": "2026-01-01T00:00:00+00:00",
            "period_end": "2026-01-08T00:00:00+00:00",
            "tasks_resolved": 3,
            "prs_opened": 1,
            "prs_merged": 0,
            "dependencies_updated": 2,
            "top_issues": ["the checkout bug recurred", "CSV export still drops rows"],
        },
    )

    async def _fake_post(path, json=None):
        assert path == "/reports"
        return fake_response

    monkeypatch.setattr(bot_module, "_api_post", _fake_post)

    update = _fake_update(user_id=42)
    await bot_module.report_command(update, _fake_context())

    texts = _reply_texts(update)
    assert any("checkout bug recurred" in t for t in texts)
    assert any("CSV export" in t for t in texts)


async def test_report_command_rejects_an_unlisted_user(patch_bot_api):
    update = _fake_update(user_id=999)
    await bot_module.report_command(update, _fake_context())
    assert "not authorized" in _reply_texts(update)[0].lower()


# --- proactive notifications ------------------------------------------


async def test_notify_job_does_not_spam_tasks_already_waiting_at_startup(
    patch_bot_api, session
):
    """The priming tick must never treat everything already sitting in
    WAITING_FOR_APPROVAL/FAILED as a fresh arrival."""
    existing = await create_task(session, task_type="bug_fix", task_context={})
    for state in _APPROVE_PATH:
        await transition(session, existing, state)

    context = _fake_context()
    await bot_module._notify_job(context)

    context.bot.send_message.assert_not_called()


async def test_notify_job_notifies_only_on_new_arrivals(patch_bot_api, session):
    context = _fake_context()
    await bot_module._notify_job(context)  # primes on an empty DB

    new_task = await create_task(session, task_type="bug_fix", task_context={})
    for state in _APPROVE_PATH:
        await transition(session, new_task, state)

    await bot_module._notify_job(context)

    context.bot.send_message.assert_called_once()
    kwargs = context.bot.send_message.call_args.kwargs
    assert kwargs["chat_id"] == 42
    assert str(new_task.id)[:8] in kwargs["text"]


# ---------------------------------------------------------------------
# Stage 3 -- Web dashboard: session-cookie login (same shared
# AMOP_API_TOKEN every other interface holds), task list grouped by
# state, task detail with a diff viewer.
# ---------------------------------------------------------------------


async def test_web_login_page_loads(client):
    r = await client.get("/web/login")
    assert r.status_code == 200
    assert "form" in r.text.lower()


async def test_web_login_wrong_token_redirects_with_error(client):
    r = await client.post("/web/login", data={"token": "wrong"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/web/login?error=1"
    assert "amop_session" not in client.cookies


async def test_web_login_correct_token_sets_cookie_and_redirects(client):
    r = await client.post(
        "/web/login", data={"token": TEST_TOKEN}, follow_redirects=False
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/web/tasks"
    assert "amop_session" in r.cookies


async def test_web_tasks_requires_a_session(client):
    r = await client.get("/web/tasks", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/web/login"


async def test_web_tasks_lists_a_real_task_grouped_by_state(client, session):
    task = await create_task(
        session, task_type="bug_fix", task_context={"prompt": "checkout page 500s"}
    )
    await transition(session, task, TaskState.TRIAGING)

    await client.post("/web/login", data={"token": TEST_TOKEN})
    r = await client.get("/web/tasks")

    assert r.status_code == 200
    assert "TRIAGING" in r.text
    assert str(task.id)[:8] in r.text
    assert "checkout page 500s" in r.text


async def test_web_task_detail_shows_the_diff(client, session):
    task = await create_task(
        session,
        task_type="bug_fix",
        task_context={"prompt": "fix it", "diff": "--- a/x.py\n+++ b/x.py\n@@\n-old\n+new\n"},
    )

    await client.post("/web/login", data={"token": TEST_TOKEN})
    r = await client.get(f"/web/tasks/{task.id}")

    assert r.status_code == 200
    assert "fix it" in r.text
    assert "-old" in r.text
    assert "+new" in r.text


async def test_web_task_detail_404s_for_an_unknown_id(client):
    await client.post("/web/login", data={"token": TEST_TOKEN})
    r = await client.get(f"/web/tasks/{uuid.uuid4()}")
    assert r.status_code == 404
