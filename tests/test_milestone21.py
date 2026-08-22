"""Milestone 21 — GitHub Webhook Ingestion.

Signature verification is the security-critical piece (spec 8.5/3.3) and
gets treated that way: every processing path is proven unreachable
without a valid signature, not merely "usually correct."

Idempotency is Postgres-only (`processed_events`), not Redis Streams --
CLAUDE.md's own explicit scope decision, reaffirmed in ROADMAP.md, not
relitigated here.

The webhook route's own background job (`_process_issue_opened`) reuses
Milestone 9's real functions unchanged (`find_existing_task_for_issue`,
`check_anomaly_rate`, `WatcherAgent`, `triage_anomaly`) -- the "reuse
confirmed directly" tests below prove that by triggering the SAME real
effects (an existing task blocks a duplicate; a tripped breaker blocks a
new one) through the webhook path, not by re-deriving Milestone 9's own
coverage from scratch.

Requires: Postgres at TEST_DATABASE_URL. No live Ollama needed --
`WatcherAgent`'s model is a scripted stub throughout, same pattern
Milestone 9's own tests use.
"""

import json

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select, text

from amop.api.app import app
from amop.api.deps import configure_session_factory, get_session
from amop.api.routes import webhooks as webhooks_mod
from amop.api.webhook_auth import compute_signature, verify_signature
from amop.database.models import ProcessedEvent, Repository, Task
from amop.database.session import init_db, make_engine, make_session_factory
from amop.models.base import BaseLLM, ModelResponse
from amop.orchestrator.task import create_task

TEST_DATABASE_URL = "postgresql+asyncpg://localhost/amop_test"
GLOBAL_SECRET = "global-webhook-secret"
REPO_SECRET = "repo-specific-secret"


class ScriptedLLM(BaseLLM):
    """Same shape as every prior milestone's scripted model (Milestone
    9's own `ScriptedLLM`, duplicated per this project's established
    per-file convention rather than cross-imported between test
    modules)."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, messages, tools=None) -> ModelResponse:
        content = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return ModelResponse(content=content, model="scripted")

    async def embed(self, texts):
        raise NotImplementedError


def watcher_final_answer(alerts: list[dict]) -> str:
    return json.dumps({"final_answer": {"alerts": alerts}})


def issue_opened_payload(
    repo_full_name: str = "octo/demo",
    number: int = 101,
    title: str = "Crash on startup",
    body: str = "The app crashes immediately after launch.",
    action: str = "opened",
) -> bytes:
    payload = {
        "action": action,
        "issue": {"number": number, "title": title, "body": body},
        "repository": {"full_name": repo_full_name},
    }
    return json.dumps(payload).encode("utf-8")


def signed_headers(raw_body: bytes, secret: str, delivery_id: str, event: str = "issues") -> dict:
    return {
        "X-Hub-Signature-256": compute_signature(raw_body, secret),
        "X-GitHub-Delivery": delivery_id,
        "X-GitHub-Event": event,
        "Content-Type": "application/json",
    }


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
                "TRUNCATE task_transitions, tasks, repositories, "
                "processed_events RESTART IDENTITY CASCADE"
            )
        )


@pytest.fixture(autouse=True)
def _global_secret(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", GLOBAL_SECRET)


@pytest_asyncio.fixture
async def client(engine):
    """httpx.AsyncClient over ASGITransport, in-process -- same pattern
    every prior API test file uses (see test_milestone15.py's `client`
    docstring for why, not Starlette's TestClient).

    ALSO configures deps.py's module-level session factory (via
    `configure_session_factory`), not just the request-scoped
    `get_session` override: the webhook's background job runs AFTER the
    HTTP response and reads that module-level factory directly (it isn't
    part of FastAPI's per-request dependency-injection cycle, so
    `app.dependency_overrides` alone doesn't reach it). Starlette runs
    `BackgroundTasks` as part of the same request-handling coroutine
    before control returns to the caller, so by the time `await
    client.post(...)` returns below, the background job has already run
    to completion -- no polling/sleeping needed in any test here.
    """
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


@pytest.fixture(autouse=True)
def _scripted_watcher(monkeypatch):
    """The webhook route always constructs a real OllamaProvider; swap
    it for a scripted one so no test here needs live Ollama. One alert
    for issue_index=1 (the only valid index for a length-1 batch) by
    default -- tests that need a different script override this fixture
    inline by monkeypatching webhooks_mod.OllamaProvider again."""

    def _make(model):
        return ScriptedLLM(
            [
                watcher_final_answer(
                    [
                        {
                            "severity": "high",
                            "summary": "crash on startup",
                            "confidence": 0.85,
                            "issue_index": 1,
                        }
                    ]
                )
            ]
        )

    monkeypatch.setattr(webhooks_mod, "OllamaProvider", _make)


async def _register(session, repo_path, url, **kwargs) -> Repository:
    row = Repository(repo_path=str(repo_path), url=url, **kwargs)
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


# ---------------------------------------------------------------------
# Pure signature verification -- no HTTP, no DB.
# ---------------------------------------------------------------------


def test_verify_signature_accepts_a_correctly_signed_body():
    body = b'{"a": 1}'
    sig = compute_signature(body, "s3cret")
    assert verify_signature(body, sig, "s3cret") is True


def test_verify_signature_rejects_wrong_secret():
    body = b'{"a": 1}'
    sig = compute_signature(body, "s3cret")
    assert verify_signature(body, sig, "wrong") is False


def test_verify_signature_rejects_a_tampered_body():
    body = b'{"a": 1}'
    sig = compute_signature(body, "s3cret")
    assert verify_signature(body + b"x", sig, "s3cret") is False


@pytest.mark.parametrize(
    "header,secret",
    [(None, "s3cret"), ("deadbeef", "s3cret"), ("sha256=deadbeef", "s3cret"), ("sha256=x", "")],
)
def test_verify_signature_rejects_malformed_or_missing_input(header, secret):
    assert verify_signature(b"{}", header, secret) is False


# ---------------------------------------------------------------------
# Signature verification, over real HTTP, through the real route.
# ---------------------------------------------------------------------


async def test_valid_signature_and_ping_is_accepted_without_processing(client, session):
    await _register(session, "/tmp/x", "octo/demo", webhook_secret=REPO_SECRET)
    body = issue_opened_payload(repo_full_name="octo/demo")
    headers = signed_headers(body, REPO_SECRET, "delivery-ping-1", event="ping")

    r = await client.post("/webhooks/github", content=body, headers=headers)

    assert r.status_code == 200
    assert r.json() == {"status": "pong"}
    rows = (await session.execute(select(ProcessedEvent))).scalars().all()
    assert rows == [], "ping must not be recorded as a processed delivery"


async def test_missing_signature_is_rejected_before_any_processing(client, session):
    await _register(session, "/tmp/x", "octo/demo", webhook_secret=REPO_SECRET)
    body = issue_opened_payload(repo_full_name="octo/demo")

    r = await client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-GitHub-Delivery": "delivery-nosig",
            "X-GitHub-Event": "issues",
            "Content-Type": "application/json",
        },
    )

    assert r.status_code == 401
    assert r.json()["error_code"] == "INVALID_SIGNATURE"
    assert (await session.execute(select(ProcessedEvent))).scalars().all() == []
    assert (await session.execute(select(Task))).scalars().all() == []


async def test_wrong_signature_is_rejected_before_any_processing(client, session):
    await _register(session, "/tmp/x", "octo/demo", webhook_secret=REPO_SECRET)
    body = issue_opened_payload(repo_full_name="octo/demo")
    headers = signed_headers(body, "not-the-real-secret", "delivery-wrongsig")

    r = await client.post("/webhooks/github", content=body, headers=headers)

    assert r.status_code == 401
    assert r.json()["error_code"] == "INVALID_SIGNATURE"
    assert (await session.execute(select(ProcessedEvent))).scalars().all() == []
    assert (await session.execute(select(Task))).scalars().all() == []


async def test_a_repo_with_no_secret_configured_anywhere_rejects_rather_than_accepting(
    client, session, monkeypatch
):
    """Never fail open on an auth gate: no per-repo secret AND no global
    env var means every request to this route is rejected, even one an
    attacker signs with a guessed/empty value."""
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
    await _register(session, "/tmp/x", "octo/demo")  # no webhook_secret
    body = issue_opened_payload(repo_full_name="octo/demo")
    headers = signed_headers(body, "anything-at-all", "delivery-nosecret")

    r = await client.post("/webhooks/github", content=body, headers=headers)

    assert r.status_code == 401
    assert r.json()["error_code"] == "INVALID_SIGNATURE"


async def test_per_repo_secret_overrides_the_global_one(client, session):
    """A request signed with the GLOBAL secret must be rejected for a
    repo that has its OWN secret registered -- proves the override is
    actually used, not merely accepted as a fallback that never fires."""
    await _register(session, "/tmp/x", "octo/demo", webhook_secret=REPO_SECRET)
    body = issue_opened_payload(repo_full_name="octo/demo")
    headers = signed_headers(body, GLOBAL_SECRET, "delivery-globalsig")

    r = await client.post("/webhooks/github", content=body, headers=headers)

    assert r.status_code == 401


async def test_global_secret_works_when_repo_has_no_override(client, session):
    await _register(session, "/tmp/x", "octo/demo")  # no webhook_secret -> falls back
    body = issue_opened_payload(repo_full_name="octo/demo")
    headers = signed_headers(body, GLOBAL_SECRET, "delivery-globalok", event="ping")

    r = await client.post("/webhooks/github", content=body, headers=headers)

    assert r.status_code == 200
    assert r.json() == {"status": "pong"}


# ---------------------------------------------------------------------
# Event/action filtering -- only issues.opened is in scope.
# ---------------------------------------------------------------------


async def test_non_opened_issue_action_is_ignored(client, session):
    await _register(session, "/tmp/x", "octo/demo", webhook_secret=REPO_SECRET)
    body = issue_opened_payload(repo_full_name="octo/demo", action="closed")
    headers = signed_headers(body, REPO_SECRET, "delivery-closed")

    r = await client.post("/webhooks/github", content=body, headers=headers)

    assert r.status_code == 200
    assert r.json()["status"] == "ignored"
    assert (await session.execute(select(Task))).scalars().all() == []


async def test_an_unregistered_repo_is_acked_but_not_processed(client, session):
    body = issue_opened_payload(repo_full_name="octo/never-registered")
    headers = signed_headers(body, GLOBAL_SECRET, "delivery-unregistered")

    r = await client.post("/webhooks/github", content=body, headers=headers)

    assert r.status_code == 200
    assert r.json() == {"status": "accepted", "processed": False, "reason": "no local repo_path registered"}


# ---------------------------------------------------------------------
# Delivery-ID idempotency -- the real point of processed_events.
# ---------------------------------------------------------------------


async def test_the_same_delivery_sent_twice_produces_exactly_one_task(client, session, tmp_path):
    """`tmp_path` is an empty directory, not a real git checkout, so the
    background job's own `run_fix()` hand-off fails at materialization
    -- deliberately: the task-creation half of triage_anomaly() (which
    IS what this test is about) runs before that hand-off, so a real
    Task row lands at TRIAGING either way, and the newly-added try/except
    around the background job (found live while building this test)
    means that failure is logged, not raised into the response cycle.
    Full chain success is exercised live (ngrok demo), not here -- same
    tiering Milestone 9's own tests use (local_path=None in every fast
    test, live-only for the full run_fix() path)."""
    await _register(session, tmp_path, "octo/demo", webhook_secret=REPO_SECRET)
    body = issue_opened_payload(repo_full_name="octo/demo", number=55)
    headers = signed_headers(body, REPO_SECRET, "delivery-dup-1")

    r1 = await client.post("/webhooks/github", content=body, headers=headers)
    assert r1.status_code == 200
    assert r1.json() == {"status": "accepted", "processed": True}

    r2 = await client.post("/webhooks/github", content=body, headers=headers)
    assert r2.status_code == 200, "a harmless retry must never look like a failed delivery to GitHub"
    assert r2.json()["processed"] is False
    assert r2.json()["reason"] == "duplicate delivery"

    events = (await session.execute(select(ProcessedEvent))).scalars().all()
    assert len(events) == 1

    tasks = (await session.execute(select(Task))).scalars().all()
    assert len(tasks) == 1
    assert tasks[0].task_context["github_issue_number"] == 55
    assert tasks[0].state == "TRIAGING"


async def test_a_real_task_is_created_end_to_end(client, session, tmp_path):
    """"End to end" for this test's own scope: webhook -> real HMAC
    verification -> real idempotency insert -> real background job ->
    real WatcherAgent classification (scripted model, real BaseAgent
    loop) -> real triage_anomaly() -> a real Task row in Postgres. Only
    the final run_fix() hand-off is out of scope here (needs a real git
    checkout + live Ollama) -- see the previous test's docstring."""
    await _register(session, tmp_path, "octo/demo", webhook_secret=REPO_SECRET)
    body = issue_opened_payload(repo_full_name="octo/demo", number=7, title="Crash on startup")
    headers = signed_headers(body, REPO_SECRET, "delivery-e2e")

    r = await client.post("/webhooks/github", content=body, headers=headers)
    assert r.status_code == 200
    assert r.json() == {"status": "accepted", "processed": True}

    tasks = (await session.execute(select(Task))).scalars().all()
    assert len(tasks) == 1
    task = tasks[0]
    assert task.task_context["source"] == "github_watcher"
    assert task.task_context["repo"] == "octo/demo"
    assert task.task_context["github_issue_number"] == 7
    assert task.state == "TRIAGING"


# ---------------------------------------------------------------------
# "Reuse confirmed directly": the SAME Milestone 9 functions fire.
# ---------------------------------------------------------------------


async def test_an_existing_task_for_the_issue_blocks_a_duplicate_via_dedup(
    client, session, tmp_path
):
    await _register(session, tmp_path, "octo/demo", webhook_secret=REPO_SECRET)
    await create_task(
        session,
        task_type="bug_fix",
        task_context={"source": "github_watcher", "repo": "octo/demo", "github_issue_number": 9},
    )
    body = issue_opened_payload(repo_full_name="octo/demo", number=9)
    headers = signed_headers(body, REPO_SECRET, "delivery-alreadyhastask")

    r = await client.post("/webhooks/github", content=body, headers=headers)
    assert r.status_code == 200  # ack'd -- the delivery itself succeeded

    tasks = (await session.execute(select(Task))).scalars().all()
    assert len(tasks) == 1, "find_existing_task_for_issue should have blocked a second task"


async def test_a_tripped_anomaly_rate_breaker_blocks_a_new_task(client, session, tmp_path):
    from amop.safety.circuit_breakers import MAX_ANOMALIES_PER_HOUR_PER_REPO

    await _register(session, tmp_path, "octo/demo", webhook_secret=REPO_SECRET)
    for i in range(MAX_ANOMALIES_PER_HOUR_PER_REPO):
        await create_task(
            session,
            task_type="bug_fix",
            task_context={
                "source": "github_watcher", "repo": "octo/demo", "github_issue_number": 1000 + i,
            },
        )

    body = issue_opened_payload(repo_full_name="octo/demo", number=9999)
    headers = signed_headers(body, REPO_SECRET, "delivery-ratelimited")

    r = await client.post("/webhooks/github", content=body, headers=headers)
    assert r.status_code == 200  # the delivery is still acked -- the breaker isn't an HTTP error

    tasks = (await session.execute(select(Task))).scalars().all()
    assert len(tasks) == MAX_ANOMALIES_PER_HOUR_PER_REPO, (
        "check_anomaly_rate should have blocked the new task"
    )
