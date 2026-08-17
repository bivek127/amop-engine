"""Milestone 9 — Watcher + Circuit Breakers.

Two tiers, same discipline as every prior milestone:

  * Deterministic (always on): circuit breaker queries (pure, real
    Postgres, no LLM/GitHub), dedup layers 1a/1b (real Postgres; layer 1b
    needs an embedding call, which is monkeypatched to a fake, cheap,
    deterministic vector rather than hitting real Ollama -- matching
    embed_texts()'s own documented `provider` override support), and the
    full triage_anomaly() flow driven by a scripted WatcherReport
    response instead of a real model.
  * The actual live `amop watch` run against a real GitHub repo is a CLI
    demo (CLAUDE.md's Verification & Commit section), not a pytest tier
    -- Milestone 9's own test-file spec lists exactly the five
    deterministic scenarios below, no AMOP_E2E_GITHUB-gated test.

Requires: Postgres at TEST_DATABASE_URL, no Docker/Ollama/GitHub needed.
"""

import json
import os

import pytest
import pytest_asyncio
from sqlalchemy import text

from amop.agents.handoffs import AnomalyAlert
from amop.agents.watcher import WatcherAgent
from amop.database.session import init_db, make_engine, make_session_factory
from amop.models.base import BaseLLM, ModelResponse
from amop.orchestrator import watch as watch_mod
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import create_task
from amop.safety import circuit_breakers as cb

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://localhost/amop_test"
)

_FAKE_EMBED_DIM = 8


def _fake_vector(seed: str) -> list[float]:
    """A cheap, deterministic, seed-dependent fake embedding -- distinct
    seeds give near-orthogonal vectors, identical seeds give identical
    vectors. No real Ollama call, matching embed_texts()'s own documented
    `provider` override support for tests."""
    h = abs(hash(seed))
    return [((h >> (i * 4)) % 16) / 16 for i in range(_FAKE_EMBED_DIM)]


async def _fake_embed_texts(texts: list[str]) -> list[list[float]]:
    return [_fake_vector(t) for t in texts]


class ScriptedLLM(BaseLLM):
    """Same shape as prior milestones' scripted models -- one canned
    response per call, cycling to the last if exhausted."""

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
        await conn.execute(text("TRUNCATE task_transitions, tasks RESTART IDENTITY CASCADE"))


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    """Every test in this file that reaches dedup layer 1b (semantic
    similarity) goes through this fake instead of real Ollama -- cheap,
    deterministic, no network needed."""
    monkeypatch.setattr(watch_mod, "embed_texts", _fake_embed_texts)


# ---------------------------------------------------------------------
# 1. Watcher identifies a new issue and creates a task
# ---------------------------------------------------------------------


async def test_watcher_identifies_new_issue_and_creates_task(session):
    model = ScriptedLLM(
        [
            watcher_final_answer(
                [
                    {
                        "severity": "high",
                        "summary": "crash when uploading files over 2GB",
                        "confidence": 0.85,
                        "issue_index": 1,
                    }
                ]
            )
        ]
    )
    agent = WatcherAgent(model)
    result = await agent.run("Issue 1 (#42): Crash on large upload\nDetails...")

    assert result.success, result.error
    assert len(result.handoff.alerts) == 1
    alert = result.handoff.alerts[0].model_copy(
        update={"repo": "acme/widgets", "github_issue_number": 42}
    )

    task = await watch_mod.triage_anomaly(session, alert, local_path=None, model=None)

    assert task is not None
    assert task.task_type == "bug_fix"
    assert task.task_context["repo"] == "acme/widgets"
    assert task.task_context["github_issue_number"] == 42
    assert task.task_context["source"] == "github_watcher"
    # High severity, above the default floor, no local_path given -> stops
    # at TRIAGING this milestone (see triage_anomaly's docstring).
    assert task.state == TaskState.TRIAGING.value


async def test_watcher_emits_no_alerts_for_a_batch_with_no_real_bugs():
    model = ScriptedLLM([watcher_final_answer([])])
    agent = WatcherAgent(model)
    result = await agent.run("Issue 1 (#7): How do I configure X?\nJust a question.")

    assert result.success, result.error
    assert result.handoff.alerts == []


# ---------------------------------------------------------------------
# 2. Dedup: a second poll cycle with the same issue does NOT duplicate
# ---------------------------------------------------------------------


async def test_dedup_layer_1a_prevents_exact_duplicate_on_second_poll(session):
    alert = AnomalyAlert(
        severity="high", summary="crash on large upload", confidence=0.9, issue_index=1
    ).model_copy(update={"repo": "acme/widgets", "github_issue_number": 42})

    first = await watch_mod.triage_anomaly(session, alert, local_path=None, model=None)
    assert first is not None

    # Layer 1a: the exact-match filter cli/main.py's poll loop applies
    # BEFORE an issue ever reaches Watcher's prompt again.
    existing = await watch_mod.find_existing_task_for_issue(
        session, "acme/widgets", 42
    )
    assert existing is not None
    assert existing.id == first.id


async def test_dedup_layer_1b_merges_a_semantically_similar_report(session):
    first_alert = AnomalyAlert(
        severity="high",
        summary="uploads crash for files larger than 2 gigabytes",
        confidence=0.9,
        issue_index=1,
    ).model_copy(update={"repo": "acme/widgets", "github_issue_number": 42})
    first = await watch_mod.triage_anomaly(session, first_alert, local_path=None, model=None)
    assert first.state == TaskState.TRIAGING.value

    # A DIFFERENT issue number, same underlying report -- exact match
    # (layer 1a) would miss this; semantic similarity (layer 1b) should
    # catch it. The fake embedder gives identical text identical vectors,
    # so an identical summary is a clean similarity=1.0 case.
    second_alert = first_alert.model_copy(
        update={"github_issue_number": 99, "anomaly_id": "different-id"}
    )
    second = await watch_mod.triage_anomaly(session, second_alert, local_path=None, model=None)

    assert second.state == TaskState.MERGED_INTO_EXISTING.value


async def test_dedup_never_matches_the_task_being_created_against_itself(session):
    # Regression guard for the self-similarity bug found during
    # implementation: the candidate task is committed in TRIAGING (a
    # non-terminal state) before dedup runs, so without excluding it by
    # id, every new task would match itself at similarity 1.0.
    alert = AnomalyAlert(
        severity="high", summary="unique unrelated report", confidence=0.9, issue_index=1
    ).model_copy(update={"repo": "acme/widgets", "github_issue_number": 1})
    task = await watch_mod.triage_anomaly(session, alert, local_path=None, model=None)
    assert task.state == TaskState.TRIAGING.value  # not MERGED_INTO_EXISTING


# ---------------------------------------------------------------------
# 3. Cost breaker
# ---------------------------------------------------------------------


async def test_cost_breaker_rejects_near_cap(session):
    allowed = await cb.check_cost_breaker(session, estimate_usd=1.0, cap_usd=10.0)
    assert allowed.allow

    await create_task(
        session, task_type="bug_fix", task_context={"cost_estimate_usd": 9.5}
    )
    blocked = await cb.check_cost_breaker(session, estimate_usd=1.0, cap_usd=10.0)
    assert not blocked.allow
    assert "max_cost_per_day_usd" in blocked.reason


async def test_cost_breaker_blocks_triage_anomaly_from_creating_a_task(session):
    # Seed enough prior "spend" (today's stored per-task estimates) that
    # the real, unpatched MAX_COST_PER_DAY_USD default is already
    # exhausted -- monkeypatching the module constant directly doesn't
    # work here, since check_cost_breaker's cap_usd default is bound to
    # the constant's value at function-definition time (Python's normal
    # default-argument-evaluated-once semantics), not looked up fresh
    # per call.
    await create_task(
        session,
        task_type="bug_fix",
        task_context={"cost_estimate_usd": cb.MAX_COST_PER_DAY_USD},
    )
    alert = AnomalyAlert(
        severity="high", summary="blocked by cost cap", confidence=0.9, issue_index=1
    ).model_copy(update={"repo": "acme/widgets", "github_issue_number": 5})

    task = await watch_mod.triage_anomaly(session, alert, local_path=None, model=None)
    assert task is None  # no task created, not an error -- see docstring


# ---------------------------------------------------------------------
# 4. Anomaly-rate breaker: a burst produces one meta-alert, not N tasks
# ---------------------------------------------------------------------


async def test_anomaly_rate_breaker_trips_after_burst(session):
    for i in range(cb.MAX_ANOMALIES_PER_HOUR_PER_REPO):
        await create_task(
            session,
            task_type="bug_fix",
            task_context={"repo": "acme/widgets", "source": "github_watcher"},
        )

    result = await cb.check_anomaly_rate(session, "acme/widgets")
    assert not result.allow
    assert "detection rate abnormal" in result.reason


async def test_anomaly_rate_breaker_does_not_count_other_repos_or_sources(session):
    for i in range(cb.MAX_ANOMALIES_PER_HOUR_PER_REPO):
        await create_task(
            session,
            task_type="bug_fix",
            task_context={"repo": "acme/OTHER-repo", "source": "github_watcher"},
        )
        await create_task(
            session,
            task_type="bug_fix",
            task_context={"repo": "acme/widgets", "source": "cli"},  # not watcher-sourced
        )

    result = await cb.check_anomaly_rate(session, "acme/widgets")
    assert result.allow


# ---------------------------------------------------------------------
# 5. Failure-streak breaker: stops auto-retry after N consecutive FAILED
# ---------------------------------------------------------------------


async def test_failure_streak_breaker_trips_after_n_consecutive_failures(session):
    for i in range(cb.FAILURE_STREAK_THRESHOLD):
        task = await create_task(
            session,
            task_type="bug_fix",
            task_context={"repo": "acme/widgets", "github_issue_number": 77},
        )
        task.state = TaskState.FAILED.value
        session.add(task)
    await session.commit()

    result = await cb.check_failure_streak(session, "acme/widgets", 77)
    assert not result.allow
    assert "failure_streak_breaker" in result.reason


async def test_failure_streak_breaker_ignores_a_different_issue_on_the_same_repo(session):
    for i in range(cb.FAILURE_STREAK_THRESHOLD):
        task = await create_task(
            session,
            task_type="bug_fix",
            task_context={"repo": "acme/widgets", "github_issue_number": 77},
        )
        task.state = TaskState.FAILED.value
        session.add(task)
    await session.commit()

    # A different issue on the SAME repo must not be blocked -- spec
    # scopes this per-incident (repo, issue_number), not per-repo (see
    # circuit_breakers.check_failure_streak's docstring for why).
    result = await cb.check_failure_streak(session, "acme/widgets", 88)
    assert result.allow


async def test_failure_streak_breaker_routes_triage_anomaly_to_needs_human_input(session):
    for i in range(cb.FAILURE_STREAK_THRESHOLD):
        task = await create_task(
            session,
            task_type="bug_fix",
            task_context={"repo": "acme/widgets", "github_issue_number": 77},
        )
        task.state = TaskState.FAILED.value
        session.add(task)
    await session.commit()

    alert = AnomalyAlert(
        severity="high", summary="same bug reported again", confidence=0.9, issue_index=1
    ).model_copy(update={"repo": "acme/widgets", "github_issue_number": 77})
    task = await watch_mod.triage_anomaly(session, alert, local_path=None, model=None)

    assert task.state == TaskState.NEEDS_HUMAN_INPUT.value


# ---------------------------------------------------------------------
# Severity floor (activates the second previously-dead TRIAGING edge)
# ---------------------------------------------------------------------


async def test_low_severity_below_floor_cancels_without_running_the_chain(session):
    alert = AnomalyAlert(
        severity="low", summary="minor cosmetic issue", confidence=0.6, issue_index=1
    ).model_copy(update={"repo": "acme/widgets", "github_issue_number": 3})

    task = await watch_mod.triage_anomaly(session, alert, local_path=None, model=None)
    assert task.state == TaskState.CANCELLED.value
