"""Milestone 14 — Remaining Phase 2 Agents + Long-Term Memory.

Tiers, same discipline as every prior milestone:
  * (a) pure, always-on, no I/O -- summary construction, what gets
    embedded, the write-state set.
  * (b) real Postgres (TEST_DATABASE_URL) with a deterministic fake
    embedder -- the memory store's read/write/dispute behavior against
    the real pgvector column, without needing Ollama running.

The fake embedder is a bag-of-words vector rather than a hash of the
whole string (Milestone 9's approach): texts that share vocabulary get
genuinely higher cosine similarity, so retrieval-ordering assertions
below test real ranking instead of a hash coincidence. It is full-width
(EMBEDDING_DIM) because unlike Milestone 9's in-Python dedup, these
vectors are actually stored in a Vector(768) column.

Requires: Postgres at TEST_DATABASE_URL.
"""

import hashlib
import re
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select, text

from amop.database.models import EMBEDDING_DIM, MemoryItem, Task
from amop.database.session import init_db, make_engine, make_session_factory
from amop.memory import store as memory_store
from amop.orchestrator.chain import MEMORY_WRITE_STATES, ChainResult
from amop.orchestrator.state_machine import TERMINAL_STATES, TaskState

TEST_DATABASE_URL = "postgresql+asyncpg://localhost/amop_test"


# ---------------------------------------------------------------------
# Deterministic fake embedder
# ---------------------------------------------------------------------


def _fake_vector(text_value: str) -> list[float]:
    vec = [0.0] * EMBEDDING_DIM
    for word in re.findall(r"[a-z0-9]+", text_value.lower()):
        digest = hashlib.sha256(word.encode()).digest()
        vec[int.from_bytes(digest[:4], "big") % EMBEDDING_DIM] += 1.0
    if not any(vec):
        # pgvector rejects nothing here, but an all-zero vector has
        # undefined cosine distance -- give it a stable non-zero axis.
        vec[0] = 1.0
    return vec


async def _fake_embed_texts(texts: list[str], provider=None) -> list[list[float]]:
    return [_fake_vector(t) for t in texts]


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    monkeypatch.setattr(memory_store, "embed_texts", _fake_embed_texts)


# ---------------------------------------------------------------------
# Postgres fixtures
# ---------------------------------------------------------------------


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
                "TRUNCATE memory_items, task_transitions, tasks "
                "RESTART IDENTITY CASCADE"
            )
        )


def _chain_result(
    *,
    prompt: str,
    root_cause: str | None = None,
    files_changed: list[str] | None = None,
    final_state: TaskState = TaskState.WAITING_FOR_APPROVAL,
    task: Task | None = None,
) -> ChainResult:
    """A ChainResult shaped like a real one, without running a chain."""
    from amop.agents.handoffs import CodeChangeReport, RootCauseReport

    task = task or Task(
        id=uuid.uuid4(),
        task_type="bug_fix",
        state=final_state.value,
        task_context={"prompt": prompt, "repo": "/repo"},
    )
    result = ChainResult(task=task, final_state=final_state)
    if root_cause is not None:
        result.root_cause_report = RootCauseReport(
            task_id=str(task.id),
            root_cause=root_cause,
            confidence=0.9,
            affected_files=["app/thing.py"],
            suggested_fix_plan="do the thing",
        )
    if files_changed is not None:
        result.code_change_report = CodeChangeReport(
            task_id=str(task.id),
            status="success",
            branch="amop/fix-test",
            files_changed=files_changed,
        )
    return result


# ---------------------------------------------------------------------
# Tier (a) — pure
# ---------------------------------------------------------------------


def test_memory_write_states_cover_every_terminal_state_plus_waiting():
    """Section 10.2 says "every resolved task". Against this state
    machine that has to mean every terminal state PLUS
    WAITING_FOR_APPROVAL -- nothing ever reaches RESOLVED today, so a
    literal RESOLVED/FAILED reading would record failures only."""
    assert TERMINAL_STATES <= MEMORY_WRITE_STATES
    assert TaskState.WAITING_FOR_APPROVAL in MEMORY_WRITE_STATES
    assert TaskState.RESOLVED in MEMORY_WRITE_STATES  # ready for when it's reachable
    # Mid-chain states must NOT trigger a write.
    assert TaskState.CODING not in MEMORY_WRITE_STATES
    assert TaskState.INVESTIGATING not in MEMORY_WRITE_STATES


def test_build_incident_summary_uses_ground_truth_not_self_report():
    result = _chain_result(
        prompt="average() returns the wrong number for a 3-item list",
        root_cause="divides by len(values) - 1",
        files_changed=["calculator.py"],
        final_state=TaskState.WAITING_FOR_APPROVAL,
    )

    summary = memory_store.build_incident_summary(result)

    assert summary["anomaly_signature"].startswith("average() returns")
    assert summary["root_cause"] == "divides by len(values) - 1"
    # outcome is the real landing state, not a model's claim of success
    assert summary["outcome"] == "WAITING_FOR_APPROVAL"
    # files_changed came from CodeChangeReport, which the orchestrator
    # builds from git (Section 6.3.9), never from the Coder's own report
    assert summary["files_changed"] == ["calculator.py"]


def test_build_incident_summary_survives_a_chain_that_never_got_a_diagnosis():
    result = _chain_result(prompt="something is broken", final_state=TaskState.FAILED)

    summary = memory_store.build_incident_summary(result)

    assert summary["root_cause"] is None
    assert summary["files_changed"] == []
    assert summary["fix_summary"] == "no code change was produced"
    assert summary["outcome"] == "FAILED"


def test_embedding_text_is_symptom_side_only():
    """The retrieval query at read time is a new bug report -- symptoms,
    by someone who doesn't know the cause yet. The stored vector has to
    live in that space, so the fix summary is deliberately excluded."""
    content = {
        "anomaly_signature": "NixOS has no /bin/bash",
        "root_cause": "shell path is hardcoded",
        "fix_summary": "success: changed invoke/runners.py",
        "files_changed": ["invoke/runners.py"],
    }

    text_value = memory_store.embedding_text(content)

    assert "NixOS has no /bin/bash" in text_value
    assert "shell path is hardcoded" in text_value
    assert "runners.py" not in text_value
    assert "success" not in text_value


def test_embedding_text_truncates_a_pathological_bug_report():
    content = {"anomaly_signature": "x" * 50_000, "root_cause": "y"}
    assert len(memory_store.embedding_text(content)) <= 8_000


# ---------------------------------------------------------------------
# Tier (b) — real Postgres + real pgvector column
# ---------------------------------------------------------------------


async def test_write_incident_memory_creates_a_real_row(session):
    task = Task(
        id=uuid.uuid4(),
        task_type="bug_fix",
        state=TaskState.WAITING_FOR_APPROVAL.value,
        task_context={"prompt": "the login page 500s on submit", "repo": "/repo"},
    )
    session.add(task)
    await session.commit()

    result = _chain_result(
        prompt="the login page 500s on submit",
        root_cause="null session token",
        files_changed=["auth.py"],
        task=task,
    )

    item = await memory_store.write_incident_memory(session, result, repo_path="/repo")

    assert item is not None
    assert item.memory_type == memory_store.MEMORY_TYPE_INCIDENT
    assert item.disputed is False
    assert item.task_id == task.id
    assert item.content["root_cause"] == "null session token"
    assert len(item.embedding) == EMBEDDING_DIM

    stored = (await session.execute(select(MemoryItem))).scalars().all()
    assert len(stored) == 1


async def test_write_incident_memory_skips_a_task_with_no_description(session):
    result = _chain_result(prompt="", final_state=TaskState.FAILED)

    item = await memory_store.write_incident_memory(session, result, repo_path="/repo")

    assert item is None
    assert (await session.execute(select(MemoryItem))).scalars().all() == []


async def _seed(session, prompt: str, root_cause: str, repo_path: str = "/repo"):
    result = _chain_result(prompt=prompt, root_cause=root_cause, files_changed=["a.py"])
    return await memory_store.write_incident_memory(
        session, result, repo_path=repo_path
    )


async def test_search_memory_surfaces_the_genuinely_similar_incident(session):
    await _seed(
        session,
        "database connection pool exhausted under load",
        "pool size too small",
    )
    await _seed(
        session,
        "the sidebar renders the wrong icon colour",
        "css variable typo",
    )

    matches = await memory_store.search_memory(
        session, "connection pool exhausted when load is high", repo_path="/repo"
    )

    assert matches
    assert "connection pool exhausted" in matches[0].item.content["anomaly_signature"]
    assert matches[0].similarity > 0.0


async def test_search_memory_excludes_disputed_but_keeps_the_row(session):
    item = await _seed(
        session, "cache never invalidates after a write", "missing bust call"
    )

    before = await memory_store.search_memory(
        session, "cache never invalidates after a write", repo_path="/repo"
    )
    assert [m.item.id for m in before] == [item.id]

    assert await memory_store.mark_disputed(session, item.id) is True

    after = await memory_store.search_memory(
        session, "cache never invalidates after a write", repo_path="/repo"
    )
    assert after == []

    # Section 10.4: excluded from retrieval, NOT deleted -- still auditable.
    row = (
        await session.execute(select(MemoryItem).where(MemoryItem.id == item.id))
    ).scalar_one()
    assert row.disputed is True
    assert row.content["root_cause"] == "missing bust call"


async def test_search_memory_scopes_to_repo_path(session):
    await _seed(session, "widget overflows its container", "bad flex rule", "/repo-a")

    same_repo = await memory_store.search_memory(
        session, "widget overflows its container", repo_path="/repo-a"
    )
    other_repo = await memory_store.search_memory(
        session, "widget overflows its container", repo_path="/repo-b"
    )

    assert len(same_repo) == 1
    assert other_repo == []


async def test_search_memory_respects_the_relevance_floor(session):
    await _seed(session, "kafka consumer lag climbing", "slow downstream write")

    unrelated = await memory_store.search_memory(
        session,
        "typography kerning wrong heading",
        repo_path="/repo",
        min_similarity=memory_store.RELEVANCE_FLOOR,
    )

    assert unrelated == []


async def test_mark_disputed_reports_a_miss_for_an_unknown_id(session):
    assert await memory_store.mark_disputed(session, uuid.uuid4()) is False
