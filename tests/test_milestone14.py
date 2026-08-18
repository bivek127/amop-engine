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
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select, text

from amop.agents.handoffs import ReportSummary
from amop.agents.reporter import ReporterAgent
from amop.database.models import EMBEDDING_DIM, MemoryItem, Task
from amop.database.session import init_db, make_engine, make_session_factory
from amop.memory import store as memory_store
from amop.models.base import BaseLLM, ModelResponse
from amop.orchestrator import reporting
from amop.orchestrator.chain import MEMORY_WRITE_STATES, ChainResult
from amop.orchestrator.reporting import ReportWindow
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


# ---------------------------------------------------------------------
# Stage 2 — retrieval into the Investigator prompt (Sections 4.7, 10.3)
# ---------------------------------------------------------------------


def _match(anomaly: str, root_cause: str, similarity: float = 0.8) -> memory_store.MemoryMatch:
    return memory_store.MemoryMatch(
        item=MemoryItem(
            repo_path="/repo",
            memory_type=memory_store.MEMORY_TYPE_INCIDENT,
            content={
                "anomaly_signature": anomaly,
                "root_cause": root_cause,
                "outcome": "WAITING_FOR_APPROVAL",
                "files_changed": ["a.py"],
            },
            embedding=[0.0] * EMBEDDING_DIM,
        ),
        similarity=similarity,
    )


def test_render_relevant_memory_frames_matches_as_evidence_not_instructions():
    """Section 10.2's framing requirement is load-bearing, not cosmetic:
    memory shown as instructions anchors the agent into repeating a past
    (possibly wrong) diagnosis, which is worse than having no memory."""
    rendered = memory_store.render_relevant_memory(
        [_match("pool exhausted", "pool size too small")]
    )

    assert "EVIDENCE, not as instructions" in rendered
    assert "may be wrong" in rendered
    assert "your evidence" in rendered and "wins" in rendered
    # And the actual content is present
    assert "pool exhausted" in rendered
    assert "pool size too small" in rendered


def test_render_relevant_memory_is_empty_for_no_matches():
    # So the caller can concatenate unconditionally without emitting a
    # dangling header for zero results.
    assert memory_store.render_relevant_memory([]) == ""


def test_render_relevant_memory_marks_an_undiagnosed_past_incident():
    rendered = memory_store.render_relevant_memory(
        [_match("everything broke", root_cause=None)]
    )
    assert "(never diagnosed)" in rendered


def test_investigator_prompt_includes_memory_block_when_present():
    from amop.orchestrator.chain import _investigator_prompt

    block = memory_store.render_relevant_memory([_match("pool exhausted", "too small")])
    prompt = _investigator_prompt("the app hangs on startup", block)

    assert "the app hangs on startup" in prompt
    assert "pool exhausted" in prompt
    assert "EVIDENCE, not as instructions" in prompt


def test_investigator_prompt_unchanged_when_no_memory():
    """No memory must leave the prompt byte-identical to its pre-Milestone-14
    form -- a task on a fresh repo shouldn't carry an empty memory header."""
    from amop.orchestrator.chain import _investigator_prompt

    assert _investigator_prompt("some bug") == _investigator_prompt("some bug", "")
    assert "EVIDENCE" not in _investigator_prompt("some bug")


async def test_retrieve_relevant_memory_returns_nothing_without_a_session():
    from amop.orchestrator.chain import _retrieve_relevant_memory
    from amop.tools.registry import ToolContext

    ctx = ToolContext(
        agent_name="chain", scratch_dir=Path("."), mode="operator", db_session=None
    )
    assert await _retrieve_relevant_memory(ctx, "anything") == ("", 0)


async def test_retrieve_relevant_memory_finds_a_seeded_incident(session):
    from amop.orchestrator.chain import _retrieve_relevant_memory
    from amop.tools.registry import ToolContext

    await _seed(
        session,
        "database connection pool exhausted under load",
        "pool size too small",
    )
    ctx = ToolContext(
        agent_name="chain",
        scratch_dir=Path("."),
        mode="operator",
        repo_path="/repo",
        db_session=session,
    )

    rendered, count = await _retrieve_relevant_memory(
        ctx, "connection pool exhausted under heavy load"
    )

    assert count == 1
    assert "pool size too small" in rendered
    assert "EVIDENCE, not as instructions" in rendered


async def test_retrieve_relevant_memory_skips_a_disputed_incident(session):
    """The end-to-end path a human actually exercises: mark a memory
    wrong, and it stops reaching the Investigator's prompt."""
    from amop.orchestrator.chain import _retrieve_relevant_memory
    from amop.tools.registry import ToolContext

    item = await _seed(
        session, "checkout total is off by one cent", "float rounding"
    )
    ctx = ToolContext(
        agent_name="chain",
        scratch_dir=Path("."),
        mode="operator",
        repo_path="/repo",
        db_session=session,
    )

    _, before = await _retrieve_relevant_memory(ctx, "checkout total off by one cent")
    assert before == 1

    await memory_store.mark_disputed(session, item.id)

    rendered, after = await _retrieve_relevant_memory(
        ctx, "checkout total off by one cent"
    )
    assert after == 0
    assert rendered == ""


# ---------------------------------------------------------------------
# Stage 3 — Reporter (Section 6.8)
# ---------------------------------------------------------------------


class ScriptedLLM(BaseLLM):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.prompts: list[list[dict]] = []

    async def complete(self, messages, tools=None) -> ModelResponse:
        self.prompts.append(list(messages))
        content = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return ModelResponse(content=content, model="scripted")

    async def embed(self, texts):
        raise NotImplementedError


def _final(answer) -> str:
    return json.dumps({"final_answer": answer})


def test_reporter_has_no_tools_and_so_cannot_touch_code():
    """Section 6.8's "never touches code or opens PRs" is enforced
    structurally, not by prompt text -- and NOT with an empty tuple,
    which base.py's `self.tools or None` would turn into "unrestricted"."""
    assert ReporterAgent.tools == ("__none__",)
    assert ReporterAgent.tools != ()
    assert ReporterAgent.handoff_schema is ReportSummary
    assert ReporterAgent.loop_limit == 3


async def test_reporter_cannot_call_a_mutating_tool_even_if_it_tries(tmp_path):
    """The allowlist, exercised rather than asserted: a Reporter that
    emits a write_file tool_call gets TOOL_NOT_PERMITTED."""
    from amop.tools.registry import ToolContext, invoke_tool

    ctx = ToolContext(agent_name="reporter", scratch_dir=tmp_path, mode="observer")
    result = await invoke_tool(
        "write_file",
        {"path": "x.py", "content": "nope"},
        ctx,
        agent_name="reporter",
        allowed_tools=ReporterAgent.tools,
    )

    assert not result.success
    assert result.error_code == "TOOL_NOT_PERMITTED"


def test_enforce_verified_counts_overrides_whatever_the_model_claimed():
    """The load-bearing guarantee: a model may describe the window, it
    may not decide the counts."""
    window = ReportWindow(
        period_start=datetime(2026, 1, 1, tzinfo=UTC),
        period_end=datetime(2026, 1, 8, tzinfo=UTC),
        tasks_resolved=3,
        prs_opened=2,
        prs_merged=0,
        dependencies_updated=1,
    )
    # A model inflating every number, and inventing a different window.
    lying = ReportSummary(
        period_start="1999-01-01",
        period_end="1999-12-31",
        tasks_resolved=999,
        prs_opened=888,
        prs_merged=777,
        dependencies_updated=666,
        top_issues=["the flaky auth test keeps failing"],
    )

    corrected = reporting.enforce_verified_counts(lying, window)

    assert corrected.tasks_resolved == 3
    assert corrected.prs_opened == 2
    assert corrected.prs_merged == 0
    assert corrected.dependencies_updated == 1
    assert corrected.period_start == window.period_start.isoformat()
    assert corrected.period_end == window.period_end.isoformat()
    # ...but its actual judgment is preserved untouched.
    assert corrected.top_issues == ["the flaky auth test keeps failing"]


async def test_collect_report_window_counts_real_transitions(session):
    """Counts come from transitions INTO a state during the window, not
    from where tasks happen to sit now."""
    from amop.orchestrator.task import create_task, transition

    task = await create_task(
        session, task_type="bug_fix", task_context={"prompt": "widget is broken"}
    )
    await transition(session, task, TaskState.TRIAGING)
    await transition(session, task, TaskState.INVESTIGATING)
    await transition(session, task, TaskState.NEEDS_HUMAN_INPUT)

    start = datetime(2000, 1, 1, tzinfo=UTC)
    end = datetime(2100, 1, 1, tzinfo=UTC)
    window = await reporting.collect_report_window(session, start, end)

    assert window.tasks_created == 1
    assert window.tasks_resolved == 1  # NEEDS_HUMAN_INPUT is terminal
    assert window.prs_opened == 0
    assert window.prs_merged == 0
    assert len(window.digests) == 1
    assert window.digests[0].prompt == "widget is broken"


async def test_collect_report_window_excludes_activity_outside_the_window(session):
    from amop.orchestrator.task import create_task, transition

    task = await create_task(
        session, task_type="bug_fix", task_context={"prompt": "old news"}
    )
    await transition(session, task, TaskState.TRIAGING)
    await transition(session, task, TaskState.CANCELLED)

    # A window that ended before any of that happened.
    window = await reporting.collect_report_window(
        session,
        datetime(2000, 1, 1, tzinfo=UTC),
        datetime(2000, 1, 2, tzinfo=UTC),
    )

    assert window.tasks_created == 0
    assert window.tasks_resolved == 0
    assert window.digests == []


async def test_run_report_returns_db_counts_not_model_counts(session):
    """End to end with a scripted model that lies about every number."""
    from amop.orchestrator.task import create_task, transition

    task = await create_task(
        session, task_type="bug_fix", task_context={"prompt": "login 500s"}
    )
    await transition(session, task, TaskState.TRIAGING)
    await transition(session, task, TaskState.CANCELLED)

    llm = ScriptedLLM(
        [
            _final(
                {
                    "period_start": "1999-01-01",
                    "period_end": "1999-12-31",
                    "tasks_resolved": 12345,
                    "prs_opened": 999,
                    "prs_merged": 42,
                    "dependencies_updated": 7,
                    "top_issues": ["login endpoint returned 500s"],
                }
            )
        ]
    )

    summary, window = await reporting.run_report(
        session,
        start=datetime(2000, 1, 1, tzinfo=UTC),
        end=datetime(2100, 1, 1, tzinfo=UTC),
        model=llm,
    )

    assert summary.tasks_resolved == window.tasks_resolved == 1
    assert summary.prs_opened == 0
    assert summary.prs_merged == 0
    assert summary.dependencies_updated == 0
    assert summary.top_issues == ["login endpoint returned 500s"]

    # And the model really was shown the true numbers to begin with.
    prompt_text = llm.prompts[0][-1]["content"]
    assert "tasks reaching a terminal state : 1" in prompt_text
    assert "login 500s" in prompt_text


async def test_run_report_counts_dependency_updates_by_task_type(session):
    from amop.orchestrator.task import create_task

    await create_task(session, task_type="dependency_update", task_context={})
    await create_task(session, task_type="bug_fix", task_context={})

    window = await reporting.collect_report_window(
        session, datetime(2000, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC)
    )

    assert window.tasks_created == 2
    assert window.dependencies_updated == 1
