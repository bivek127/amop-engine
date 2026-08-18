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

from amop.agents.dependency_updater import DependencyUpdaterAgent
from amop.agents.handoffs import DependencyUpdateReport, ReportSummary
from amop.agents.optimizer import OptimizerAgent
from amop.agents.reporter import ReporterAgent
from amop.database.models import EMBEDDING_DIM, MemoryItem, Task
from amop.database.session import init_db, make_engine, make_session_factory
from amop.memory import store as memory_store
from amop.models.base import BaseLLM, ModelResponse
from amop.orchestrator import deps, optimize, reporting
from amop.orchestrator.chain import MEMORY_WRITE_STATES, ChainResult
from amop.orchestrator.reporting import ReportWindow
from amop.orchestrator.state_machine import TERMINAL_STATES, TaskState
from amop.sandbox.manager import SandboxManager
from amop.tools import advisories
from amop.tools.registry import ToolContext, invoke_tool

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


@pytest.fixture
def sandbox_manager():
    manager = SandboxManager()
    yield manager
    manager.destroy_all()


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


# ---------------------------------------------------------------------
# Stage 4 — DependencyUpdater (Section 6.7)
# ---------------------------------------------------------------------


def test_dependency_updater_defaults_to_autonomous_permission():
    """6.7: the only agent whose default is `autonomous`, on the grounds
    that a version bump is mechanical, low-blast-radius, and revertible."""
    assert deps.DEFAULT_PERMISSION_MODE == "autonomous"
    assert deps.MAX_FILES_FOR_AUTO_FIX == 3
    assert DependencyUpdaterAgent.handoff_schema is DependencyUpdateReport
    assert "check_advisories" in DependencyUpdaterAgent.tools


def test_source_files_changed_excludes_the_manifest_being_bumped():
    """6.7 caps "the fix" -- the source changes a new version forces --
    not the bump itself. Counting the manifest would silently turn a
    documented budget of 3 into an actual 2."""
    changed = ["requirements.txt", "app/a.py", "app/b.py"]
    assert deps.source_files_changed(changed) == ["app/a.py", "app/b.py"]


def test_source_files_changed_handles_nested_manifests():
    assert deps.source_files_changed(["sub/pyproject.toml", "sub/x.py"]) == ["sub/x.py"]


def test_parse_requirements_only_accepts_pinned_versions():
    """An advisory is a claim about a specific version, so an unpinned
    requirement isn't checkable -- it must be reported as skipped, never
    silently dropped (which would read as "checked and clean")."""
    pinned, skipped = advisories.parse_requirements(
        "\n".join(
            [
                "# a comment",
                "requests==2.19.1",
                "flask>=2.0  # unpinned, not checkable",
                "",
                "-r other.txt",
                "jinja2==2.10",
            ]
        )
    )

    assert pinned == [("requests", "2.19.1"), ("jinja2", "2.10")]
    assert skipped == ["flask>=2.0"]


def test_parse_requirements_ignores_comments_and_blanks():
    pinned, skipped = advisories.parse_requirements("\n  \n# nothing here\n")
    assert pinned == []
    assert skipped == []


def test_fixed_versions_only_reports_the_named_package():
    """A single OSV advisory can list several affected packages; picking
    up another package's fixed version would send the agent to a version
    that doesn't exist for the one it's bumping."""
    vuln = {
        "affected": [
            {
                "package": {"name": "requests"},
                "ranges": [{"events": [{"introduced": "0"}, {"fixed": "2.20.0"}]}],
            },
            {
                "package": {"name": "urllib3"},
                "ranges": [{"events": [{"introduced": "0"}, {"fixed": "1.99.0"}]}],
            },
        ]
    }
    assert advisories._fixed_versions(vuln, "requests") == ["2.20.0"]


async def test_check_advisories_offline_is_an_error_not_an_empty_result(
    tmp_path, monkeypatch, sandbox_manager
):
    """The single most dangerous failure mode for an autonomous updater:
    "I could not reach the advisory database" must never be reported as
    "no advisories found"."""
    (tmp_path / "requirements.txt").write_text("requests==2.19.1\n")
    monkeypatch.setattr(
        advisories, "OSV_QUERY_URL", "http://127.0.0.1:9/definitely-not-listening"
    )
    sandbox = sandbox_manager.create("t-adv-offline", tmp_path)
    try:
        ctx = ToolContext(
            agent_name="dependency_updater",
            scratch_dir=tmp_path,
            mode="autonomous",
            sandbox=sandbox,
        )
        result = await advisories.check_advisories("requirements.txt", ctx=ctx)

        assert not result.success
        assert result.error_code == "ADVISORY_LOOKUP_FAILED"
        assert "NOT a statement that the dependencies are clean" in result.message
    finally:
        sandbox_manager.destroy("t-adv-offline")


async def test_check_advisories_reports_zero_checked_for_an_unpinned_manifest(
    tmp_path, sandbox_manager
):
    # No network call happens at all here -- nothing is pinned, so there
    # is nothing to look up.
    (tmp_path / "requirements.txt").write_text("flask\nrequests>=2\n")
    sandbox = sandbox_manager.create("t-adv-unpinned", tmp_path)
    try:
        ctx = ToolContext(
            agent_name="dependency_updater",
            scratch_dir=tmp_path,
            mode="autonomous",
            sandbox=sandbox,
        )
        result = await advisories.check_advisories("requirements.txt", ctx=ctx)

        assert result.success
        assert result.output["checked"] == 0
        assert result.output["advisories"] == []
        assert sorted(result.output["skipped_unpinned"]) == ["flask", "requests>=2"]
    finally:
        sandbox_manager.destroy("t-adv-unpinned")


async def test_check_advisories_refuses_a_path_outside_the_workspace(
    tmp_path, sandbox_manager
):
    sandbox = sandbox_manager.create("t-adv-escape", tmp_path)
    try:
        ctx = ToolContext(
            agent_name="dependency_updater",
            scratch_dir=tmp_path,
            mode="autonomous",
            sandbox=sandbox,
        )
        result = await advisories.check_advisories("../../etc/passwd", ctx=ctx)
        assert not result.success
        assert result.error_code == "PATH_NOT_PERMITTED"
    finally:
        sandbox_manager.destroy("t-adv-escape")


# --- the two paths that must be enforced in code, not by the prompt ---

DEPS_FIXTURE = Path(__file__).parent / "fixtures" / "outdated_deps"


def _tool_call(name: str, **arguments) -> str:
    return json.dumps({"tool_call": {"name": name, "arguments": arguments}})


def _deps_answer(**overrides) -> str:
    payload = {
        "task_id": "will-be-overwritten",
        "package": "requests",
        "from_version": "2.19.1",
        "to_version": "2.32.0",
        "cve_ids": ["GHSA-9wx4-h78v-vm56"],
        "status": "success",
        "tests_passed": True,
    }
    payload.update(overrides)
    return _final(payload)


# A targeted one-line bump -- the correct way to edit a manifest that
# also holds comments and (in a real repo) other pins. A whole-file
# write_file here is refused by Milestone 13's SUSPICIOUS_SHRINK guard,
# which is the guard doing its job: see
# test_manifest_whole_file_rewrite_is_refused_then_recovers_via_patch.
_BUMP_DIFF = (
    "--- a/requirements.txt\n"
    "+++ b/requirements.txt\n"
    "@@ -8,1 +8,1 @@\n"
    "-requests==2.19.1\n"
    "+requests==2.32.0\n"
)


async def test_dependency_update_success_path_bumps_manifest_and_verifies_tests(
    session, tmp_path
):
    """The happy path, with the verdict coming from a real pytest run in
    a real container -- not from the model's `tests_passed: True`."""
    from amop.orchestrator.task import create_task

    task = await create_task(session, task_type="dependency_update", task_context={})
    llm = ScriptedLLM(
        [
            _tool_call("patch_file", path="requirements.txt", diff=_BUMP_DIFF),
            _deps_answer(),
        ]
    )

    result = await deps.run_dependency_update(
        session,
        task,
        repo_path=DEPS_FIXTURE,
        model=llm,
        scratch_root=tmp_path,
    )

    assert result.report.status == "success"
    assert result.report.tests_passed is True
    assert result.report.files_changed == ["requirements.txt"]
    assert result.reverted is False
    assert "requests==2.32.0" in result.diff


async def test_dependency_update_aborts_and_reverts_past_the_file_cap(
    session, tmp_path
):
    """Section 6.7's escalation rule, enforced mechanically: an agent
    that sprawls across more than max_files_for_auto_fix source files
    gets its work reverted and handed back as needs_manual_review --
    regardless of it reporting `status: success, tests_passed: True`."""
    from amop.orchestrator.task import create_task

    task = await create_task(session, task_type="dependency_update", task_context={})
    llm = ScriptedLLM(
        [
            _tool_call("patch_file", path="requirements.txt", diff=_BUMP_DIFF),
            _tool_call("write_file", path="app/one.py", content="ONE = 1\n"),
            _tool_call("write_file", path="app/two.py", content="TWO = 2\n"),
            _tool_call("write_file", path="app/three.py", content="THREE = 3\n"),
            _tool_call("write_file", path="app/four.py", content="FOUR = 4\n"),
            # The model insists everything is fine. It does not get a vote.
            _deps_answer(status="success", tests_passed=True),
        ]
    )

    result = await deps.run_dependency_update(
        session,
        task,
        repo_path=DEPS_FIXTURE,
        model=llm,
        scratch_root=tmp_path,
    )

    assert result.report.status == "needs_manual_review"
    assert result.report.tests_passed is False
    assert result.reverted is True
    assert "over the max_files_for_auto_fix cap" in result.report.diagnostic
    # And the working tree really was restored, not just relabelled.
    scratch = tmp_path / str(task.id)
    assert not (scratch / "app" / "four.py").exists()
    assert (scratch / "requirements.txt").read_text().strip().endswith("requests==2.19.1")


async def test_dependency_update_reverts_when_the_suite_fails(session, tmp_path):
    """A bump that breaks the suite is reverted too -- an autonomous
    agent must not leave a red tree behind."""
    from amop.orchestrator.task import create_task

    task = await create_task(session, task_type="dependency_update", task_context={})
    llm = ScriptedLLM(
        [
            _tool_call("patch_file", path="requirements.txt", diff=_BUMP_DIFF),
            # Break the suite with a single in-budget source edit.
            # patch_file, not write_file: a whole-file rewrite down to a
            # stub trips Milestone 13's SUSPICIOUS_SHRINK guard, so the
            # edit would be refused and the suite would stay green --
            # the guard protecting the file, correctly, from the very
            # damage this test is trying to cause.
            _tool_call(
                "patch_file",
                path="app/urls.py",
                diff=(
                    "--- a/app/urls.py\n"
                    "+++ b/app/urls.py\n"
                    "@@ -12,5 +12,5 @@\n"
                    " def normalize(url: str) -> str:\n"
                    '     """Strip whitespace and a single trailing slash from a URL."""\n'
                    "-    cleaned = url.strip()\n"
                    '+    cleaned = "definitely not the url"\n'
                    '     if cleaned.endswith("/") and len(cleaned) > 1:\n'
                    "         cleaned = cleaned[:-1]\n"
                ),
            ),
            _deps_answer(status="success", tests_passed=True),
        ]
    )

    result = await deps.run_dependency_update(
        session,
        task,
        repo_path=DEPS_FIXTURE,
        model=llm,
        scratch_root=tmp_path,
    )

    assert result.report.status == "needs_manual_review"
    assert result.report.tests_passed is False
    assert result.reverted is True
    scratch = tmp_path / str(task.id)
    assert "def normalize(url: str)" in (scratch / "app" / "urls.py").read_text()


async def test_dependency_update_reports_no_change_honestly(session, tmp_path):
    from amop.orchestrator.task import create_task

    task = await create_task(session, task_type="dependency_update", task_context={})
    llm = ScriptedLLM([_deps_answer(status="success", tests_passed=True)])

    result = await deps.run_dependency_update(
        session,
        task,
        repo_path=DEPS_FIXTURE,
        model=llm,
        scratch_root=tmp_path,
    )

    assert result.report.status == "needs_manual_review"
    assert result.report.files_changed == []
    assert "no change" in result.report.diagnostic


async def test_changed_files_sees_new_files_but_not_gitignored_build_artifacts(
    session, tmp_path
):
    """Two halves of the same Milestone 14 finding, pinned together.

    `changed_files()` used to run on `git diff` alone, which ignores
    untracked paths -- so an agent that CREATED files looked like it had
    changed nothing, and Section 6.7's max_files_for_auto_fix cap could
    be walked straight past by creating rather than editing.

    Including untracked files fixed that but immediately surfaced the
    other half, live: the agent's own run_tests leaves __pycache__/*.pyc
    behind, and three stray .pyc files are enough to exhaust a cap of 3
    on their own. `--exclude-standard` honors .gitignore, which is why
    the fixture now ships one (the Milestone 6 finding, recurring).

    This test drives an agent that both creates a file AND runs the
    suite, so a regression in either half fails it.
    """
    from amop.orchestrator.task import create_task

    task = await create_task(session, task_type="dependency_update", task_context={})
    llm = ScriptedLLM(
        [
            _tool_call("patch_file", path="requirements.txt", diff=_BUMP_DIFF),
            _tool_call("write_file", path="app/newly_created.py", content="X = 1\n"),
            # Generates __pycache__ inside the workspace, as a real run does.
            _tool_call("run_tests"),
            _deps_answer(),
        ]
    )

    result = await deps.run_dependency_update(
        session,
        task,
        repo_path=DEPS_FIXTURE,
        model=llm,
        scratch_root=tmp_path,
    )

    changed = result.report.files_changed
    # The created file is visible (the bug this fix closed)...
    assert "app/newly_created.py" in changed
    # ...and build artifacts are not (the bug the fix introduced).
    assert not [p for p in changed if "__pycache__" in p or p.endswith(".pyc")], changed
    # One manifest + one real source file is within the cap, so this stands.
    assert result.report.status == "success"
    assert result.reverted is False


# ---------------------------------------------------------------------
# Stage 5 — Optimizer (Section 6.6)
# ---------------------------------------------------------------------

OPT_FIXTURE = Path(__file__).parent / "fixtures" / "slow_report"

# The known-good optimization: swap the O(n^2) list scan for a set.
_FAST_AGGREGATE = '''"""Report aggregation."""


def count_unique_visitors(visitor_ids: list[str]) -> int:
    """Count distinct visitor ids."""
    return len(set(visitor_ids))


def summarize(visitor_ids: list[str]) -> dict:
    return {
        "total_events": len(visitor_ids),
        "unique_visitors": count_unique_visitors(visitor_ids),
    }
'''

# A change that is real but trivially small -- adding a comment. Behavior
# identical, speed identical, so it must be reverted as no_improvement.
_COSMETIC_AGGREGATE = '''"""Report aggregation with one deliberate, real hot spot."""


def count_unique_visitors(visitor_ids: list[str]) -> int:
    """Count distinct visitor ids. (now with a nicer comment)"""
    seen: list[str] = []
    for visitor in visitor_ids:
        # check membership before appending
        if visitor not in seen:
            seen.append(visitor)
    return len(seen)


def summarize(visitor_ids: list[str]) -> dict:
    return {
        "total_events": len(visitor_ids),
        "unique_visitors": count_unique_visitors(visitor_ids),
    }
'''


def test_improvement_pct_math_and_zero_baseline_guard():
    assert optimize.improvement_pct(100.0, 50.0) == 50.0
    assert optimize.improvement_pct(100.0, 95.0) == 5.0
    assert optimize.improvement_pct(100.0, 120.0) == -20.0
    # A zero baseline means the harness failed, not infinite speed --
    # dividing by it would manufacture an "improvement" out of nothing.
    assert optimize.improvement_pct(0.0, 5.0) == 0.0


def test_optimizer_defaults_match_the_spec():
    assert optimize.MIN_IMPROVEMENT_PCT == 10.0
    assert optimize.DEFAULT_PERMISSION_MODE == "suggestor"
    assert OptimizerAgent.loop_limit == 15
    assert "profile_code" in OptimizerAgent.tools
    assert "run_benchmark" in OptimizerAgent.tools


async def test_run_benchmark_measures_the_fixture(session, tmp_path, sandbox_manager):
    """The measurement tool itself, against real code in a real container."""
    from amop.sandbox import repo as git_repo

    scratch = tmp_path / "ws"
    git_repo.materialize(OPT_FIXTURE, scratch)
    sandbox = sandbox_manager.create("t-bench", scratch)
    try:
        ctx = ToolContext(
            agent_name="optimizer", scratch_dir=scratch, mode="suggestor", sandbox=sandbox
        )
        result = await invoke_tool(
            "run_benchmark", {"path": "benchmark.py", "iterations": 3}, ctx,
            agent_name="orchestrator",
        )
        assert result.success, result.message
        assert result.output["iterations"] == 3
        assert result.output["median_ms"] > 0
        assert result.output["min_ms"] <= result.output["median_ms"] <= result.output["max_ms"]
    finally:
        sandbox_manager.destroy("t-bench")


async def test_profile_code_points_at_the_real_hot_function(
    session, tmp_path, sandbox_manager
):
    """6.6's "never guess at a bottleneck without a profile in evidence"
    is only worth anything if the profile actually names the culprit."""
    from amop.sandbox import repo as git_repo

    scratch = tmp_path / "ws"
    git_repo.materialize(OPT_FIXTURE, scratch)
    sandbox = sandbox_manager.create("t-prof", scratch)
    try:
        ctx = ToolContext(
            agent_name="optimizer", scratch_dir=scratch, mode="suggestor", sandbox=sandbox
        )
        result = await invoke_tool(
            "profile_code", {"path": "benchmark.py"}, ctx, agent_name="orchestrator"
        )
        assert result.success, result.message

        # Presence is not enough -- RANKING is the point. The first live
        # run got a top-5 of nothing but runpy scaffolding frames (whose
        # cumulative time covers the entire program by construction) and
        # reported "No bottleneck found in the profile". Scaffolding is
        # now filtered, and self time is what actually names a hot spot.
        by_self = result.output["top_by_self_time"]
        assert "count_unique_visitors" in by_self[0]["function"], by_self

        joined = " ".join(r["function"] for r in result.output["top"])
        assert "frozen" not in joined, joined
    finally:
        sandbox_manager.destroy("t-prof")


async def test_optimizer_keeps_a_genuine_improvement(session, tmp_path):
    """The real optimization: O(n^2) -> O(n), measured, kept."""
    from amop.orchestrator.task import create_task

    task = await create_task(session, task_type="optimization", task_context={})
    llm = ScriptedLLM(
        [
            _tool_call("profile_code", path="benchmark.py"),
            _tool_call("write_file", path="app/aggregate.py", content=_FAST_AGGREGATE),
            _final(
                {
                    "task_id": "overwritten",
                    "status": "improved",
                    "baseline_ms": 999.0,
                    "optimized_ms": 1.0,
                    "improvement_pct": 99.9,
                    "technique": "replaced the linear membership scan with a set",
                }
            ),
        ]
    )

    result = await optimize.run_optimization(
        session, task, repo_path=OPT_FIXTURE, model=llm, scratch_root=tmp_path
    )

    assert result.report.status == "improved"
    assert result.report.reverted is False
    assert result.report.improvement_pct >= 10.0
    # Measured, not the 99.9 the model claimed.
    assert result.report.baseline_ms != 999.0
    assert result.report.optimized_ms != 1.0
    assert "app/aggregate.py" in result.report.files_changed
    # The model's own words survive -- that part IS its judgment.
    assert "set" in result.report.technique


async def test_optimizer_reverts_a_below_threshold_change(session, tmp_path):
    """6.6's actual safety property: a marginal change is REVERTED, not
    shipped -- even when the model reports a large improvement."""
    from amop.orchestrator.task import create_task

    task = await create_task(session, task_type="optimization", task_context={})
    llm = ScriptedLLM(
        [
            _tool_call("profile_code", path="benchmark.py"),
            _tool_call("write_file", path="app/aggregate.py", content=_COSMETIC_AGGREGATE),
            _final(
                {
                    "task_id": "overwritten",
                    "status": "improved",
                    "baseline_ms": 500.0,
                    "optimized_ms": 5.0,
                    "improvement_pct": 99.0,
                    "technique": "made it dramatically faster",
                }
            ),
        ]
    )

    result = await optimize.run_optimization(
        session, task, repo_path=OPT_FIXTURE, model=llm, scratch_root=tmp_path
    )

    assert result.report.status == "no_improvement"
    assert result.report.reverted is True
    assert result.report.improvement_pct < 10.0
    # And the file really went back -- the slow implementation is intact.
    scratch = tmp_path / str(task.id)
    restored = (scratch / "app" / "aggregate.py").read_text()
    assert "seen: list[str] = []" in restored
    assert "nicer comment" not in restored


_BROKEN_BUT_FAST = '"""Report aggregation.\n\nOptimized: the quadratic membership scan has been removed in favor of a\nsingle pass that tracks the running count directly, which avoids\nre-scanning the accumulated list on every element.\n"""\n\n\ndef count_unique_visitors(visitor_ids: list[str]) -> int:\n    """Count distinct visitor ids in a single pass."""\n    unique_count = 0\n    for _visitor in visitor_ids:\n        # Intentionally wrong: this counts nothing, but it is fast and\n        # it *looks* like a plausible single-pass rewrite.\n        pass\n    return unique_count\n\n\ndef summarize(visitor_ids: list[str]) -> dict:\n    return {\n        "total_events": len(visitor_ids),\n        "unique_visitors": count_unique_visitors(visitor_ids),\n    }\n'


async def test_optimizer_reverts_a_faster_but_broken_change(session, tmp_path):
    """Not in 6.6's literal text, deliberately added: the cheapest way to
    make code fast is to make it wrong, and an agent optimizing against a
    timer has every incentive to find that out."""
    from amop.orchestrator.task import create_task

    task = await create_task(session, task_type="optimization", task_context={})
    broken_but_instant = _BROKEN_BUT_FAST
    llm = ScriptedLLM(
        [
            _tool_call("write_file", path="app/aggregate.py", content=broken_but_instant),
            _final(
                {
                    "task_id": "overwritten",
                    "status": "improved",
                    "baseline_ms": 500.0,
                    "optimized_ms": 0.1,
                    "improvement_pct": 99.9,
                    "technique": "removed the loop entirely",
                }
            ),
        ]
    )

    result = await optimize.run_optimization(
        session, task, repo_path=OPT_FIXTURE, model=llm, scratch_root=tmp_path
    )

    assert result.report.status == "no_improvement"
    assert result.report.reverted is True
    assert "test suite failed" in result.report.diagnostic
    scratch = tmp_path / str(task.id)
    assert "seen: list[str] = []" in (scratch / "app" / "aggregate.py").read_text()


async def test_optimizer_reports_no_change_honestly(session, tmp_path):
    from amop.orchestrator.task import create_task

    task = await create_task(session, task_type="optimization", task_context={})
    llm = ScriptedLLM(
        [
            _tool_call("profile_code", path="benchmark.py"),
            _final(
                {
                    "task_id": "overwritten",
                    "status": "improved",
                    "baseline_ms": 1.0,
                    "optimized_ms": 0.5,
                    "improvement_pct": 50.0,
                    "technique": "nothing, actually",
                }
            ),
        ]
    )

    result = await optimize.run_optimization(
        session, task, repo_path=OPT_FIXTURE, model=llm, scratch_root=tmp_path
    )

    assert result.report.status == "no_improvement"
    assert result.report.files_changed == []
    assert result.report.diagnostic == "no change was made"


def test_drop_filler_issues_removes_padding_but_keeps_real_content():
    """Found live in Checkpoint 3: asked for the notable themes in a
    window, the model gave three real ones and padded with "N/A". The
    prompt already asks it not to pad; this makes it a guarantee."""
    issues = [
        "Calculator.average() returns the wrong value",
        "N/A",
        "The CSV export drops the last row",
        "None.",
        "  ",
    ]
    assert reporting.drop_filler_issues(issues) == [
        "Calculator.average() returns the wrong value",
        "The CSV export drops the last row",
    ]


def test_drop_filler_issues_can_empty_the_list():
    # A genuinely quiet window should report nothing, not "N/A".
    assert reporting.drop_filler_issues(["N/A", "none"]) == []


def test_enforce_verified_counts_also_strips_filler():
    window = ReportWindow(
        period_start=datetime(2026, 1, 1, tzinfo=UTC),
        period_end=datetime(2026, 1, 8, tzinfo=UTC),
    )
    summary = ReportSummary(
        period_start="x", period_end="y", top_issues=["a real finding", "N/A"]
    )
    assert reporting.enforce_verified_counts(summary, window).top_issues == [
        "a real finding"
    ]
