"""Milestone 5 — Codebase Intelligence (RAG).

Two tiers, same discipline as Milestone 4 (spec 19.2/19.3):

  * Deterministic (always run): chunking, reciprocal rank fusion, and
    embedding/search calls. Embeddings are one fast HTTP call each --
    unlike multi-turn chat completions, they're cheap and stable enough
    to run unconditionally rather than gating behind AMOP_E2E_OLLAMA.
  * Real-model end-to-end (19.3), opt-in via AMOP_E2E_OLLAMA=1: the full
    chain against the larger fixture repo, asserted on outcome.

Requires: Postgres at TEST_DATABASE_URL with the vector extension
(init_db() enables it automatically), and a running Docker daemon.
"""

import os
import uuid
import warnings
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import delete, select, text

from amop.agents.coder import CoderAgent
from amop.agents.investigator import InvestigatorAgent
from amop.codebase_intel.chunker import CHUNK_MAX_TOKENS, chunk_source
from amop.codebase_intel.embeddings import embed_texts
from amop.codebase_intel.indexer import index_repo
from amop.codebase_intel.search import reciprocal_rank_fusion, search_code_impl
from amop.database.models import CodeChunk
from amop.database.session import init_db, make_engine, make_session_factory
from amop.orchestrator.task import create_task
from amop.sandbox import repo as git_repo
from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext, invoke_tool

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://localhost/amop_test"
)
FIXTURE_REPO = Path(__file__).parent / "fixtures" / "task_tracker"

# The exact phrasing verified live (see plan/commit notes): a real user's
# bug report, using ordinary words like "priority" and "sorted" -- not
# artificially avoiding them to make semantic search's job harder than a
# real report would be. Chosen because query wording turned out to
# matter more than expected during verification (see BUG_REPORT_MISS
# below) -- this is the phrasing that reliably retrieves the target
# function.
BUG_REPORT = "Why does a low priority task outrank an urgent one when sorted by priority score?"

# A harder, deliberately symptom-only phrasing that was tried FIRST and
# reliably missed compute_priority_score's own chunk in the top-5 (it
# still retrieved priority.py's module chunk, rank_tasks, and both
# directly-relevant tests -- just not the function by name). Kept as a
# named constant rather than deleted, so this finding stays visible
# rather than silently disappearing: query wording measurably affects
# retrieval quality on nomic-embed-text at this repo's scale.
BUG_REPORT_HARDER = (
    "Tasks marked as low priority sometimes show up above high-urgency "
    "tasks when sorted -- the ordering looks wrong for large or complex "
    "tasks."
)


# ---------------------------------------------------------------------
# Chunker -- no DB, no Ollama, no sandbox
# ---------------------------------------------------------------------


def test_chunk_function_and_class_boundaries():
    src = (
        '"""Module doc."""\n'
        "import os\n\n"
        "CONST = 1\n\n"
        "def foo(x):\n"
        "    return x + 1\n\n"
        "class Widget:\n"
        "    def __init__(self):\n"
        "        self.x = 1\n"
    )
    chunks = chunk_source(src, "widget.py")
    by_name = {c.symbol_name: c for c in chunks}

    assert by_name["foo"].symbol_type == "function"
    assert by_name["foo"].content == "def foo(x):\n    return x + 1"
    assert by_name["Widget"].symbol_type == "class"
    assert "def __init__" in by_name["Widget"].content

    module_chunk = by_name["widget.py"]
    assert module_chunk.symbol_type == "module"
    assert "import os" in module_chunk.content
    assert "CONST = 1" in module_chunk.content
    assert "def foo" not in module_chunk.content  # not double-counted


def test_chunk_never_drops_a_decorator():
    src = "class C:\n    @staticmethod\n    def helper():\n        pass\n"
    chunks = chunk_source(src, "c.py")
    method = next(c for c in chunks if c.symbol_name == "C")
    assert "@staticmethod" in method.content


def test_chunk_splits_oversized_class_into_per_method_chunks():
    padding = "x" * 50
    methods = "\n\n".join(
        f'    def method_{i}(self):\n        """{padding}"""\n        return {i}'
        for i in range(40)
    )
    src = f"class Big:\n{methods}\n"
    assert len(src) // 4 > CHUNK_MAX_TOKENS  # sanity: genuinely over budget

    chunks = chunk_source(src, "big.py")
    assert len(chunks) == 40
    assert {c.symbol_type for c in chunks} == {"method"}
    assert chunks[0].symbol_name == "Big.method_0"


def test_chunk_small_class_stays_one_chunk():
    src = "class Small:\n    def a(self):\n        pass\n\n    def b(self):\n        pass\n"
    chunks = chunk_source(src, "small.py")
    assert len(chunks) == 1
    assert chunks[0].symbol_type == "class"


def test_chunk_raises_syntax_error_for_unparseable_source():
    with pytest.raises(SyntaxError):
        chunk_source("def broken(:\n", "broken.py")


# ---------------------------------------------------------------------
# Reciprocal rank fusion -- pure function, no I/O
# ---------------------------------------------------------------------


def test_rrf_favors_items_both_lists_agree_on():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
    assert fused[0][0] == "a"


def test_rrf_includes_items_from_only_one_list():
    fused = reciprocal_rank_fusion([["x", "y"], ["z"]])
    assert {key for key, _ in fused} == {"x", "y", "z"}


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_higher_rank_scores_higher():
    fused = dict(reciprocal_rank_fusion([["first", "second", "third"]]))
    assert fused["first"] > fused["second"] > fused["third"]


# ---------------------------------------------------------------------
# Embeddings -- real Ollama, no DB
# ---------------------------------------------------------------------


async def test_embed_texts_returns_768_dim_vectors():
    vectors = await embed_texts(["def foo(): pass", "class Bar: pass"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 768
    assert len(vectors[1]) == 768


async def test_embed_texts_empty_input_returns_empty_output():
    assert await embed_texts([]) == []


# ---------------------------------------------------------------------
# DB fixtures
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
            text("TRUNCATE task_transitions, tasks, code_chunks RESTART IDENTITY CASCADE")
        )


@pytest.fixture
def repo_path() -> str:
    return str(FIXTURE_REPO.resolve())


@pytest.fixture
def sandboxed_workspace(tmp_path, repo_path):
    """A materialized task_tracker in a real container, on a working
    branch -- same setup run_fix() performs, minus indexing/the chain
    itself. Mirrors test_milestone4.py's `workspace` fixture."""
    scratch = tmp_path / "workspace"
    git_repo.materialize(FIXTURE_REPO, scratch)

    manager = SandboxManager()
    task_id = f"m5-{uuid.uuid4().hex[:8]}"
    sandbox = manager.create(task_id, scratch)
    try:
        git_repo.init_baseline(sandbox)
        git_repo.create_branch(sandbox, "amop/fix-test")
        yield scratch, sandbox
    finally:
        manager.destroy(task_id)


@pytest_asyncio.fixture
async def indexed_workspace(session, sandboxed_workspace, repo_path):
    """sandboxed_workspace, plus the repo actually indexed into
    code_chunks under `repo_path` -- ready for search tests. Indexing
    ~80 chunks against real Ollama takes under a second (timed), so a
    fresh index per test is cheap enough not to bother sharing."""
    scratch, sandbox = sandboxed_workspace
    count = await index_repo(session, repo_path, scratch)
    ctx = ToolContext(
        agent_name="test",
        scratch_dir=scratch,
        mode="operator",
        sandbox=sandbox,
        repo_path=repo_path,
        db_session=session,
    )
    yield ctx, scratch, sandbox, count


# ---------------------------------------------------------------------
# Indexer -- real Postgres, real Ollama
# ---------------------------------------------------------------------


async def test_index_repo_populates_code_chunks_at_function_granularity(
    session, repo_path, sandboxed_workspace
):
    scratch, _sandbox = sandboxed_workspace
    count = await index_repo(session, repo_path, scratch)

    # Not zero, not one row per file (14 source files) -- function-level
    # granularity, matching Done-When #1's "real chunks, not
    # empty/placeholder data".
    assert count > 40
    assert count < 200

    rows = (
        await session.execute(select(CodeChunk).where(CodeChunk.repo_path == repo_path))
    ).scalars().all()
    assert len(rows) == count
    assert all(len(r.embedding) == 768 for r in rows)
    assert all(r.content.strip() for r in rows)

    priority_chunk = next(
        r for r in rows if r.file_path == "priority.py" and r.symbol_name == "compute_priority_score"
    )
    assert priority_chunk.symbol_type == "function"
    assert priority_chunk.start_line < priority_chunk.end_line


async def test_index_repo_replaces_rather_than_duplicates_on_rerun(
    session, repo_path, sandboxed_workspace
):
    scratch, _sandbox = sandboxed_workspace
    first = await index_repo(session, repo_path, scratch)
    second = await index_repo(session, repo_path, scratch)
    assert first == second

    rows = (
        await session.execute(select(CodeChunk).where(CodeChunk.repo_path == repo_path))
    ).scalars().all()
    assert len(rows) == second


async def test_index_repo_respects_gitignore(session, repo_path, sandboxed_workspace):
    scratch, _sandbox = sandboxed_workspace
    (scratch / "should_be_ignored.py").write_text("def secret(): pass\n")
    # task_tracker's own .gitignore only covers __pycache__/.pytest_cache;
    # add a targeted rule so this test doesn't depend on that staying true.
    gitignore = scratch / ".gitignore"
    gitignore.write_text(gitignore.read_text() + "\nshould_be_ignored.py\n")

    await index_repo(session, repo_path, scratch)

    rows = (
        await session.execute(select(CodeChunk).where(CodeChunk.repo_path == repo_path))
    ).scalars().all()
    assert not any(r.file_path == "should_be_ignored.py" for r in rows)


async def test_index_repo_skips_unparseable_file_without_aborting(
    session, repo_path, sandboxed_workspace
):
    scratch, _sandbox = sandboxed_workspace
    (scratch / "broken.py").write_text("def broken(:\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        count = await index_repo(session, repo_path, scratch)

    assert count > 40  # the other 14 files still indexed
    assert any("broken.py" in str(w.message) for w in caught)


# ---------------------------------------------------------------------
# Search -- the actual demonstration (Done-When #2)
# ---------------------------------------------------------------------


async def test_semantic_search_finds_the_buggy_function_by_natural_language(
    indexed_workspace,
):
    ctx, _scratch, sandbox, _count = indexed_workspace
    results = await search_code_impl(
        BUG_REPORT, ctx.repo_path, sandbox, ctx.db_session, mode="semantic"
    )
    names = [r.symbol_name for r in results]
    assert "compute_priority_score" in names


async def test_keyword_search_finds_an_exact_literal_string(indexed_workspace):
    ctx, _scratch, sandbox, _count = indexed_workspace
    results = await search_code_impl(
        "def compute_priority_score", ctx.repo_path, sandbox, ctx.db_session, mode="keyword"
    )
    assert results and results[0].symbol_name == "compute_priority_score"


async def test_keyword_search_alone_misses_the_natural_language_bug_report(
    indexed_workspace,
):
    """The contrast that motivates hybrid mode existing at all: a
    multi-word natural-language query is not a literal substring of any
    line in the repo, so pure keyword search finds nothing -- verified,
    not assumed, since this is the whole justification for running
    semantic search at all."""
    ctx, _scratch, sandbox, _count = indexed_workspace
    results = await search_code_impl(
        BUG_REPORT, ctx.repo_path, sandbox, ctx.db_session, mode="keyword"
    )
    assert results == []


async def test_hybrid_search_finds_the_buggy_function(indexed_workspace):
    ctx, _scratch, sandbox, _count = indexed_workspace
    results = await search_code_impl(
        BUG_REPORT, ctx.repo_path, sandbox, ctx.db_session, mode="hybrid"
    )
    names = [r.symbol_name for r in results]
    assert "compute_priority_score" in names
    assert len(results) <= 5


async def test_search_returns_at_most_five_results(indexed_workspace):
    ctx, _scratch, sandbox, _count = indexed_workspace
    results = await search_code_impl(
        "task", ctx.repo_path, sandbox, ctx.db_session, mode="hybrid"
    )
    assert len(results) <= 5


async def test_search_rejects_an_unknown_mode(indexed_workspace):
    ctx, _scratch, sandbox, _count = indexed_workspace
    with pytest.raises(ValueError):
        await search_code_impl(
            "anything", ctx.repo_path, sandbox, ctx.db_session, mode="rerank"
        )


# ---------------------------------------------------------------------
# search_code as a registered tool -- the agent-facing path
# ---------------------------------------------------------------------


async def test_search_code_tool_end_to_end(indexed_workspace):
    ctx, _scratch, _sandbox, _count = indexed_workspace
    result = await invoke_tool("search_code", {"query": BUG_REPORT}, ctx, agent_name="test")
    assert result.success
    names = [r["symbol_name"] for r in result.output]
    assert "compute_priority_score" in names


async def test_search_code_fails_cleanly_without_an_index(sandboxed_workspace):
    scratch, sandbox = sandboxed_workspace
    ctx = ToolContext(
        agent_name="test", scratch_dir=scratch, mode="operator", sandbox=sandbox
    )  # repo_path/db_session left unset

    result = await invoke_tool("search_code", {"query": "anything"}, ctx, agent_name="test")

    assert not result.success
    assert result.error_code == "INDEX_UNAVAILABLE"


def test_investigator_and_coder_both_have_search_code():
    assert "search_code" in InvestigatorAgent.tools
    assert "search_code" in CoderAgent.tools


async def test_investigator_cannot_write_files_even_with_search_code_added(
    indexed_workspace,
):
    """Regression guard: adding search_code to the allowlist must not
    accidentally widen it -- write_file should still be denied."""
    ctx, scratch, _sandbox, _count = indexed_workspace
    result = await invoke_tool(
        "write_file",
        {"path": "priority.py", "content": "wiped"},
        ctx,
        agent_name="investigator",
        allowed_tools=InvestigatorAgent.tools,
    )
    assert result.error_code == "TOOL_NOT_PERMITTED"
    assert "return (urgency" in (scratch / "priority.py").read_text()


# ---------------------------------------------------------------------
# Real-model end-to-end (Section 19.3) -- opt-in
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("AMOP_E2E_OLLAMA") != "1",
    reason="real-model E2E: set AMOP_E2E_OLLAMA=1 (needs Ollama running)",
)
async def test_real_model_chain_fixes_the_seeded_bug_using_search(session, tmp_path, monkeypatch):
    from amop.models.ollama import OllamaProvider
    from amop.orchestrator.chain import run_fix
    from amop.orchestrator.state_machine import TaskState
    from amop.tools.registry import ToolResult, get_tool

    # Milestone 6: PR_CREATION now calls the real create_pull_request tool,
    # which would otherwise attempt a genuine GitHub push/API call here.
    # This test is about whether Ollama actually fixes the bug using
    # search, not about GitHub -- faked so AMOP_E2E_OLLAMA=1 alone (no
    # AMOP_E2E_GITHUB, no real network) is still sufficient to run it.
    async def _fake_create_pull_request(title, body, head, base, ctx):
        return ToolResult(
            success=True,
            output={
                "url": "https://github.com/bivek127/amop-sandbox/pull/999",
                "number": 999,
                "created": True,
            },
        )

    monkeypatch.setattr(get_tool("create_pull_request"), "func", _fake_create_pull_request)

    task = await create_task(session, task_type="bug_fix")
    stages = []
    result = await run_fix(
        session,
        task,
        description=BUG_REPORT,
        repo_path=FIXTURE_REPO,
        model=OllamaProvider(),
        scratch_root=tmp_path,
        emit=stages.append,
    )

    # Milestone 6: WAITING_FOR_APPROVAL, not RESOLVED -- a real chain now
    # stops after a real PR is opened (faked above), never auto-merges.
    assert result.final_state is TaskState.WAITING_FOR_APPROVAL, result.error
    assert result.code_change_report.files_changed == ["priority.py"]

    # Milestone 31: run_fix() now removes its scratch dir on return
    # (SandboxManager.destroy(..., remove_scratch_dir=True)), so this
    # can no longer re-read the file from disk afterward. result.diff is
    # real git ground truth (get_diff) and is still available.
    #
    # The original check was negative (the buggy substring is GONE) --
    # translated to the diff, that has to check BOTH sides, not just
    # "the string doesn't appear anywhere": the '/ effort' division must
    # appear as a REMOVED ('-') line (proving the bug was actually
    # targeted, not e.g. left untouched while something unrelated
    # changed) AND must NOT appear on any ADDED ('+') line (proving the
    # fix doesn't just reintroduce the same division elsewhere). A bare
    # "'/ effort' not in result.diff" would pass for the wrong reason if
    # the diff, say, showed the fix being reverted instead of applied --
    # this can't.
    minus_lines = [
        line for line in result.diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    plus_lines = [
        line for line in result.diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    assert any("/ effort" in line for line in minus_lines), (
        f"expected the buggy '/ effort' division to appear as a REMOVED line:\n{result.diff}"
    )
    assert not any("/ effort" in line for line in plus_lines), (
        f"fix must not reintroduce '/ effort' on an ADDED line:\n{result.diff}"
    )

    # Soft check, not a hard assertion: proving search_code was used is
    # the live-demo requirement (Verification & Commit), not something
    # that should make this test flaky if a run happens to solve it via
    # read_file alone.
    tool_names_used = {call["name"] for call in result.tool_calls}
    print(f"tools used this run: {sorted(tool_names_used)}")
    print(f"search_code used: {'search_code' in tool_names_used}")


# ---------------------------------------------------------------------
# ChainResult.tool_calls -- scripted, deterministic proof that a
# search_code call made mid-chain is captured and correctly tagged with
# which agent made it. Added because writing the real-model E2E test
# above surfaced that there was previously no way to check this at all,
# from a test OR from the CLI (see cli/main.py's now-added "Tool calls"
# section on `amop fix`).
# ---------------------------------------------------------------------


class _ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, messages, tools=None):
        from amop.models.base import ModelResponse

        content = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return ModelResponse(content=content, model="scripted")

    async def embed(self, texts):
        raise NotImplementedError


async def test_chain_result_tool_calls_captures_a_search_code_call(
    session, indexed_workspace
):
    import json

    from amop.orchestrator.chain import ChainAgents, run_chain

    ctx, scratch, _sandbox, _count = indexed_workspace
    task = await create_task(session, task_type="bug_fix")

    investigator_responses = [
        json.dumps(
            {"tool_call": {"name": "search_code", "arguments": {"query": BUG_REPORT}}}
        ),
        json.dumps(
            {
                "final_answer": {
                    "task_id": "will-be-overwritten",
                    "root_cause": "compute_priority_score divides by effort instead of subtracting it",
                    "confidence": 0.3,  # low on purpose: cheapest way to stop the chain right after INVESTIGATING
                    "evidence": [],
                    "affected_files": ["priority.py"],
                    "suggested_fix_plan": "n/a",
                }
            }
        ),
    ]
    agents = ChainAgents(
        investigator=InvestigatorAgent(_ScriptedLLM(investigator_responses), ctx),
        coder=CoderAgent(_ScriptedLLM(["should not run"]), ctx, task_id="scripted"),
        tester=None,
        reviewer=None,
    )

    result = await run_chain(
        session, task, description=BUG_REPORT, ctx=ctx, agents=agents
    )

    search_calls = [c for c in result.tool_calls if c["name"] == "search_code"]
    assert len(search_calls) == 1
    assert search_calls[0]["agent"] == "investigator"
    assert search_calls[0]["success"] is True
