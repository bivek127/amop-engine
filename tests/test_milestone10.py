"""Milestone 10 — Diagnose & Fix Coder on Large Real Files.

Root cause, established against the real bivek127/amop-invoke-scratch
repo (not guessed -- see docs-internal/ROADMAP.md for the full
diagnostic trail): NOT the model's context-window size in general
(confirmed by a deterministic, temperature=0/seed-pinned A/B rerun that
produced byte-identical Coder behavior at two different context sizes).
Two real, independent bugs instead:

  1. indexer.py's embedding-truncation for oversized single-function
     chunks kept only the head, so a chunk whose relevant content sits
     past that cutoff (pyinvoke/invoke's Runner.run: the `/bin/bash`
     default is at char 10,328 of a 12,674-char chunk, past the old
     8,000-char head-only window) never got a representative embedding,
     and semantic search never surfaced it.
  2. Coder had no fallback to read_file the Investigator's own
     affected_files directly when search_code came back empty/
     irrelevant -- it just gave up, despite already having a confident,
     specific answer in hand.

A third, smaller gap surfaced while confirming fix #2 against the real
repo: read_file had no way to read part of a file, so a large real file
(runners.py, 65,509 chars) still silently overflowed context on a
whole-file read. Fixed with optional start_line/end_line plus a
prepended (not appended -- truncation drops the END of an oversized
message, confirmed directly) size notice.

Tiers, same discipline as every prior milestone: pure/scripted tests
here need only Postgres-free, Docker-based sandbox fixtures (tier a/b);
no Ollama required for any test in this file -- the actual real-model
confirmation was done live, once, and is not re-run in CI (see
docs-internal/ROADMAP.md for that transcript).
"""

import json
import uuid
from pathlib import Path

import pytest

from amop.agents.coder import CoderAgent
from amop.agents.handoffs import RootCauseReport
from amop.models.base import BaseLLM, ModelResponse
from amop.orchestrator.chain import ChainResult, _run_coder
from amop.sandbox import repo as git_repo
from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext, invoke_tool

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "buggy_calculator"


# ---------------------------------------------------------------------
# Tier (a) — Coder prompt construction, real Docker sandbox (needed for
# _run_coder's git operations), scripted LLM (no Ollama)
# ---------------------------------------------------------------------


class RecordingLLM(BaseLLM):
    """Same shape as the other milestones' ScriptedLLM, but this file
    only cares about what it was ASKED (llm.prompts), not chain routing,
    so a single canned final_answer is enough."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[list[dict]] = []

    async def complete(self, messages, tools=None) -> ModelResponse:
        self.prompts.append(list(messages))
        return ModelResponse(content=self._response, model="scripted")

    async def embed(self, texts):
        raise NotImplementedError


def _final(answer: str) -> str:
    return json.dumps({"final_answer": answer})


@pytest.fixture
def workspace(tmp_path):
    scratch = tmp_path / "workspace"
    git_repo.materialize(FIXTURE_REPO, scratch)

    manager = SandboxManager()
    task_id = f"m10-{uuid.uuid4().hex[:8]}"
    sandbox = manager.create(task_id, scratch)
    try:
        git_repo.init_baseline(sandbox)
        git_repo.create_branch(sandbox, "amop/fix-test")
        ctx = ToolContext(
            agent_name="chain", scratch_dir=scratch, mode="operator", sandbox=sandbox
        )
        yield ctx, scratch, sandbox
    finally:
        manager.destroy(task_id)


async def test_run_coder_prompt_puts_known_affected_files_first(workspace):
    """Milestone 10 root cause #2's fix: when the Investigator already
    named specific files, the Coder's prompt tells it to read them
    directly, not re-discover them via search_code."""
    ctx, _scratch, _sandbox = workspace
    llm = RecordingLLM(_final("nothing to do"))
    coder = CoderAgent(llm, ctx, task_id="t1")
    report = RootCauseReport(
        task_id="t1",
        root_cause="the shell path is hardcoded",
        confidence=0.9,
        affected_files=["invoke/runners.py"],
        suggested_fix_plan="resolve the shell path at runtime",
    )
    chain_result = ChainResult(task=None, final_state=None)

    await _run_coder(coder, ctx, report, "", "t1", chain_result)

    assert len(llm.prompts) == 1
    prompt_text = llm.prompts[0][-1]["content"]
    assert "Affected files (from the investigation" in prompt_text
    assert "read these directly with read_file first" in prompt_text
    assert "invoke/runners.py" in prompt_text


async def test_run_coder_prompt_omits_affected_files_line_when_none_given(workspace):
    """No known target files -- the prompt shouldn't claim one exists or
    crash formatting an empty list."""
    ctx, _scratch, _sandbox = workspace
    llm = RecordingLLM(_final("nothing to do"))
    coder = CoderAgent(llm, ctx, task_id="t1")
    report = RootCauseReport(
        task_id="t1",
        root_cause="unclear",
        confidence=0.5,
        affected_files=[],
        suggested_fix_plan="investigate further",
    )
    chain_result = ChainResult(task=None, final_state=None)

    await _run_coder(coder, ctx, report, "", "t1", chain_result)

    prompt_text = llm.prompts[0][-1]["content"]
    assert "Affected files" not in prompt_text


def test_coder_system_prompt_prefers_known_files_over_search():
    ctx = ToolContext(agent_name="coder", scratch_dir=Path("."), mode="suggestor")
    coder = CoderAgent(RecordingLLM(_final("x")), ctx, task_id="t1")

    prompt = coder.system_prompt()

    assert "start by calling read_file directly" in prompt
    assert "not a reason to give up" in prompt


# ---------------------------------------------------------------------
# Tier (b) — read_file line-range + large-file notice, real Docker
# sandbox (Milestone 3's "no mock sandbox path" discipline)
# ---------------------------------------------------------------------


@pytest.fixture
def sandbox_manager():
    manager = SandboxManager()
    yield manager
    manager.destroy_all()


def make_ctx(scratch_dir, sandbox, mode="suggestor"):
    return ToolContext(agent_name="test", scratch_dir=scratch_dir, mode=mode, sandbox=sandbox)


async def test_read_file_start_end_line_returns_only_that_slice(tmp_path, sandbox_manager):
    lines = [f"line {i}\n" for i in range(1, 21)]
    (tmp_path / "big.txt").write_text("".join(lines))
    sandbox = sandbox_manager.create("t-range", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "read_file",
            {"path": "big.txt", "start_line": 5, "end_line": 8},
            ctx,
            agent_name="test",
        )
        assert result.success, result.message
        assert result.output == "line 5\nline 6\nline 7\nline 8\n"
    finally:
        sandbox_manager.destroy("t-range")


async def test_read_file_start_line_only_reads_to_end_of_file(tmp_path, sandbox_manager):
    lines = [f"line {i}\n" for i in range(1, 6)]
    (tmp_path / "small.txt").write_text("".join(lines))
    sandbox = sandbox_manager.create("t-range-open", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "read_file", {"path": "small.txt", "start_line": 3}, ctx, agent_name="test"
        )
        assert result.success, result.message
        assert result.output == "line 3\nline 4\nline 5\n"
    finally:
        sandbox_manager.destroy("t-range-open")


async def test_read_file_range_out_of_bounds_is_invalid_args(tmp_path, sandbox_manager):
    (tmp_path / "tiny.txt").write_text("only one line\n")
    sandbox = sandbox_manager.create("t-range-oob", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "read_file",
            {"path": "tiny.txt", "start_line": 50, "end_line": 60},
            ctx,
            agent_name="test",
        )
        assert not result.success
        assert result.error_code == "INVALID_ARGS"
    finally:
        sandbox_manager.destroy("t-range-oob")


async def test_read_file_without_range_still_returns_whole_small_file(tmp_path, sandbox_manager):
    # Regression guard: adding start_line/end_line must not change the
    # existing no-args behavior for a normal-sized file.
    (tmp_path / "note.txt").write_text("just a small file")
    sandbox = sandbox_manager.create("t-range-unaffected", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool("read_file", {"path": "note.txt"}, ctx, agent_name="test")
        assert result.success
        assert result.output == "just a small file"
    finally:
        sandbox_manager.destroy("t-range-unaffected")


async def test_read_file_large_whole_file_gets_a_prepended_notice(tmp_path, sandbox_manager):
    """Milestone 10 root cause #3, reproduced as a regression guard: a
    file bigger than a model can take in whole (pyinvoke/invoke's real
    runners.py is 65,509 chars) must carry an in-band, front-loaded
    warning -- appended would get silently dropped by the same
    truncation this is warning about (confirmed: truncation drops the
    END of an oversized message, not the start)."""
    big_content = "x = 1\n" * 2000  # well over the 8,000-char notice threshold
    (tmp_path / "huge.py").write_text(big_content)
    sandbox = sandbox_manager.create("t-large-notice", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool("read_file", {"path": "huge.py"}, ctx, agent_name="test")
        assert result.success
        assert result.output.startswith("[NOTE:")
        assert "start_line/end_line" in result.output.split("\n\n", 1)[0]
        # The full, real content still follows -- nothing was actually
        # cut, only a heads-up was added.
        assert big_content in result.output
    finally:
        sandbox_manager.destroy("t-large-notice")


async def test_read_file_ranged_read_of_large_file_has_no_notice(tmp_path, sandbox_manager):
    # The notice is a whole-file-read concern; a deliberately scoped
    # read shouldn't be prefixed with advice about a problem it doesn't
    # have.
    big_content = "x = 1\n" * 2000
    (tmp_path / "huge2.py").write_text(big_content)
    sandbox = sandbox_manager.create("t-large-ranged", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "read_file",
            {"path": "huge2.py", "start_line": 1, "end_line": 5},
            ctx,
            agent_name="test",
        )
        assert result.success
        assert not result.output.startswith("[NOTE:")
        assert result.output == "x = 1\n" * 5
    finally:
        sandbox_manager.destroy("t-large-ranged")
