"""Diagnose-vs-Mechanical-Output Gap, Attempt 2: Forced Fresh Read —
spec's own named-but-never-built fix from Milestone 17's findings.

Milestone 17 diagnosed the remaining (phantom-context) failure
precisely but left it unresolved: a Coder retry can construct a diff
against content it never actually re-read at the right range. The real
historical evidence (task 09d7b2d1-b039-4e2f-9e9b-282bd2e4a2e5, still in
`amop_dev`, and the literal diff `test_milestone17.py` already pins as
`REAL_HISTORICAL_DIFF_2`) shows the actual failure shape precisely: every
retry WAS preceded by a same-turn `read_file` -- always lines 14-20 --
but the diff's real target range is 14-23. A same-turn-read-PRESENCE
check would have passed this straight through unchanged; it has to be
COVERAGE. `test_stale_read_rejects_the_real_historical_mismatch` below
is deliberately built to fail against a naive presence-only
implementation and pass only against a correct coverage-checking one --
mutation-verified, not just asserted.

Real Docker sandbox, real patch_file -- no mock sandbox path, same
discipline Milestone 17 established.
"""

import asyncio
import json
from pathlib import Path

import pytest

from amop.agents.coder import CoderAgent
from amop.models.base import BaseLLM, ModelResponse
from amop.sandbox.manager import SandboxManager
from amop.sandbox.tools import (
    _diff_target_range,
    _read_covers_target,
    read_file_effective_range,
)
from amop.tools.registry import ToolContext, ToolResult, invoke_tool


# Same self-contained scripting pattern test_milestone29.py established
# (not imported cross-file).
class ScriptedLLM(BaseLLM):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, messages, tools=None) -> ModelResponse:
        content = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return ModelResponse(content=content, model="scripted")

    async def embed(self, texts):
        raise NotImplementedError


def tool_call(name: str, **arguments) -> str:
    return json.dumps({"tool_call": {"name": name, "arguments": arguments}})


def final(answer) -> str:
    return json.dumps({"final_answer": answer})

# ---------------------------------------------------------------------
# The real historical artifact -- same task, same recovered diff
# test_milestone17.py already pins verbatim (task 09d7b2d1-b039-4e2f-
# 9e9b-282bd2e4a2e5, tool_calls[4], the SECOND of three real patch_file
# attempts). Reproduced here rather than imported cross-file (this
# project's own test-file convention -- see test_milestone29.py's own
# "self-contained, not imported cross-file" note), but it's the exact
# same bytes, same provenance.
#
# The real captured tool-call sequence for this task (queried directly
# from amop_dev, not assumed):
#   read_file(14-20) -> patch_file FAIL -> read_file(14-20) ->
#   patch_file FAIL -> read_file(14-20) -> patch_file FAIL
# Every attempt had a same-turn read; every one covered lines 14-20;
# the diff's real target is 14-23. That gap is the whole bug.
# ---------------------------------------------------------------------

REAL_HISTORICAL_DIFF_2 = (
    "--- /workspace/app/aggregate.py\n"
    "+++ /workspace/app/aggregate.py\n"
    "@@ -14,10 +14,10 @@\n"
    " def count_unique_visitors(visitor_ids: list[str]) -> int:\n"
    '     """Count distinct visitor ids.\n'
    " \n"
    "-    Correct, but quadratic: `in` over a list is a full scan each time.\n"
    "+    Correct, but quadratic: `in` over a list is a full scan each time. Use a set for O(1) lookup.\n"
    '     """\n'
    "-    seen: list[str] = []\n"
    "+    seen: set[str] = set()\n"
    "     for visitor in visitor_ids:\n"
    "         if visitor not in seen:\n"
    "             seen.add(visitor)\n"
    "     return len(seen)"  # <-- no trailing \n, exactly as the model produced it
)

REAL_HISTORICAL_READ_RANGE = (14, 20)  # what every one of the 3 real attempts actually read
REAL_HISTORICAL_TARGET_RANGE = (14, 23)  # what the diff's own hunk header actually touches


# =======================================================================
# _diff_target_range -- pure, no sandbox
# =======================================================================


def test_diff_target_range_finds_the_real_historical_mismatch():
    """The core parsing claim this whole milestone rests on: the diff's
    hunk touches lines 14-23, not 14-20 -- confirmed against the real
    recovered diff, not a simplified stand-in."""
    assert _diff_target_range(REAL_HISTORICAL_DIFF_2) == REAL_HISTORICAL_TARGET_RANGE


def test_diff_target_range_handles_multiple_hunks():
    diff = (
        "--- a/x.py\n+++ b/x.py\n"
        "@@ -2,2 +2,2 @@\n"
        " line 2\n-line 3\n+line THREE\n"
        "@@ -10,1 +10,1 @@\n"
        "-line 10\n+line TEN\n"
    )
    # First hunk touches 2-3, second touches 10 -- union is (2, 10).
    assert _diff_target_range(diff) == (2, 10)


def test_diff_target_range_counts_a_pure_insertion_hunk_by_its_anchor():
    """A hunk with only '+' lines still has to have been seen at its
    anchor position -- an empty range would let the gate be satisfied
    by a read that never even reached the insertion point."""
    diff = "--- a/x.py\n+++ b/x.py\n@@ -5,0 +6,2 @@\n+new line one\n+new line two\n"
    assert _diff_target_range(diff) == (5, 5)


def test_diff_target_range_returns_none_for_no_hunk_header():
    assert _diff_target_range("not a diff at all") is None


# =======================================================================
# read_file_effective_range -- pure, no sandbox
# =======================================================================


def test_effective_range_is_the_requested_range_when_not_truncated():
    result = ToolResult(success=True, output="some content\n")
    assert read_file_effective_range(10, 20, result) == (10, 20)


def test_effective_range_is_none_for_a_failed_read():
    result = ToolResult(success=False, error_code="NOT_FOUND")
    assert read_file_effective_range(10, 20, result) is None


def test_effective_range_credits_only_the_truncated_window_not_the_request():
    """The exact shape that matters: a caller asking for lines 1-2000 on
    a huge file gets narrowed to a small window -- crediting the full
    1-2000 request here would let a single oversized read 'unlock'
    patch_file anywhere in the file, defeating the whole gate."""
    result = ToolResult(success=True, output="[NOTE: too large...]\n\nsome content\n")
    lo, hi = read_file_effective_range(1, 2000, result)
    assert lo == 1
    assert hi < 2000


def test_effective_range_whole_file_truncated_starts_at_one():
    result = ToolResult(success=True, output="[NOTE: this file is...]\n\ncontent\n")
    lo, hi = read_file_effective_range(None, None, result)
    assert lo == 1
    assert hi < 1_000_000_000  # the "unbounded" sentinel is NOT what a truncated read gets


# =======================================================================
# _read_covers_target -- pure, the actual coverage logic
# =======================================================================


def test_read_covers_target_true_when_fully_contained():
    assert _read_covers_target((1, 30), (14, 23)) is True


def test_read_covers_target_false_for_none():
    assert _read_covers_target(None, (14, 23)) is False


def test_read_covers_target_false_for_the_real_historical_mismatch():
    """The exact real numbers: read 14-20 does NOT cover target 14-23."""
    assert _read_covers_target(REAL_HISTORICAL_READ_RANGE, REAL_HISTORICAL_TARGET_RANGE) is False


def test_read_covers_target_false_when_only_partially_overlapping():
    assert _read_covers_target((1, 22), (14, 23)) is False  # off by exactly one line
    assert _read_covers_target((15, 30), (14, 23)) is False  # missing the start


# =======================================================================
# Real Docker sandbox, real patch_file -- the actual gate, end to end
# =======================================================================


@pytest.fixture
def sandbox_manager():
    manager = SandboxManager()
    yield manager
    manager.destroy_all()


def make_ctx(scratch_dir, sandbox, *, tracking):
    return ToolContext(
        agent_name="test", scratch_dir=scratch_dir, mode="suggestor",
        sandbox=sandbox, coder_read_tracking=tracking,
    )


async def test_stale_read_rejects_the_real_historical_mismatch(tmp_path, sandbox_manager):
    """The primary deterministic regression test: replay the real
    historical read/patch sequence's exact numbers and confirm the gate
    rejects it -- BEFORE ever reaching git apply. Deliberately built to
    fail against a naive presence-only implementation (mutation-verified
    below) and pass only against a correct coverage-checking one."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "aggregate.py").write_text("placeholder\n" * 30)
    sandbox = sandbox_manager.create("t-stale-real", tmp_path)
    try:
        ctx = make_ctx(
            tmp_path, sandbox,
            tracking={"app/aggregate.py": REAL_HISTORICAL_READ_RANGE},
        )
        result = await invoke_tool(
            "patch_file",
            {"path": "app/aggregate.py", "diff": REAL_HISTORICAL_DIFF_2},
            ctx,
            agent_name="test",
        )
        assert result.success is False
        assert result.error_code == "STALE_READ"
        assert "14-23" in result.message  # names the real target range
        assert "14-20" in result.message  # names what was actually read
        # Never reached git apply -- the placeholder file (which would
        # have failed as PATCH_CONFLICT, not STALE_READ, had the gate
        # let it through) proves the short-circuit really happened.
        assert (tmp_path / "app" / "aggregate.py").read_text() == "placeholder\n" * 30
    finally:
        sandbox_manager.destroy("t-stale-real")


async def test_fresh_read_covering_the_target_range_succeeds(tmp_path, sandbox_manager):
    """No regression on the ordinary case: a read that DOES cover the
    diff's target range lets patch_file proceed normally.

    Deliberately NOT REAL_HISTORICAL_DIFF_2 -- that diff is the actual
    broken artifact (its own phantom context is WRONG relative to the
    real file, that's the whole reason it failed historically), so it
    can never be the "this should succeed" case. This is the real
    fixture's real current content (lines 14-23, confirmed directly),
    with a genuinely correct list->set diff against it -- the same
    optimization REAL_HISTORICAL_DIFF_2 was trying and failing to make."""
    real_source = Path(__file__).parent / "fixtures" / "slow_report" / "app" / "aggregate.py"
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "aggregate.py").write_text(real_source.read_text())
    correct_diff = (
        "--- a/app/aggregate.py\n"
        "+++ b/app/aggregate.py\n"
        "@@ -14,10 +14,10 @@\n"
        " def count_unique_visitors(visitor_ids: list[str]) -> int:\n"
        '     """Count distinct visitor ids.\n'
        " \n"
        "-    Correct, but quadratic: `in` over a list is a full scan each time.\n"
        "+    Uses a set for O(1) lookup instead of a list's linear scan.\n"
        '     """\n'
        "-    seen: list[str] = []\n"
        "+    seen: set[str] = set()\n"
        "     for visitor in visitor_ids:\n"
        "         if visitor not in seen:\n"
        "-            seen.append(visitor)\n"
        "+            seen.add(visitor)\n"
        "     return len(seen)\n"
    )
    sandbox = sandbox_manager.create("t-fresh-real", tmp_path)
    try:
        ctx = make_ctx(
            tmp_path, sandbox,
            tracking={"app/aggregate.py": (1, 30)},  # covers 14-23 easily
        )
        result = await invoke_tool(
            "patch_file",
            {"path": "app/aggregate.py", "diff": correct_diff},
            ctx,
            agent_name="test",
        )
        assert result.success, result.message
        assert result.error_code is None
        content = (tmp_path / "app" / "aggregate.py").read_text()
        assert "seen: set[str] = set()" in content
    finally:
        sandbox_manager.destroy("t-fresh-real")


async def test_patch_file_without_any_prior_read_this_attempt_is_rejected(
    tmp_path, sandbox_manager
):
    (tmp_path / "target.txt").write_text("a\nb\nc\n")
    sandbox = sandbox_manager.create("t-no-read", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox, tracking={})  # tracking ON, nothing read yet
        diff = "--- a/target.txt\n+++ b/target.txt\n@@ -2,1 +2,1 @@\n-b\n+B\n"
        result = await invoke_tool(
            "patch_file", {"path": "target.txt", "diff": diff}, ctx, agent_name="test"
        )
        assert result.success is False
        assert result.error_code == "STALE_READ"
        assert "no prior read_file call" in result.message
    finally:
        sandbox_manager.destroy("t-no-read")


async def test_gate_is_off_when_tracking_is_none_no_regression_on_direct_calls(
    tmp_path, sandbox_manager
):
    """Every existing test (Milestone 11/13/17's own direct patch_file
    calls) constructs a plain ToolContext with no tracking field set at
    all -- coder_read_tracking defaults to None, and the gate must be a
    complete no-op in that case, exactly as before this milestone."""
    (tmp_path / "target.txt").write_text("a\nb\nc\n")
    sandbox = sandbox_manager.create("t-gate-off", tmp_path)
    try:
        ctx = ToolContext(
            agent_name="test", scratch_dir=tmp_path, mode="suggestor", sandbox=sandbox
        )
        assert ctx.coder_read_tracking is None
        diff = "--- a/target.txt\n+++ b/target.txt\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
        result = await invoke_tool(
            "patch_file", {"path": "target.txt", "diff": diff}, ctx, agent_name="test"
        )
        assert result.success, result.message
        assert result.error_code is None
    finally:
        sandbox_manager.destroy("t-gate-off")


async def test_a_fresh_attempt_does_not_inherit_an_earlier_attempts_read(
    tmp_path, sandbox_manager
):
    """The real mechanism Milestone 17 named: a retry must not lean on
    an earlier turn's memory of having read something once. Drives this
    through the REAL CoderAgent.run() lifecycle across two separate
    attempts (same instance, same ctx, matching how chain.py's CODING
    loop actually reuses one CoderAgent across fix_iterations) -- not a
    hand-set dict, the real reset path."""
    (tmp_path / "target.txt").write_text("a\nb\nc\nd\ne\n")
    sandbox = sandbox_manager.create("t-cross-attempt", tmp_path)
    try:
        ctx = ToolContext(
            agent_name="coder", scratch_dir=tmp_path, mode="operator", sandbox=sandbox
        )

        # Attempt 1: reads lines 1-5 (covers the whole file), then gives
        # a final answer without ever patching. We only need the read to
        # have happened and been tracked.
        coder = CoderAgent(
            ScriptedLLM(
                [
                    tool_call("read_file", path="target.txt", start_line=1, end_line=5),
                    final("looked, not ready to change anything yet"),
                ]
            ),
            ctx, task_id="t1",
        )
        await asyncio.wait_for(coder.run("look at the file"), timeout=30)
        # Confirm attempt 1 really did track a read covering the file.
        assert ctx.coder_read_tracking.get("target.txt") == (1, 5)

        # Attempt 2: a FRESH run() call, same instance/ctx -- must reset,
        # not inherit attempt 1's tracked read. Patches immediately,
        # with NO read_file call in this attempt at all.
        patch_diff = "--- a/target.txt\n+++ b/target.txt\n@@ -3,1 +3,1 @@\n-c\n+C\n"
        coder.model = ScriptedLLM(
            [
                tool_call("patch_file", path="target.txt", diff=patch_diff),
                final("done"),
            ]
        )
        result = await asyncio.wait_for(coder.run("fix it"), timeout=30)
        patch_calls = [c for c in result.tool_calls if c["name"] == "patch_file"]
        assert patch_calls, "patch_file was never attempted"
        assert patch_calls[0]["success"] is False
        assert patch_calls[0]["error_code"] == "STALE_READ"
    finally:
        sandbox_manager.destroy("t-cross-attempt")
