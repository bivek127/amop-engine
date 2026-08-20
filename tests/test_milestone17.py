"""Milestone 17 — Diagnose the Local-Model Diff-Construction Gap.

`PATCH_CONFLICT` recurred across Milestones 11-12 (Coder on
`invoke/runners.py`) and 14 (Optimizer, live twice -- once independently
by the human). None of the raw historical data survived anywhere
(checked: zero `patch_file` rows in `amop_dev`, zero real traces across
every local session transcript including all subagent sidechains) --
this milestone's real evidence came from a fresh live rerun against the
exact same fixture (`tests/fixtures/slow_report`) and model
(`qwen2.5-coder:14b`), reproduced byte-for-byte against a real `git
apply`, not guessed at.

Two distinct, independently-confirmed causes, previously both hiding
under one error code:

1. **Missing trailing newline.** The model's raw diff output
   consistently omits the newline after the last hunk line, which `git
   apply` rejects outright as "corrupt patch at line N" regardless of
   whether the diff's content is otherwise correct. Confirmed as the
   SOLE cause of one recovered failure: adding nothing but `\\n` made it
   apply cleanly.
2. **Phantom/stale context.** A separate, deeper defect found in two
   other recovered attempts: the diff lists a line as unchanged
   *context* whose text doesn't match the file's real current content --
   it matches what an EARLIER (also-rejected) attempt in the same retry
   loop would have produced, had it landed. The model appears to build
   each retry against its own intended end state rather than the file's
   real, still-unedited content -- consistent with each retry's
   `read_file` call requesting a range that stopped one line short of
   the actual edit site.

`REAL_HISTORICAL_DIFF_2` below is the literal diff text recovered from
Postgres (`task_context.tool_calls`) for the live rerun's second
`patch_file` attempt -- quoted verbatim, not reconstructed, so this
suite pins the actual failure, not a simplified stand-in for it.
"""

from pathlib import Path

import pytest

from amop.sandbox.manager import SandboxManager
from amop.sandbox.tools import _diagnose_context_mismatch, _normalize_diff_headers
from amop.tools.registry import ToolContext, invoke_tool

# The real fixture file's relevant region at the time of the live rerun
# (tests/fixtures/slow_report/app/aggregate.py, lines 14-23) -- included
# here as a fixed, known-good baseline so these tests don't depend on
# that fixture file never changing in the future. Padded with 13 filler
# lines so line numbers line up with REAL_HISTORICAL_DIFF_2's own
# `@@ -14,...` header below, which is quoted verbatim (real absolute line
# numbers from the actual file) and deliberately left un-renumbered.
AGGREGATE_SNIPPET = "\n".join(f"# filler {n}" for n in range(1, 14)) + "\n"
AGGREGATE_SNIPPET += (
    "def count_unique_visitors(visitor_ids: list[str]) -> int:\n"
    '    """Count distinct visitor ids.\n'
    "\n"
    "    Correct, but quadratic: `in` over a list is a full scan each time.\n"
    '    """\n'
    "    seen: list[str] = []\n"
    "    for visitor in visitor_ids:\n"
    "        if visitor not in seen:\n"
    "            seen.append(visitor)\n"
    "    return len(seen)\n"
)

# Recovered verbatim from Postgres, task 09d7b2d1-b039-4e2f-9e9b-282bd2e4a2e5,
# tool_calls[4] (the SECOND of three patch_file attempts, all against the
# real fixture). Note the raw text has NO trailing newline -- reproduced
# exactly, including that omission, since it's the first defect these
# tests pin.
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


# ---------------------------------------------------------------------
# _normalize_diff_headers -- trailing-newline fix. Pure, no sandbox.
# ---------------------------------------------------------------------


def test_normalize_diff_headers_adds_a_missing_trailing_newline():
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-old\n+new"  # no trailing \n
    result = _normalize_diff_headers(diff, "x.py")
    assert result.endswith("\n")


def test_normalize_diff_headers_is_idempotent_when_newline_already_present():
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
    result = _normalize_diff_headers(diff, "x.py")
    assert result == diff


def test_normalize_diff_headers_fixes_the_real_historical_diff():
    """The regression pin: the exact diff that failed live now comes out
    newline-terminated."""
    result = _normalize_diff_headers(REAL_HISTORICAL_DIFF_2, "app/aggregate.py")
    assert result.endswith("\n")


# ---------------------------------------------------------------------
# _diagnose_context_mismatch -- phantom-context detection. Pure, no
# sandbox. Never decides accept/reject; only explains a rejection that
# already happened.
# ---------------------------------------------------------------------


def test_diagnose_context_mismatch_finds_the_real_historical_defect():
    normalized = _normalize_diff_headers(REAL_HISTORICAL_DIFF_2, "app/aggregate.py")
    detail = _diagnose_context_mismatch(normalized, AGGREGATE_SNIPPET)
    assert detail is not None
    assert "seen.append(visitor)" in detail  # the file's real content
    assert "seen.add(visitor)" in detail  # what the diff wrongly assumed
    assert "22" in detail  # the exact line, not a vague "somewhere"


def test_diagnose_context_mismatch_is_silent_when_only_the_newline_was_wrong():
    """The other real attempt from the same rerun: newline-only defect,
    no content mismatch. Must not report a false mismatch."""
    diff = (
        "--- a/app/aggregate.py\n"
        "+++ b/app/aggregate.py\n"
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
        "-        if visitor not in seen:\n"
        "+        if visitor not in seen:\n"
        "             seen.append(visitor)\n"
        "     return len(seen)\n"
    )
    assert _diagnose_context_mismatch(diff, AGGREGATE_SNIPPET) is None


def test_diagnose_context_mismatch_ignores_added_lines():
    """A '+' line never has to match anything in the old file -- only
    ' ' and '-' lines consume an old-file line number."""
    diff = (
        "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,3 @@\n"
        " line one\n+brand new line\n line two\n"
    )
    assert _diagnose_context_mismatch(diff, "line one\nline two\n") is None


# ---------------------------------------------------------------------
# Real Docker sandbox, real patch_file -- no mock sandbox path, same
# discipline Milestone 11 established.
# ---------------------------------------------------------------------


@pytest.fixture
def sandbox_manager():
    manager = SandboxManager()
    yield manager
    manager.destroy_all()


def make_ctx(scratch_dir, sandbox, mode="suggestor"):
    return ToolContext(agent_name="test", scratch_dir=scratch_dir, mode=mode, sandbox=sandbox)


async def test_patch_file_applies_a_diff_missing_its_trailing_newline(
    tmp_path, sandbox_manager
):
    """Regression pin for Finding 1: a diff that is otherwise perfectly
    valid, but (like every recovered historical case) omits the final
    newline, must now apply -- this exact shape used to fail with
    'corrupt patch', full stop, before Milestone 17."""
    (tmp_path / "target.txt").write_text("line 1\nline 2\nline 3\n")
    sandbox = sandbox_manager.create("t-patch-no-nl", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        diff_no_trailing_newline = (
            "--- a/target.txt\n+++ b/target.txt\n@@ -1,3 +1,3 @@\n"
            " line 1\n-line 2\n+line TWO\n line 3"  # no trailing \n
        )
        result = await invoke_tool(
            "patch_file",
            {"path": "target.txt", "diff": diff_no_trailing_newline},
            ctx,
            agent_name="test",
        )
        assert result.success, result.message
        assert (tmp_path / "target.txt").read_text() == "line 1\nline TWO\nline 3\n"
    finally:
        sandbox_manager.destroy("t-patch-no-nl")


async def test_patch_file_conflict_message_names_the_exact_mismatch(
    tmp_path, sandbox_manager
):
    """Regression pin for Finding 2: a phantom-context diff still
    correctly refuses (conflict detection stays git apply's job,
    unchanged) but the message now names the real line and content
    instead of git's opaque 'corrupt patch'/'does not apply'."""
    (tmp_path / "target.txt").write_text("line 1\nline 2\nline 3\n")
    sandbox = sandbox_manager.create("t-patch-phantom", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        phantom_context_diff = (
            "--- a/target.txt\n+++ b/target.txt\n@@ -1,3 +1,3 @@\n"
            " line 1\n-line 2\n+line TWO\n line THREE-not-really\n"
        )
        result = await invoke_tool(
            "patch_file",
            {"path": "target.txt", "diff": phantom_context_diff},
            ctx,
            agent_name="test",
        )
        assert not result.success
        assert result.error_code == "PATCH_CONFLICT"
        assert "line 3" in result.message  # the real content
        assert "line THREE-not-really" in result.message  # what the diff wrongly assumed
        assert (tmp_path / "target.txt").read_text() == "line 1\nline 2\nline 3\n"
    finally:
        sandbox_manager.destroy("t-patch-phantom")


async def test_patch_file_still_applies_the_milestone_11_clean_diff(
    tmp_path, sandbox_manager
):
    """Non-regression: a normal, already-well-formed diff (Milestone 11's
    own case) is completely unaffected by either Milestone 17 change."""
    (tmp_path / "target.txt").write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")
    sandbox = sandbox_manager.create("t-patch-still-clean", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        clean_diff = (
            "--- a/target.txt\n+++ b/target.txt\n@@ -1,5 +1,5 @@\n"
            " line 1\n line 2\n-line 3\n+line THREE\n line 4\n line 5\n"
        )
        result = await invoke_tool(
            "patch_file", {"path": "target.txt", "diff": clean_diff}, ctx, agent_name="test"
        )
        assert result.success, result.message
        assert (tmp_path / "target.txt").read_text() == (
            "line 1\nline 2\nline THREE\nline 4\nline 5\n"
        )
    finally:
        sandbox_manager.destroy("t-patch-still-clean")
