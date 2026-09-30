"""Milestone 36 — pre-flight patch_file validation, pinned against the
5 real cases from Milestone 35's diagnostic run (logged in
docs-internal/ROADMAP.md's Milestone 35 Findings): the 4 real
PATCH_CONFLICT captures (ee31f113, b0ae5a10, c1cef787, d3799905) and the
one real successful patch from the same task (ac787975's 8-line header
miscount that still applied cleanly).

Every diff/current_content below is pinned verbatim from the real
captured JSON (amop_diagnostics/, gitignored -- not relied on here so
these tests are reproducible even if that directory is gone), same
"pin the literal recovered text, not a simplified stand-in" discipline
test_milestone30.py already established for REAL_HISTORICAL_DIFF_2.

Requires a real Docker daemon, same as test_milestone30.py /
test_patch_conflict_capture.py.
"""

import pytest

from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext, invoke_tool

# Shared real baseline content -- byte-identical across all 5 pinned
# cases below (all five calls happened against the same pristine
# calculator.py, before any successful edit landed on either task).
CALCULATOR_BASELINE = (
    '"""A small calculator library.\n'
    "\n"
    "Deliberately contains one seeded bug for AMOP's Milestone 4 fixture repo\n"
    "(spec Section 19.3: real code, real failing test, known-good fix so a\n"
    "chain run can be scored pass/fail). Do not \"helpfully\" fix the bug by\n"
    "hand -- the whole point is that the agent chain finds and fixes it.\n"
    '"""\n'
    "\n"
    "\n"
    "class DivisionByZeroError(ValueError):\n"
    '    """Raised instead of ZeroDivisionError so callers can catch a single\n'
    '    library-specific exception type."""\n'
    "\n"
    "\n"
    "class Calculator:\n"
    "    def add(self, a: float, b: float) -> float:\n"
    "        return a + b\n"
    "\n"
    "    def subtract(self, a: float, b: float) -> float:\n"
    "        return a - b\n"
    "\n"
    "    def multiply(self, a: float, b: float) -> float:\n"
    "        return a * b\n"
    "\n"
    "    def divide(self, a: float, b: float) -> float:\n"
    "        if b == 0:\n"
    '            raise DivisionByZeroError("cannot divide by zero")\n'
    "        return a / b\n"
    "\n"
    "    def percentage(self, value: float, total: float) -> float:\n"
    "        if total == 0:\n"
    '            raise DivisionByZeroError("cannot take a percentage of zero")\n'
    "        return (value / total) * 100\n"
    "\n"
    "    def average(self, values: list[float]) -> float:\n"
    '        """Arithmetic mean of `values`."""\n'
    "        if not values:\n"
    '            raise ValueError("cannot average an empty list")\n'
    "        return sum(values) / (len(values) - 1)\n"
)

# Case 1: task ac787975, capture ee31f113 -- preceded by a real,
# unnumbered, unscoped whole-file read_file({"path": ...}) call
# (agent_actions timestamp 2026-09-28T19:37:19, no start_line/end_line/
# with_line_numbers at all). ~15-line miscount, drops the average()
# method's docstring and guard clause.
EE31F113_DIFF = (
    "diff --git a/calculator.py b/calculator.py\n"
    "index 0123456..789abcde 100644\n"
    "--- a/calculator.py\n"
    "+++ b/calculator.py\n"
    "@@ -18,7 +18,7 @@ class Calculator:\n"
    "         return (value / total) * 100\n"
    " \n"
    "     def average(self, values: list[float]) -> float:\n"
    "-        return sum(values) / (len(values) - 1)\n"
    "+        return sum(values) / len(values)\n"
)

# Case 2: task ac787975, capture b0ae5a10 -- preceded by a real, scoped,
# NUMBERED read_file(start_line=19, end_line=22, with_line_numbers=true)
# call (agent_actions timestamp 2026-09-28T19:38:01) that correctly
# showed "19: def subtract.../20: return a-b/21: /22: def multiply...".
# The diff still claims line 19 = "return a - b" -- a uniform +1 offset
# across the whole hunk, contradicting what was just shown.
B0AE5A10_DIFF = (
    "diff --git a/calculator.py b/calculator.py\n"
    "index 0123456..789abcde 100644\n"
    "--- a/calculator.py\n"
    "+++ b/calculator.py\n"
    "@@ -19,7 +19,7 @@ class Calculator:\n"
    "         return a - b\n"
    " \n"
    "     def multiply(self, a: float, b: float) -> float:\n"
    "-        return a * b\n"
    "+        return a * b\n"
)

# Case 3: task ac787975, capture c1cef787 -- preceded by a real, scoped,
# NUMBERED read_file(start_line=21, end_line=27, with_line_numbers=true)
# call (agent_actions timestamp 2026-09-28T19:38:43) that correctly
# showed the real blank line at 24. The diff's first 3 lines match
# reality with zero offset, then drops that real blank line, shifting
# everything from "def divide" onward by +1.
C1CEF787_DIFF = (
    "diff --git a/calculator.py b/calculator.py\n"
    "index 0123456..789abcde 100644\n"
    "--- a/calculator.py\n"
    "+++ b/calculator.py\n"
    "@@ -21,10 +21,10 @@ class Calculator:\n"
    " \n"
    "     def multiply(self, a: float, b: float) -> float:\n"
    "         return a * b\n"
    "-    def divide(self, a: float, b: float) -> float:\n"
    "+    def average(self, values: list[float]) -> float:\n"
    "         if b == 0:\n"
    '             raise DivisionByZeroError("cannot divide by zero")\n'
    "-        return a / b\n"
    "+        return sum(values) / len(values)\n"
)

# Case 4: task 637c1bda, capture d3799905 -- preceded by a real,
# unnumbered, unscoped whole-file read (same shape as ee31f113). Every
# context line in this diff is missing its leading unified-diff marker
# character entirely (confirmed: after stripping one character, every
# line is exactly one space short of any real line in the file), and
# its blank-context line is a truly empty string instead of a single
# space.
D3799905_DIFF = (
    "--- calculator.py\n"
    "+++ calculator.py\n"
    "@@ -22,7 +22,7 @@\n"
    "       return (value / total) * 100\n"
    "\n"
    "    def average(self, values: list[float]) -> float:\n"
    '       """Arithmetic mean of `values`."""\n'
    "-        if not values:\n"
    "+        if not values:\n"
    '           raise ValueError("cannot average an empty list")\n'
    "-        return sum(values) / (len(values) - 1)\n"
    "+        return sum(values) / len(values)\n"
)

# Case 5: the REAL successful patch from the SAME task/session as cases
# 1-3 (ac787975, agent_actions timestamp 2026-09-28T19:40:08, "OK:
# applied patch to calculator.py"). Preceded by a real, scoped but
# UNNUMBERED read_file(start_line=1, end_line=50) call. Header claims
# line 25; the content it actually transcribes (percentage's return,
# blank, average's def+docstring+guard+return) is 100% faithful and
# really lives at line 33 -- an 8-line miscount that still applied
# cleanly, confirmed by direct git apply replay during Milestone 36's
# diagnosis. This is the case the new pre-flight checks must NOT reject.
SUCCESS_DIFF = (
    "--- calculator.py\n"
    "+++ calculator.py\n"
    "@@ -25,7 +25,7 @@ class Calculator:\n"
    "         return (value / total) * 100\n"
    " \n"
    "     def average(self, values: list[float]) -> float:\n"
    '-        """Arithmetic mean of `values`."""\n'
    '+        """Arithmetic mean of `values`."""\n'
    "         if not values:\n"
    '             raise ValueError("cannot average an empty list")\n'
    "-        return sum(values) / (len(values) - 1)\n"
    "+        return sum(values) / len(values)\n"
)


@pytest.fixture
def sandbox_manager():
    manager = SandboxManager()
    yield manager
    manager.destroy_all()


async def _run(sandbox_manager, task_id, diff, *, tracking, numbered):
    """Materializes the real pinned baseline, runs patch_file against
    it with ToolContext state matching the real historical read exactly
    (coder_read_tracking/coder_read_numbered), and returns the result."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "calculator.py").write_text(CALCULATOR_BASELINE)
        sandbox = sandbox_manager.create(task_id, tmp_path)
        try:
            ctx = ToolContext(
                agent_name="coder", scratch_dir=tmp_path, mode="suggestor",
                sandbox=sandbox,
                coder_read_tracking=dict(tracking),
                coder_read_numbered=dict(numbered),
            )
            return await invoke_tool(
                "patch_file",
                {"path": "calculator.py", "diff": diff},
                ctx,
                agent_name="coder",
            )
        finally:
            sandbox_manager.destroy(task_id)


async def test_ee31f113_unnumbered_whole_file_read_caught_by_numbered_gate(
    sandbox_manager,
):
    """Check 1 (freshness/coverage gate, extended to require numbering
    for an unscoped whole-file read): the real preceding read was
    read_file({"path": ...}) -- no start_line/end_line/with_line_numbers
    at all -- which the M30 gate's own coverage math trivially "covers"
    (range (1, UNBOUNDED_END)) but Milestone 36 now also requires to
    have been numbered for exactly this unscoped shape."""
    result = await _run(
        sandbox_manager,
        "pinned-ee31f113",
        EE31F113_DIFF,
        tracking={"calculator.py": (1, 1_000_000_000)},
        numbered={},
    )
    assert result.success is False
    assert result.error_code == "STALE_READ"
    assert "unnumbered" in result.message


async def test_b0ae5a10_content_mismatch_caught_by_content_fidelity_check(
    sandbox_manager,
):
    """Check 2 (content fidelity, strict/position-anchored mode): the
    real preceding read WAS numbered and covered the target range
    exactly, so this diff's claimed line-19 content, which contradicts
    what was just shown, is a genuine transcription fault -- caught
    before git apply, not by it."""
    result = await _run(
        sandbox_manager,
        "pinned-b0ae5a10",
        B0AE5A10_DIFF,
        tracking={"calculator.py": (19, 22)},
        numbered={"calculator.py": True},
    )
    assert result.success is False
    assert result.error_code == "CONTENT_MISMATCH"


async def test_c1cef787_dropped_blank_line_caught_by_content_fidelity_check(
    sandbox_manager,
):
    """Check 2 again, same strict mode -- the real preceding read was
    also numbered and covering, and explicitly showed the real blank
    line at 24 that this diff silently drops."""
    result = await _run(
        sandbox_manager,
        "pinned-c1cef787",
        C1CEF787_DIFF,
        tracking={"calculator.py": (21, 27)},
        numbered={"calculator.py": True},
    )
    assert result.success is False
    assert result.error_code == "CONTENT_MISMATCH"


async def test_d3799905_missing_diff_markers_caught_by_format_validation(
    sandbox_manager,
):
    """Check 3 (format validation), which runs first of all three and
    needs no file access: this diff's hunk lines are missing their
    leading unified-diff marker character. Reached even though this
    case's preceding read was ALSO the unnumbered-whole-file shape
    (same as ee31f113) -- format validation runs before the numbered
    gate, so this is the check that actually catches it."""
    result = await _run(
        sandbox_manager,
        "pinned-d3799905",
        D3799905_DIFF,
        tracking={"calculator.py": (1, 1_000_000_000)},
        numbered={},
    )
    assert result.success is False
    assert result.error_code == "MALFORMED_DIFF"


async def test_pinned_successful_8_line_miscount_case_still_allowed_through(
    sandbox_manager,
):
    """The real successful case from the same session -- must NOT
    regress. Preceded by a real scoped-but-unnumbered read, so Milestone
    36's numbered gate does not apply (not the unscoped-whole-file
    shape) and the content-fidelity check runs in its lenient,
    position-independent mode -- the transcribed content is faithful
    and appears in the real file, just not at the position claimed, so
    it's let through to git apply, which (replayed here for real, not
    asserted) applies it cleanly, exactly as it did historically."""
    result = await _run(
        sandbox_manager,
        "pinned-success",
        SUCCESS_DIFF,
        tracking={"calculator.py": (1, 50)},
        numbered={},
    )
    assert result.success is True, result.message
