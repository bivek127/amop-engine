"""Durable capture for the still-unexplained "correct region, wrong
transcription" gap named in Milestone 30's own closing note. Not an
attempt to fix that gap -- these tests only confirm the trail gets
left: a PATCH_CONFLICT that the Milestone 30 fresh-read gate did NOT
reject (i.e. a real CoderAgent turn whose read genuinely covered the
diff's target range, or had no parseable target at all) is written to
AMOP_DIAGNOSTICS_DIR as a JSON snapshot with both sides of the failed
comparison (the diff and the file's real content) -- not just a
2000-char-capped message in `agent_actions`. A STALE_READ rejection
(the gate itself firing, an already-diagnosed and handled case) must
NOT also produce a capture; that would conflate two different things.

Milestone 36 update: two of Milestone 35's checks (_check_content_
fidelity, _find_diff_format_defect) now catch most genuinely-fabricated
or malformed diffs BEFORE they ever reach git apply -- narrowing what
still reaches the capture point below. The fixture diffs here were
updated accordingly: content that's real and present SOMEWHERE in the
file (passing the new lenient, position-independent fidelity check) but
claimed at a position far enough away that git apply's own limited
fuzz-matching still can't find it -- a genuine, still-real residual
PATCH_CONFLICT shape the new pre-flight checks don't (and, per
Milestone 35/36's own diagnosis, correctly shouldn't) prevent.

Requires a real Docker daemon, same as test_milestone30.py.
"""

import json

import pytest

from amop.sandbox import tools as sandbox_tools
from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext, invoke_tool

# 40 distinct, real lines with one unique marker far from where any
# diff below claims it lives -- big enough a gap that git apply's own
# fuzz-matching (confirmed empirically during Milestone 36's diagnosis:
# an 8-line gap with faithful content still applies; a ~35-line gap does
# not) genuinely can't find it, even though the content is real and
# present, just not where claimed.
_REAL_LINES = [f"line{i}" for i in range(1, 41)]
_REAL_LINES[34] = "UNIQUE_MARKER_LINE"  # real line 35 (1-indexed)
_REAL_CONTENT = "\n".join(_REAL_LINES) + "\n"

_FAR_MISPOSITIONED_DIFF = (
    "--- a/app/thing.py\n"
    "+++ b/app/thing.py\n"
    "@@ -1,1 +1,1 @@\n"
    "-UNIQUE_MARKER_LINE\n"
    "+UNIQUE_MARKER_LINE_CHANGED\n"
)


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


async def test_patch_conflict_with_a_covering_read_is_durably_captured(
    tmp_path, sandbox_manager, monkeypatch
):
    diagnostics_dir = tmp_path / "diag"
    monkeypatch.setattr(sandbox_tools, "DIAGNOSTICS_DIR", diagnostics_dir)

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "thing.py").write_text(_REAL_CONTENT)
    sandbox = sandbox_manager.create("t-capture", tmp_path)
    try:
        # A read that fully covers the diff's claimed target range --
        # the M30 gate passes this through, same as it would for a
        # genuinely fresh, correctly-scoped read. Not numbered (no
        # coder_read_numbered entry set), so Milestone 36's content-
        # fidelity check uses its lenient, position-independent mode --
        # UNIQUE_MARKER_LINE really is in the file, just not at line 1,
        # so that check passes it through to git apply, which is where
        # this one genuinely still fails.
        ctx = make_ctx(tmp_path, sandbox, tracking={"app/thing.py": (1, 40)})
        result = await invoke_tool(
            "patch_file",
            {"path": "app/thing.py", "diff": _FAR_MISPOSITIONED_DIFF},
            ctx,
            agent_name="test",
        )
        assert result.success is False
        assert result.error_code == "PATCH_CONFLICT"

        captured = list(diagnostics_dir.glob("patch_conflict_*.json"))
        assert len(captured) == 1
        payload = json.loads(captured[0].read_text())
        assert payload["path"] == "app/thing.py"
        assert payload["current_content"] == _REAL_CONTENT
        assert "UNIQUE_MARKER_LINE" in payload["diff"]
        assert payload["read_range"] == [1, 40]
        assert payload["git_apply_stage"] in ("check", "apply")
        assert payload["conflict_message"]
    finally:
        sandbox_manager.destroy("t-capture")


async def test_stale_read_rejection_does_not_trigger_a_capture(
    tmp_path, sandbox_manager, monkeypatch
):
    """The gate rejecting a stale/insufficient read is a different,
    already-diagnosed case (Milestones 17/30) -- must not also be
    captured as if it were the still-unexplained residual gap."""
    diagnostics_dir = tmp_path / "diag"
    monkeypatch.setattr(sandbox_tools, "DIAGNOSTICS_DIR", diagnostics_dir)

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "aggregate.py").write_text("placeholder\n" * 30)
    sandbox = sandbox_manager.create("t-nocap", tmp_path)
    try:
        # Tracked read (lines 1-5) does not cover the diff's target
        # (line 25) -- the gate rejects before git apply ever runs.
        ctx = make_ctx(tmp_path, sandbox, tracking={"app/aggregate.py": (1, 5)})
        diff = (
            "--- a/app/aggregate.py\n"
            "+++ b/app/aggregate.py\n"
            "@@ -25,1 +25,1 @@\n"
            "-old\n"
            "+new\n"
        )
        result = await invoke_tool(
            "patch_file",
            {"path": "app/aggregate.py", "diff": diff},
            ctx,
            agent_name="test",
        )
        assert result.error_code == "STALE_READ"
        assert not diagnostics_dir.exists() or not list(diagnostics_dir.glob("*.json"))
    finally:
        sandbox_manager.destroy("t-nocap")


async def test_capture_is_skipped_outside_a_real_coder_turn(tmp_path, sandbox_manager, monkeypatch):
    """coder_read_tracking=None (every non-CoderAgent caller, including
    every direct test call site) means the M30 gate is off entirely --
    the capture is scoped the same way, since it only exists to
    diagnose the gate's own residual gap. Uses the same far-mispositioned-
    but-real-content diff as the test above (content passes Milestone
    36's lenient fidelity check and genuinely reaches git apply) so this
    is a real test of the capture's own coder-turn scoping, not just a
    diff that never gets that far."""
    diagnostics_dir = tmp_path / "diag"
    monkeypatch.setattr(sandbox_tools, "DIAGNOSTICS_DIR", diagnostics_dir)

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "thing.py").write_text(_REAL_CONTENT)
    sandbox = sandbox_manager.create("t-notracking", tmp_path)
    try:
        ctx = ToolContext(
            agent_name="test", scratch_dir=tmp_path, mode="suggestor",
            sandbox=sandbox, coder_read_tracking=None,
        )
        result = await invoke_tool(
            "patch_file",
            {"path": "app/thing.py", "diff": _FAR_MISPOSITIONED_DIFF},
            ctx,
            agent_name="test",
        )
        assert result.error_code == "PATCH_CONFLICT"
        assert not diagnostics_dir.exists() or not list(diagnostics_dir.glob("*.json"))
    finally:
        sandbox_manager.destroy("t-notracking")
