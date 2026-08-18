"""Milestone 13 — write_file Safety Guard.

Milestone 12 found, live, that write_file can report success while
silently destroying an existing file's content: Coder abandoned a
correctly-rejected patch_file attempt and fell back to write_file with
only an 18-line fragment as the entire new content of a real,
1,675-line file. The tool call succeeded; the file's actual content was
destroyed. patch_file already refuses a change that doesn't match
reality (PATCH_CONFLICT, a dry-run check before applying); write_file
had no equivalent -- this is that equivalent.

Heuristic, not a perfect check (a legitimately huge deletion could
still trip it) -- 20% chosen with real margin above the actual observed
failure (18/1675 lines is ~1% of the original), while still being
restrictive enough to matter.
"""

from pathlib import Path

import pytest

from amop.agents.coder import CoderAgent
from amop.models.base import BaseLLM
from amop.sandbox import tools as sandbox_tools  # noqa: F401 -- registers read_file/write_file/patch_file
from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext, invoke_tool


@pytest.fixture
def sandbox_manager():
    manager = SandboxManager()
    yield manager
    manager.destroy_all()


def make_ctx(scratch_dir, sandbox, mode="suggestor"):
    return ToolContext(agent_name="test", scratch_dir=scratch_dir, mode=mode, sandbox=sandbox)


# ---------------------------------------------------------------------
# Tier (a) — the guard itself, real Docker sandbox (Milestone 3's "no
# mock sandbox path" discipline)
# ---------------------------------------------------------------------


async def test_write_file_the_exact_milestone_12_failure_shape_is_now_caught(
    tmp_path, sandbox_manager
):
    """The direct regression test for the finding that motivated this
    milestone: a large real file, a tiny fragment overwrite attempt at
    roughly the same ~1% ratio Coder actually produced live against
    invoke/runners.py (18 lines replacing 1,675)."""
    real_file = "".join(f"real line {i} of the original file\n" for i in range(1, 1676))
    (tmp_path / "runners.py").write_text(real_file)
    fragment = "".join(f"broken fragment line {i}\n" for i in range(1, 19))  # ~18 lines
    assert len(fragment) < 0.02 * len(real_file)  # genuinely ~1%, not assumed

    sandbox = sandbox_manager.create("t-guard-m12-shape", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "write_file", {"path": "runners.py", "content": fragment}, ctx, agent_name="test"
        )
        assert not result.success
        assert result.error_code == "SUSPICIOUS_SHRINK"
        # File left exactly as it was -- nothing written, no corruption.
        assert (tmp_path / "runners.py").read_text() == real_file
    finally:
        sandbox_manager.destroy("t-guard-m12-shape")


async def test_write_file_new_file_creation_is_unaffected(tmp_path, sandbox_manager):
    sandbox = sandbox_manager.create("t-guard-newfile", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "write_file", {"path": "brand_new.py", "content": "x = 1\n"}, ctx, agent_name="test"
        )
        assert result.success, result.message
        assert (tmp_path / "brand_new.py").read_text() == "x = 1\n"
    finally:
        sandbox_manager.destroy("t-guard-newfile")


async def test_write_file_legitimate_rewrite_above_threshold_still_works(
    tmp_path, sandbox_manager
):
    original = "a" * 1000
    (tmp_path / "target.py").write_text(original)
    # 30% of original -- above the 20% cutoff, should succeed normally.
    replacement = "b" * 300
    sandbox = sandbox_manager.create("t-guard-legit", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "write_file", {"path": "target.py", "content": replacement}, ctx, agent_name="test"
        )
        assert result.success, result.message
        assert (tmp_path / "target.py").read_text() == replacement
    finally:
        sandbox_manager.destroy("t-guard-legit")


async def test_write_file_empty_existing_file_never_flagged(tmp_path, sandbox_manager):
    (tmp_path / "empty.py").write_text("")
    sandbox = sandbox_manager.create("t-guard-empty", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "write_file", {"path": "empty.py", "content": "x"}, ctx, agent_name="test"
        )
        assert result.success, result.message
        assert (tmp_path / "empty.py").read_text() == "x"
    finally:
        sandbox_manager.destroy("t-guard-empty")


async def test_write_file_exactly_at_threshold_boundary_is_allowed(tmp_path, sandbox_manager):
    # Exactly 20% -- the check is strict less-than, so this is the
    # smallest content that is NOT rejected. Explicit, not implicit.
    original = "a" * 1000
    (tmp_path / "target.py").write_text(original)
    replacement = "b" * 200  # exactly 20% of 1000
    sandbox = sandbox_manager.create("t-guard-boundary-allowed", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "write_file", {"path": "target.py", "content": replacement}, ctx, agent_name="test"
        )
        assert result.success, result.message
    finally:
        sandbox_manager.destroy("t-guard-boundary-allowed")


async def test_write_file_just_under_threshold_boundary_is_rejected(tmp_path, sandbox_manager):
    original = "a" * 1000
    (tmp_path / "target.py").write_text(original)
    replacement = "b" * 199  # just under 20% of 1000
    sandbox = sandbox_manager.create("t-guard-boundary-rejected", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "write_file", {"path": "target.py", "content": replacement}, ctx, agent_name="test"
        )
        assert not result.success
        assert result.error_code == "SUSPICIOUS_SHRINK"
        assert (tmp_path / "target.py").read_text() == original
    finally:
        sandbox_manager.destroy("t-guard-boundary-rejected")


# ---------------------------------------------------------------------
# Tier (b) — CoderAgent wiring, pure (no I/O)
# ---------------------------------------------------------------------


def test_coder_system_prompt_explains_suspicious_shrink():
    class _Stub(BaseLLM):
        async def complete(self, messages, tools=None):
            raise NotImplementedError

        async def embed(self, texts):
            raise NotImplementedError

    ctx = ToolContext(agent_name="coder", scratch_dir=Path("."), mode="suggestor")
    coder = CoderAgent(_Stub(), ctx, task_id="t1")

    prompt = coder.system_prompt()

    assert "SUSPICIOUS_SHRINK" in prompt
    assert "patch_file for a targeted change instead of retrying write_file" in prompt
