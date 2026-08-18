"""Milestone 12 — Close the Range Bypass, Final Real-File Attempt.

Milestone 11's scoped-default guard only checked an *implicit*
whole-file read_file() call (no start_line/end_line) -- live-confirmed,
an explicit start_line=1, end_line=1675 request (nearly the whole
1,676-line file) sailed straight past it and, with with_line_numbers=true
on top, returned something even bigger than the original unguarded read
would have. This closes that: the exact same threshold and "silently
narrow to a small bounded default with a notice, don't error" behavior
now applies to explicit ranges too, not just the argument-free case.
"""

from pathlib import Path

import pytest

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


def _make_numbered_lines_file(n: int) -> str:
    return "".join(f"content of line {i}\n" for i in range(1, n + 1))


async def test_read_file_small_explicit_range_unaffected(tmp_path, sandbox_manager):
    big = _make_numbered_lines_file(2000)
    (tmp_path / "big.py").write_text(big)
    sandbox = sandbox_manager.create("t-range-small", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "read_file",
            {"path": "big.py", "start_line": 500, "end_line": 510},
            ctx,
            agent_name="test",
        )
        assert result.success
        assert not result.output.startswith("[NOTE:")
        assert result.output == "".join(f"content of line {i}\n" for i in range(500, 511))
    finally:
        sandbox_manager.destroy("t-range-small")


async def test_read_file_huge_explicit_range_is_narrowed_not_returned_whole(
    tmp_path, sandbox_manager
):
    # The exact real shape from Milestone 11's live run: an explicit
    # range covering nearly the whole file.
    big = _make_numbered_lines_file(2000)
    (tmp_path / "big.py").write_text(big)
    sandbox = sandbox_manager.create("t-range-huge", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "read_file",
            {"path": "big.py", "start_line": 1, "end_line": 1999},
            ctx,
            agent_name="test",
        )
        assert result.success
        assert result.output.startswith("[NOTE:")
        assert "content of line 1\n" in result.output
        assert "content of line 60\n" in result.output
        # The defining behavior: genuinely narrowed, not just truncated
        # by luck -- content past the cap is provably absent.
        assert "content of line 61\n" not in result.output
        assert "content of line 1999\n" not in result.output
    finally:
        sandbox_manager.destroy("t-range-huge")


async def test_read_file_huge_explicit_range_with_line_numbers_is_narrowed(
    tmp_path, sandbox_manager
):
    # The exact combination that bypassed the old guard in Milestone
    # 11's live run: a huge range + with_line_numbers=true, which adds
    # its own overhead on top of the raw content.
    big = _make_numbered_lines_file(2000)
    (tmp_path / "big.py").write_text(big)
    sandbox = sandbox_manager.create("t-range-huge-numbered", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "read_file",
            {"path": "big.py", "start_line": 1, "end_line": 1675, "with_line_numbers": True},
            ctx,
            agent_name="test",
        )
        assert result.success
        assert result.output.startswith("[NOTE:")
        assert "1: content of line 1\n" in result.output
        assert "60: content of line 60\n" in result.output
        assert "61: content of line 61" not in result.output
        assert "1675: content of line 1675" not in result.output
    finally:
        sandbox_manager.destroy("t-range-huge-numbered")


async def test_read_file_narrowed_range_starts_from_the_requested_start_line(
    tmp_path, sandbox_manager
):
    # Narrowing keeps the requested lo, not line 1 of the file -- a
    # caller who deliberately targeted the file's middle shouldn't get
    # silently redirected to the top. Range chosen (500-2000, ~31KB) to
    # genuinely clear the 24,000-char threshold -- verified directly,
    # not assumed.
    big = _make_numbered_lines_file(2000)
    (tmp_path / "big.py").write_text(big)
    sandbox = sandbox_manager.create("t-range-offset", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "read_file",
            {"path": "big.py", "start_line": 500, "end_line": 2000},
            ctx,
            agent_name="test",
        )
        assert result.success
        assert result.output.startswith("[NOTE:")
        assert "content of line 500\n" in result.output
        assert "content of line 559\n" in result.output
        assert "content of line 560\n" not in result.output
        assert "content of line 2000\n" not in result.output
    finally:
        sandbox_manager.destroy("t-range-offset")
