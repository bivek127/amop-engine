"""read_file/write_file, moved off the host filesystem and into the
task's sandbox container (Section 9) -- Milestone 3's replacement for
tools/filesystem.py's Milestone 2 host-filesystem versions.

Same tool contract (name, schema, mutating flag) as Milestone 2's
versions, so the Safety Engine (safety/engine.py, unmodified this
milestone) still gates every call exactly as before -- path-restriction
denial happens BEFORE a call ever reaches this module. The
resolve_within_scratch() re-check below is this module's own defense in
depth on top of that gate, same rationale as Milestone 2's: a tool body
should never blindly trust that the pipeline upstream did its job. The
sandbox's container-mount boundary (Section 9.3) is a *third*,
independent layer on top of both -- even a call that somehow reached this
function's body with a bad path has nothing to reach, because the
container's filesystem outside /workspace was never the host's to begin
with.
"""

import asyncio
import os
from pathlib import Path

from amop.safety.engine import resolve_within_scratch
from amop.tools.registry import ToolContext, ToolResult, tool

SCRATCH_DIR = Path(os.environ.get("AMOP_SCRATCH_DIR", "./amop_workspace")).resolve()
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)


def _container_path(target: Path, scratch_dir: Path) -> str:
    relative = target.relative_to(scratch_dir.resolve())
    return "/workspace" if str(relative) == "." else f"/workspace/{relative}"


@tool(
    name="read_file",
    description="Read the contents of a file inside the scratch workspace.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    mutating=False,
    timeout_seconds=5,
)
async def read_file(path: str, ctx: ToolContext) -> ToolResult:
    target = resolve_within_scratch(path, ctx.scratch_dir)
    if target is None:
        return ToolResult(
            success=False,
            error_code="PATH_NOT_PERMITTED",
            message=f"{path!r} is outside the scratch directory",
        )
    if ctx.sandbox is None:
        return ToolResult(
            success=False,
            error_code="SANDBOX_UNAVAILABLE",
            message="no sandbox session for this task",
        )
    container_path = _container_path(target, ctx.scratch_dir)
    try:
        content = await asyncio.to_thread(ctx.sandbox.read_file, container_path)
    except FileNotFoundError:
        return ToolResult(
            success=False, error_code="NOT_FOUND", message=f"No such file: {path}"
        )
    except Exception as exc:
        return ToolResult(success=False, error_code="READ_ERROR", message=str(exc))
    return ToolResult(success=True, output=content)


@tool(
    name="write_file",
    description=(
        "Write content to a file inside the scratch workspace, creating "
        "parent directories as needed. Overwrites the whole file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
    mutating=True,
    timeout_seconds=5,
)
async def write_file(path: str, content: str, ctx: ToolContext) -> ToolResult:
    target = resolve_within_scratch(path, ctx.scratch_dir)
    if target is None:
        return ToolResult(
            success=False,
            error_code="PATH_NOT_PERMITTED",
            message=f"{path!r} is outside the scratch directory",
        )
    if ctx.sandbox is None:
        return ToolResult(
            success=False,
            error_code="SANDBOX_UNAVAILABLE",
            message="no sandbox session for this task",
        )
    container_path = _container_path(target, ctx.scratch_dir)
    try:
        await asyncio.to_thread(ctx.sandbox.write_file, container_path, content)
    except Exception as exc:
        return ToolResult(success=False, error_code="WRITE_ERROR", message=str(exc))
    return ToolResult(success=True, output=f"wrote {len(content)} bytes to {path}")
