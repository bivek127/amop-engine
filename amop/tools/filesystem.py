"""First real tools — spec Section 8.3, trimmed to read_file/write_file
for this milestone (patch_file/list_directory/search_* are later).

No sandbox yet: these operate directly on the host filesystem, hard-
restricted to a scratch working directory (default ./amop_workspace/,
override via AMOP_SCRATCH_DIR) so nothing can touch the real project or
system files. The Safety Engine (safety/engine.py) already denies any
out-of-scope call before the tool body below ever runs; the
resolve_within_scratch() re-check here is defense in depth, not the
primary boundary -- a tool body should never blindly trust that the
pipeline upstream did its job.
"""

import os
from pathlib import Path

from amop.safety.engine import resolve_within_scratch
from amop.tools.registry import ToolContext, ToolResult, tool

SCRATCH_DIR = Path(os.environ.get("AMOP_SCRATCH_DIR", "./amop_workspace")).resolve()
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)


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
    try:
        content = target.read_text()
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
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    except Exception as exc:
        return ToolResult(success=False, error_code="WRITE_ERROR", message=str(exc))
    return ToolResult(success=True, output=f"wrote {len(content)} bytes to {path}")
