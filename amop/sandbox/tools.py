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
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from amop.safety.engine import resolve_within_scratch
from amop.sandbox import repo
from amop.tools.registry import ToolContext, ToolResult, tool

SCRATCH_DIR = Path(os.environ.get("AMOP_SCRATCH_DIR", "./amop_workspace")).resolve()
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# Section 6.3.7: sandbox.test_timeout_seconds, default 300s for a full
# suite. The tool's own timeout_seconds sits just above it so the
# registry's wrapper doesn't fire before the sandbox's kill path does.
TEST_TIMEOUT_SECONDS = float(os.environ.get("AMOP_TEST_TIMEOUT_SECONDS", "300"))


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


# ---------------------------------------------------------------------
# Milestone 4 tools: test execution and diff inspection, both read-only
# observations of the repo's state.
# ---------------------------------------------------------------------


def _parse_junit_xml(xml_text: str) -> dict:
    """Normalize pytest's junit-xml into Section 8.6's shape:
    {passed, failed, skipped, failures: [{test_name, message, file, line}]}.

    Parsing the machine-readable report rather than scraping stdout is
    what makes this runner-agnostic per 8.6, and it's why `all_passed`
    can be a fact rather than a model's opinion.
    """
    root = ET.fromstring(xml_text)
    suite = root.find("testsuite") if root.tag == "testsuites" else root

    total = int(suite.get("tests", 0))
    failed = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))

    failures = []
    for case in suite.iter("testcase"):
        for outcome in list(case):
            if outcome.tag not in ("failure", "error"):
                continue
            failures.append(
                {
                    "test_name": case.get("name", ""),
                    "message": (outcome.get("message") or "")[:500],
                    "file": case.get("file"),
                    "line": int(case.get("line")) if case.get("line") else None,
                }
            )
    return {
        "passed": total - failed - skipped,
        "failed": failed,
        "skipped": skipped,
        "failures": failures,
    }


@tool(
    name="run_tests",
    description=(
        "Run the repository's pytest suite inside the sandbox and return "
        "structured results. Optionally pass 'path' to run only the tests "
        "at that path. Does not modify any source file."
    ),
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": [],
    },
    # Non-mutating: executing a test suite observes the code's behavior,
    # it doesn't change the source. This matters beyond bookkeeping --
    # the Investigator runs at observer permissions (Section 6.2), and a
    # mutating tool is denied outright in observer mode, so marking this
    # True would make root-cause investigation impossible.
    mutating=False,
    timeout_seconds=310,
)
async def run_tests(ctx: ToolContext, path: str | None = None) -> ToolResult:
    if ctx.sandbox is None:
        return ToolResult(
            success=False,
            error_code="SANDBOX_UNAVAILABLE",
            message="no sandbox session for this task",
        )

    # `path` is deliberately named to match read_file/write_file's
    # argument rather than something like `scope`: that way the Safety
    # Engine's existing path restriction (safety/engine.py's
    # path_restricted) covers it automatically, instead of this tool
    # introducing a second, unguarded path-shaped argument that the gate
    # doesn't know to check.
    target = "."
    if path:
        resolved = resolve_within_scratch(path, ctx.scratch_dir)
        if resolved is None:
            return ToolResult(
                success=False,
                error_code="PATH_NOT_PERMITTED",
                message=f"{path!r} is outside the scratch directory",
            )
        target = _container_path(resolved, ctx.scratch_dir)

    report_path = f"/tmp/amop-junit-{uuid.uuid4().hex}.xml"
    command = (
        f"cd /workspace && python -m pytest {target} "
        f"--junit-xml={report_path} -q"
    )
    try:
        result = await asyncio.to_thread(
            ctx.sandbox.exec_run, command, TEST_TIMEOUT_SECONDS
        )
    except Exception as exc:
        return ToolResult(success=False, error_code="RUN_ERROR", message=str(exc))

    if result.timed_out:
        return ToolResult(
            success=False,
            error_code="TIMEOUT",
            message=f"test run exceeded {TEST_TIMEOUT_SECONDS}s",
        )

    try:
        xml_text = await asyncio.to_thread(ctx.sandbox.read_file, report_path)
        summary = _parse_junit_xml(xml_text)
    except (FileNotFoundError, ET.ParseError):
        # pytest couldn't even produce a report (collection error, no
        # tests found, ...). Surface the raw output as evidence instead
        # of pretending we got a clean zero-failure result.
        return ToolResult(
            success=False,
            error_code="NO_TEST_REPORT",
            message=(result.stdout + result.stderr)[-2000:] or "pytest produced no report",
        )

    summary["all_passed"] = summary["failed"] == 0 and summary["passed"] > 0
    summary["output_tail"] = (result.stdout + result.stderr)[-2000:]
    return ToolResult(success=True, output=summary)


@tool(
    name="get_diff",
    description=(
        "Show the diff of the current working branch against the baseline "
        "branch, including uncommitted changes. Read-only."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    mutating=False,
    timeout_seconds=60,
)
async def get_diff(ctx: ToolContext) -> ToolResult:
    if ctx.sandbox is None:
        return ToolResult(
            success=False,
            error_code="SANDBOX_UNAVAILABLE",
            message="no sandbox session for this task",
        )
    try:
        diff = await asyncio.to_thread(repo.diff_against_baseline, ctx.sandbox)
    except Exception as exc:
        return ToolResult(success=False, error_code="DIFF_ERROR", message=str(exc))
    return ToolResult(success=True, output=diff)
