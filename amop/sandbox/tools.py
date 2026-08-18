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


# Milestone 10: a whole-file read_file on a large real file (e.g.
# pyinvoke/invoke's runners.py, 65,509 chars) silently overflows a
# model's context window -- the model only ever "sees" the head of the
# tool result and reasons about whatever's there, with no indication
# anything was cut off (the same silent-truncation mechanism diagnosed
# for chat completions generally, just showing up in a tool result
# instead). Reusing indexer.py's MAX_CHUNK_CHARS_FOR_EMBEDDING threshold
# here too -- not because it's an embedding budget, but because it's
# already the project's one calibrated "this is too big for a model to
# take in whole" number.
_LARGE_FILE_NOTICE_THRESHOLD = 8_000

# Milestone 11: the notice above doesn't work above some size -- live-
# confirmed against the real 65,509-char runners.py that the model never
# acts on it; it just answers off whatever fits before Ollama's own
# context truncation and stops there. Above this bar, don't even attempt
# a whole-file return: the content would be silently cut down to
# something unpredictable anyway, so cut it down to something small and
# PREDICTABLE instead -- a fixed-size head slice the model can actually
# reason about completely, with an explicit, unmissed-because-it's-the-
# only-content instruction on how to see more. 24,000 chars sits in the
# middle of a "reasonable, adjust if needed" 20-30KB band -- comfortably
# above the notice threshold (files that merely trip the notice may
# still mostly fit under some configs) and comfortably below the real
# file that motivated this (65,509 chars), so it actually exercises the
# new path rather than sitting right at the edge.
_SCOPED_READ_THRESHOLD = 24_000
_SCOPED_READ_DEFAULT_LINES = 60


def _large_file_notice(content: str, total_lines: int) -> str:
    # Prepended, not appended: the same truncation this warns about
    # drops the END of an oversized message, not the start (confirmed
    # directly -- a 21,000-char single-message repro left Ollama's own
    # prompt_eval_count covering only the first ~2,000 tokens). A notice
    # placed after the content would just get cut off with it.
    return (
        f"[NOTE: this file is {len(content)} chars / {total_lines} lines -- "
        "likely too large to fully fit in your context window at once. "
        "If you don't find what you're looking for in what follows, "
        "re-call read_file with start_line/end_line to read a specific "
        "slice instead of assuming it isn't in the file.]\n\n"
    )


def _scoped_default_notice(size_bytes: int, total_lines: int, shown_lines: int) -> str:
    return (
        f"[NOTE: this file is {size_bytes} bytes / {total_lines} lines -- "
        f"too large to read in full, so only the first {shown_lines} lines "
        "are shown below (line numbers included). This is not the whole "
        "file. To see a different part: call read_file again with "
        "start_line/end_line (add with_line_numbers=true if you're about "
        "to construct a patch_file diff and need to know exact line "
        "numbers) -- or use search_code first to find which lines are "
        "actually relevant before reading them.]\n\n"
    )


@tool(
    name="read_file",
    description=(
        "Read the contents of a file inside the scratch workspace. "
        "Optional start_line/end_line (1-indexed, inclusive) read just "
        "that slice instead of the whole file -- use this once you have "
        "a rough idea where the relevant code is (e.g. from search_code's "
        "line ranges), and keep the span reasonably small (tens of lines, "
        "not hundreds) -- an oversized range gets silently narrowed to a "
        "small default too, same as reading with no range at all. If you "
        "call this WITHOUT start_line/end_line on a file that turns out "
        "to be large, you will automatically get back only its first ~60 "
        "lines (not the whole thing, and not an error) -- that's "
        "expected, re-call with a small start_line/end_line span (using "
        "search_code first, if you can, to know which lines you actually "
        "need) to see the rest. Optional with_line_numbers=true prefixes "
        "each returned line with 'N: ' -- useful when you're about to "
        "construct a patch_file diff and need to know the exact starting "
        "line for its '@@ -start,count +start,count @@' header. The "
        "'N: ' prefix is for your reference only -- never include it in "
        "an actual diff body, that has to match the file's real content "
        "exactly."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
            "with_line_numbers": {"type": "boolean"},
        },
        "required": ["path"],
    },
    mutating=False,
    timeout_seconds=5,
)
async def read_file(
    path: str,
    ctx: ToolContext,
    start_line: int | None = None,
    end_line: int | None = None,
    with_line_numbers: bool = False,
) -> ToolResult:
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

    # Milestone 11: a size check BEFORE deciding to fetch/return the
    # whole file, only when a whole-file read was actually requested --
    # a targeted start_line/end_line call already knows what it wants
    # and shouldn't be second-guessed here.
    if start_line is None and end_line is None:
        try:
            size = await asyncio.to_thread(ctx.sandbox.stat_size, container_path)
        except FileNotFoundError:
            return ToolResult(
                success=False, error_code="NOT_FOUND", message=f"No such file: {path}"
            )
        except Exception as exc:
            return ToolResult(success=False, error_code="READ_ERROR", message=str(exc))

        if size > _SCOPED_READ_THRESHOLD:
            try:
                content = await asyncio.to_thread(ctx.sandbox.read_file, container_path)
            except FileNotFoundError:
                return ToolResult(
                    success=False, error_code="NOT_FOUND", message=f"No such file: {path}"
                )
            except Exception as exc:
                return ToolResult(success=False, error_code="READ_ERROR", message=str(exc))

            lines = content.splitlines(keepends=True)
            total_lines = len(lines)
            shown = lines[:_SCOPED_READ_DEFAULT_LINES]
            numbered = "".join(f"{i + 1}: {line}" for i, line in enumerate(shown))
            notice = _scoped_default_notice(size, total_lines, len(shown))
            return ToolResult(success=True, output=notice + numbered)

    try:
        content = await asyncio.to_thread(ctx.sandbox.read_file, container_path)
    except FileNotFoundError:
        return ToolResult(
            success=False, error_code="NOT_FOUND", message=f"No such file: {path}"
        )
    except Exception as exc:
        return ToolResult(success=False, error_code="READ_ERROR", message=str(exc))

    if start_line is not None or end_line is not None:
        lines = content.splitlines(keepends=True)
        total = len(lines)
        lo = max(1, start_line or 1)
        hi = min(total, end_line if end_line is not None else total)
        if lo > total or lo > hi:
            return ToolResult(
                success=False,
                error_code="INVALID_ARGS",
                message=f"start_line/end_line out of range for a {total}-line file",
            )
        selected = lines[lo - 1 : hi]

        # Milestone 12: an explicit range can be just as oversized as an
        # implicit whole-file read -- live-confirmed, a real
        # start_line=1, end_line=1675 request (nearly the whole file)
        # sailed straight past the whole-file-only guard below and,
        # with with_line_numbers on top, returned something even bigger
        # than the original unguarded read. Same threshold, same
        # "silently narrow to a small bounded default with a notice,
        # don't error" behavior as that guard -- consistency, not a
        # second, differently-shaped rule.
        approx_size = sum(len(line) for line in selected)
        if with_line_numbers:
            approx_size += sum(len(f"{lo + i}: ") for i in range(len(selected)))

        if approx_size > _SCOPED_READ_THRESHOLD:
            capped = selected[:_SCOPED_READ_DEFAULT_LINES]
            capped_hi = lo + len(capped) - 1
            notice = (
                f"[NOTE: the requested range (lines {lo}-{hi}, ~{approx_size} "
                f"chars) is too large to return in full -- narrowed to lines "
                f"{lo}-{capped_hi} ({len(capped)} lines) instead. Request a "
                "smaller start_line/end_line span to see a different part.]\n\n"
            )
            if with_line_numbers:
                body = "".join(f"{lo + i}: {line}" for i, line in enumerate(capped))
            else:
                body = "".join(capped)
            return ToolResult(success=True, output=notice + body)

        if with_line_numbers:
            return ToolResult(
                success=True,
                output="".join(f"{lo + i}: {line}" for i, line in enumerate(selected)),
            )
        return ToolResult(success=True, output="".join(selected))

    total_lines = content.count("\n") + 1
    if with_line_numbers:
        lines = content.splitlines(keepends=True)
        content = "".join(f"{i + 1}: {line}" for i, line in enumerate(lines))

    if len(content) > _LARGE_FILE_NOTICE_THRESHOLD:
        content = _large_file_notice(content, total_lines) + content

    return ToolResult(success=True, output=content)


# Milestone 13: found live in Milestone 12 -- Coder abandoned a
# correctly-rejected patch_file attempt and fell back to write_file with
# only a small fragment as the ENTIRE new content of a large, real,
# existing file. The tool call reported success; the file's actual
# content was destroyed (1,675 real lines replaced by 18 broken ones).
# patch_file already refuses a change that doesn't match reality
# (PATCH_CONFLICT); write_file had no equivalent. This is that
# equivalent -- a heuristic, not a perfect check (a legitimately huge
# deletion could still trip it, and that's an acceptable false positive
# given what it catches). 20% is chosen with real margin above the
# actual observed failure (18/1675 lines is ~1% of the original, nowhere
# close to this threshold) while still being restrictive enough to
# matter -- tuned against that real case, not derived from a formula.
_SUSPICIOUS_SHRINK_RATIO = 0.20


@tool(
    name="write_file",
    description=(
        "Write content to a file inside the scratch workspace, creating "
        "parent directories as needed. Overwrites the whole file. If the "
        "target file already exists and the new content is dramatically "
        "smaller than what's there now, this is refused with "
        "error_code=SUSPICIOUS_SHRINK instead of silently overwriting -- "
        "that's a signal you don't have the full picture of this file; "
        "re-read it and use patch_file for a targeted change instead of "
        "retrying write_file. New file creation is unaffected."
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
        current_size = await asyncio.to_thread(ctx.sandbox.stat_size, container_path)
    except FileNotFoundError:
        current_size = None  # new file -- nothing to protect, guard doesn't apply
    except Exception as exc:
        return ToolResult(success=False, error_code="WRITE_ERROR", message=str(exc))

    if current_size is not None and current_size > 0:
        new_size = len(content.encode("utf-8"))
        if new_size < current_size * _SUSPICIOUS_SHRINK_RATIO:
            return ToolResult(
                success=False,
                error_code="SUSPICIOUS_SHRINK",
                message=(
                    f"refusing to overwrite {path!r}: new content is {new_size} "
                    f"bytes, existing file is {current_size} bytes "
                    f"({new_size / current_size:.1%} of the original) -- this "
                    "looks like a fragment overwriting a much larger real "
                    "file, not a deliberate rewrite. Re-read the file to "
                    "confirm you have the full picture, and use patch_file "
                    "for a targeted change instead of retrying write_file."
                ),
            )

    try:
        await asyncio.to_thread(ctx.sandbox.write_file, container_path, content)
    except Exception as exc:
        return ToolResult(success=False, error_code="WRITE_ERROR", message=str(exc))
    return ToolResult(success=True, output=f"wrote {len(content)} bytes to {path}")


def _normalize_diff_headers(diff: str, relative_path: str) -> str:
    """Rewrite (or synthesize) the `--- `/`+++ ` header lines so the diff
    unambiguously targets `relative_path`, regardless of what the model
    wrote there -- paired with `git apply -p1` in patch_file(). This only
    ever touches those two header lines, never the hunk bodies: the
    actual patch application, and therefore all real conflict detection,
    is still 100% git apply's job, not reimplemented here. Removes a
    whole class of spurious failures (a local model getting a path
    prefix wrong, or omitting headers entirely) that have nothing to do
    with whether the diff's *content* is stale.
    """
    old_header = f"--- a/{relative_path}\n"
    new_header = f"+++ b/{relative_path}\n"
    lines = diff.splitlines(keepends=True)

    has_old = any(line.startswith("--- ") for line in lines)
    has_new = any(line.startswith("+++ ") for line in lines)

    if not (has_old and has_new):
        # No headers at all (or only one, a rare malformed case) --
        # synthesize both fresh, in front of whatever hunk content is
        # there. A genuinely malformed remainder still fails at git
        # apply, cleanly, as PATCH_CONFLICT.
        return old_header + new_header + diff

    out = []
    replaced_old = replaced_new = False
    for line in lines:
        if not replaced_old and line.startswith("--- "):
            out.append(old_header)
            replaced_old = True
        elif not replaced_new and line.startswith("+++ "):
            out.append(new_header)
            replaced_new = True
        else:
            out.append(line)
    return "".join(out)


@tool(
    name="patch_file",
    description=(
        "Apply a unified diff to an EXISTING file inside the scratch "
        "workspace -- preferred over write_file for edits to existing "
        "files: smaller blast radius, and you don't need the whole file "
        "in context to construct it, just the lines you're changing plus "
        "a couple of lines of surrounding context on each side (so the "
        "patch has something to anchor to). Standard unified diff format: "
        "'--- '/'+++ ' header lines (the path in them is ignored -- only "
        "the 'path' argument matters), then one or more hunks starting "
        "'@@ -start,count +start,count @@', with context lines prefixed "
        "by a space, removed lines by '-', added lines by '+'. Rejected "
        "with error_code=PATCH_CONFLICT if the diff doesn't apply cleanly "
        "(e.g. you read stale content, or got the location wrong) -- "
        "re-read the current file/region first, then retry with a fresh "
        "diff; never resubmit the same one unchanged. Use write_file "
        "instead for a brand new file or a genuinely trivial single-line "
        "change."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "diff": {"type": "string"},
        },
        "required": ["path", "diff"],
    },
    mutating=True,
    timeout_seconds=10,
)
async def patch_file(path: str, diff: str, ctx: ToolContext) -> ToolResult:
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
    relative_path = container_path.removeprefix("/workspace/")

    try:
        await asyncio.to_thread(ctx.sandbox.read_file, container_path)
    except FileNotFoundError:
        return ToolResult(
            success=False,
            error_code="NOT_FOUND",
            message=f"No such file: {path} -- use write_file to create a new file",
        )
    except Exception as exc:
        return ToolResult(success=False, error_code="READ_ERROR", message=str(exc))

    normalized = _normalize_diff_headers(diff, relative_path)
    patch_path = f"/tmp/amop-patch-{uuid.uuid4().hex}.diff"
    try:
        await asyncio.to_thread(ctx.sandbox.write_file, patch_path, normalized)

        check = await asyncio.to_thread(
            ctx.sandbox.exec_run, f"git apply --check --recount {patch_path}"
        )
        if check.exit_code != 0:
            return ToolResult(
                success=False,
                error_code="PATCH_CONFLICT",
                message=(check.stderr or check.stdout or "diff did not apply cleanly").strip(),
            )

        applied = await asyncio.to_thread(
            ctx.sandbox.exec_run, f"git apply --recount {patch_path}"
        )
        if applied.exit_code != 0:
            return ToolResult(
                success=False,
                error_code="PATCH_CONFLICT",
                message=(
                    applied.stderr or applied.stdout or "diff did not apply cleanly"
                ).strip(),
            )
    except Exception as exc:
        return ToolResult(success=False, error_code="PATCH_ERROR", message=str(exc))
    finally:
        await asyncio.to_thread(ctx.sandbox.exec_run, f"rm -f {patch_path}")

    return ToolResult(success=True, output=f"applied patch to {path}")


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
