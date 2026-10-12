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
import difflib
import json
import os
import re
import shlex
import uuid
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from amop.safety.engine import resolve_within_scratch
from amop.sandbox import repo
from amop.sandbox.repo import GitError
from amop.tools.registry import ToolContext, ToolResult, tool

SCRATCH_DIR = Path(os.environ.get("AMOP_SCRATCH_DIR", "./amop_workspace")).resolve()
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# Durable capture for the still-unexplained "gate passed, diff still
# didn't apply" residual case named in Milestone 30's own closing note
# ("leaving a separate diff-construction failure mode isolated rather
# than hidden behind it") -- a distinct directory from SCRATCH_DIR
# because scratch dirs are per-task and get cleaned up (Milestone 25/31);
# a diagnostic capture must outlive the task it came from. Never
# committed -- see .gitignore.
DIAGNOSTICS_DIR = Path(os.environ.get("AMOP_DIAGNOSTICS_DIR", "./amop_diagnostics")).resolve()
_DIAGNOSTICS_MAX_CONTENT_CHARS = 100_000

# Section 6.3.7: sandbox.test_timeout_seconds, default 300s for a full
# suite. The tool's own timeout_seconds sits just above it so the
# registry's wrapper doesn't fire before the sandbox's kill path does.
TEST_TIMEOUT_SECONDS = float(os.environ.get("AMOP_TEST_TIMEOUT_SECONDS", "300"))

# Stacks whose tests run under Jest rather than pytest (Section 8.6's
# runner-agnostic contract). This is a SET, not an equality check
# against "javascript", specifically because Milestone 26 added
# "typescript" as its own detected stack: an `== "javascript"` test
# would have silently fallen through and run *pytest* against a
# TypeScript repo -- failing in a confusing, far-downstream way rather
# than at the dispatch. Any future stack that runs Jest joins this set;
# nothing else needs to change.
JEST_STACKS = frozenset({"javascript", "typescript"})


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


def _track_read(
    ctx: ToolContext,
    path: str,
    start_line: int | None,
    end_line: int | None,
    numbered: bool,
    result: "ToolResult",
) -> None:
    """Milestone 30: record what a successful read_file call actually
    showed, for patch_file's forced-fresh-read gate. A no-op whenever
    ctx.coder_read_tracking is None -- every call site outside a real
    CoderAgent turn (see ToolContext's own docstring). Defined here,
    ahead of read_file_effective_range's own definition further down in
    this module, which is fine -- Python resolves the name at call
    time, well after the whole module has finished loading.

    Milestone 36: also records whether the OUTPUT actually shown was
    numbered ('N: ' prefixed) -- not simply the `with_line_numbers`
    argument, since the oversized-whole-file truncation branch forces
    numbering on regardless of what was asked (see its own call site
    below). This is what patch_file's gate checks in addition to
    coverage: a real historical case (Milestone 35's diagnostic
    captures ee31f113/d3799905) shows a covering-but-UNNUMBERED
    whole-file read still let a badly-miscounted patch_file attempt
    through -- coverage alone was never enough for the specific
    unscoped-whole-file shape, only for a deliberately-scoped read."""
    if ctx.coder_read_tracking is None:
        return
    rng = read_file_effective_range(start_line, end_line, result)
    if rng is not None:
        ctx.coder_read_tracking[path] = rng
        if ctx.coder_read_numbered is not None:
            ctx.coder_read_numbered[path] = numbered


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
            numbered_body = "".join(f"{i + 1}: {line}" for i, line in enumerate(shown))
            notice = _scoped_default_notice(size, total_lines, len(shown))
            result = ToolResult(success=True, output=notice + numbered_body)
            # Always numbered here regardless of the with_line_numbers
            # argument -- this branch forces line numbers on so a
            # follow-up scoped read can target an exact line.
            _track_read(ctx, path, start_line, end_line, True, result)
            return result

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
            result = ToolResult(success=True, output=notice + body)
            _track_read(ctx, path, start_line, end_line, with_line_numbers, result)
            return result

        if with_line_numbers:
            result = ToolResult(
                success=True,
                output="".join(f"{lo + i}: {line}" for i, line in enumerate(selected)),
            )
        else:
            result = ToolResult(success=True, output="".join(selected))
        _track_read(ctx, path, start_line, end_line, with_line_numbers, result)
        return result

    total_lines = content.count("\n") + 1
    if with_line_numbers:
        lines = content.splitlines(keepends=True)
        content = "".join(f"{i + 1}: {line}" for i, line in enumerate(lines))

    if len(content) > _LARGE_FILE_NOTICE_THRESHOLD:
        content = _large_file_notice(content, total_lines) + content

    result = ToolResult(success=True, output=content)
    _track_read(ctx, path, start_line, end_line, with_line_numbers, result)
    return result


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


async def _micro_commit_after_edit(
    ctx: ToolContext, tool_name: str, path: str, success_output: str
) -> str:
    """Spec 4.6.1 / D-15: called right after write_file/patch_file's own
    edit has already succeeded, never before -- this can only make a
    real edit MORE durable, never gate whether the edit happened.

    Skips quietly (returns success_output unchanged) when there's no
    task_id to attribute the commit to -- the same "plenty of call
    sites legitimately have no X" convention audit/actions.py uses for
    a missing db_session, not an error.

    On a commit failure (e.g. a stale index lock), spec 4.6.1 is
    explicit: "the tool result reports success-with-warning rather
    than failing the edit -- the edit did happen; losing durability is
    worse than a failed tool call but not worth discarding correct
    work over." The warning is appended to the tool's own success
    output rather than a new ToolResult field, since output is already
    the human/model-visible outcome and every other tool's contract
    stays unchanged.
    """
    if ctx.task_id is None or ctx.sandbox is None:
        return success_output
    try:
        await asyncio.to_thread(
            repo.micro_commit, ctx.sandbox, str(ctx.task_id), tool_name, path
        )
    except GitError as exc:
        return f"{success_output} (warning: micro-commit failed: {exc})"
    return success_output


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
    output = await _micro_commit_after_edit(
        ctx, "write_file", path, f"wrote {len(content)} bytes to {path}"
    )
    # Milestone 30: content just changed -- any tracked read of this
    # path is now stale, same reasoning as patch_file's own invalidation
    # below. A no-op when tracking is off.
    if ctx.coder_read_tracking is not None:
        ctx.coder_read_tracking.pop(path, None)
    return ToolResult(success=True, output=output)


def _normalize_diff_headers(diff: str, relative_path: str) -> str:
    """Rewrite (or synthesize) the `--- `/`+++ ` header lines so the diff
    unambiguously targets `relative_path`, regardless of what the model
    wrote there -- paired with `git apply -p1` in patch_file(). This only
    ever touches those two header lines and trailing whitespace, never
    the hunk bodies: the actual patch application, and therefore all
    real conflict detection, is still 100% git apply's job, not
    reimplemented here. Removes a whole class of spurious failures (a
    local model getting a path prefix wrong, or omitting headers
    entirely) that have nothing to do with whether the diff's *content*
    is stale.

    Milestone 17: also ensures a trailing newline. Found live, then
    confirmed by direct `git apply --check` reproduction against the
    real historical diffs: qwen2.5-coder:14b's raw output consistently
    omits the newline after the diff's last line, which `git apply`
    rejects outright as "corrupt patch at line N" -- a pure format
    defect, unrelated to whether the patch's content is otherwise
    correct. Confirmed harmless to add for a diff that already ends in
    one (splitlines/join round-trips it unchanged), and confirmed
    sufficient on its own for a diff with no other defect (one of three
    recovered failures applied cleanly once only this was fixed).
    """
    if diff and not diff.endswith("\n"):
        diff += "\n"
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


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


def _find_context_mismatch(
    diff: str, current_content: str
) -> tuple[int, str, str, str] | None:
    """The shared walk behind both _diagnose_context_mismatch (explains
    an actual git apply rejection, after the fact) and
    _check_content_fidelity (Milestone 36's pre-flight gate, before git
    apply ever runs) -- same line-accounting git apply itself does,
    factored out once so the two callers can never silently drift
    apart. Returns (line_no, marker, expected, actual) for the first
    context (' ') or removed ('-') line whose claimed content doesn't
    match the file's real content at that line number, or None if every
    context/removed line matches.
    """
    real_lines = current_content.splitlines()
    lines = diff.splitlines()
    i = 0
    while i < len(lines):
        m = _HUNK_HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        old_line_no = int(m.group(1))
        i += 1
        while i < len(lines) and not lines[i].startswith("@@ "):
            line = lines[i]
            i += 1
            if not line or line[0] not in " -":
                continue  # '+' lines and stray header/blank lines don't consume an old-file line
            expected = line[1:]
            actual = real_lines[old_line_no - 1] if 0 <= old_line_no - 1 < len(real_lines) else None
            if actual is not None and actual != expected:
                return old_line_no, line[0], expected, actual
            old_line_no += 1
    return None


def _diagnose_context_mismatch(diff: str, current_content: str) -> str | None:
    """After git apply has ALREADY rejected a diff, look for the specific
    reason and describe it precisely -- never used to decide accept/
    reject, only to explain a rejection git apply already made (that
    decision stays 100% git apply's, same boundary
    `_normalize_diff_headers` documents). Milestone 36: with
    _check_content_fidelity now catching this same mismatch shape
    BEFORE git apply runs, this function should rarely fire in
    practice -- kept as a backstop for whatever a pre-flight check
    doesn't cover (e.g. a real conflict git apply detects for a reason
    outside this walk's scope), so a genuine, un-pre-empted git apply
    rejection still gets a precise explanation rather than git's own
    opaque message.

    Milestone 17: found live, then confirmed by direct reproduction --
    two of three recovered PATCH_CONFLICT diffs listed a line as
    unchanged *context* whose text didn't match the file's real current
    content (specifically: text matching what an EARLIER, also-rejected
    attempt in the same retry loop would have produced, had it landed).
    `git apply`'s own message for this ("patch does not apply") doesn't
    say which line or what's actually there, so a retrying agent has to
    guess. This walks the same line-accounting git apply does and
    reports the first mismatch directly: the line number, what the diff
    assumed, and what the file actually contains right now.
    """
    mismatch = _find_context_mismatch(diff, current_content)
    if mismatch is None:
        return None
    old_line_no, marker, expected, actual = mismatch
    claim = "unchanged context" if marker == " " else "a line to remove"
    return (
        f"line {old_line_no} of the diff assumes (as {claim}): "
        f"{expected!r} -- but the file's real current line "
        f"{old_line_no} is: {actual!r}. This diff looks like it "
        "was built against an intended future version of the "
        "file, not what's actually there -- re-read this exact "
        "region fresh and construct a new diff against it, "
        "rather than adjusting the same one again."
    )


def _hunk_old_side_sequences(diff: str) -> list[list[str]]:
    """Every hunk's old-side line sequence (context ' ' + removed '-'
    lines, marker stripped, in order) -- one list per hunk. Used by the
    position-independent half of _check_content_fidelity below."""
    sequences: list[list[str]] = []
    lines = diff.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith("@@"):
            i += 1
            continue
        i += 1
        seq: list[str] = []
        while i < len(lines) and not lines[i].startswith("@@"):
            line = lines[i]
            i += 1
            if not line or line[0] not in " -":
                continue
            seq.append(line[1:])
        if seq:
            sequences.append(seq)
    return sequences


def _sequence_appears(seq: list[str], real_lines: list[str]) -> bool:
    """Does `seq` appear as a contiguous, exact block anywhere in
    real_lines -- regardless of position?"""
    n = len(seq)
    if n == 0:
        return True
    for start in range(len(real_lines) - n + 1):
        if real_lines[start : start + n] == seq:
            return True
    return False


_NEAREST_BLOCK_MAX_SHOWN = 12


def _locate_nearest_block(seq: list[str], real_lines: list[str]) -> int | None:
    """0-indexed start of the window of `real_lines` that `seq` most
    plausibly meant to transcribe, or None if nothing in the file
    resembles it. Every non-blank line of `seq` votes for the window
    start that would line it up with a whitespace-insensitive match;
    the most-voted start wins (earliest on ties). Purely a locator for
    the rejection message -- never used to decide accept/reject, or to
    rewrite the diff."""
    by_stripped: dict[str, list[int]] = {}
    for idx, real in enumerate(real_lines):
        stripped = real.strip()
        if stripped:
            by_stripped.setdefault(stripped, []).append(idx)
    votes: dict[int, int] = {}
    for offset, claimed in enumerate(seq):
        stripped = claimed.strip()
        if not stripped:
            continue
        for idx in by_stripped.get(stripped, ()):
            start = idx - offset
            if start >= 0:
                votes[start] = votes.get(start, 0) + 1
    if not votes:
        return None
    return min(votes, key=lambda start: (-votes[start], start))


def _describe_transcription_error(seq: list[str], real_lines: list[str]) -> str | None:
    """Explain HOW a hunk's old-side block differs from the nearest real
    block, and show that real block verbatim, so a retry can copy it
    instead of re-guessing. Added for the "right region, mistranscribed"
    failure: the pre-flight gate already rejected these, but only named
    the first bad line -- not where the real text lives or whether the
    defect was indentation, a dropped/added line, or a garbled token.
    Returns None when no nearby real block can be identified."""
    start = _locate_nearest_block(seq, real_lines)
    if start is None:
        return None
    window = real_lines[start : start + len(seq) + 2]
    matcher = difflib.SequenceMatcher(None, seq, window, autojunk=False)
    problems: list[str] = []
    opcodes = matcher.get_opcodes()
    for n, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            continue
        if tag == "replace" and n == len(opcodes) - 1 and (j2 - j1) > (i2 - i1):
            # The slack lines past the claimed block get merged into the
            # final replace; they aren't part of the transcription.
            j2 = j1 + (i2 - i1)
        if tag == "insert" and i1 >= len(seq):
            continue  # real lines past the end of the claimed block aren't a defect
        claimed = seq[i1:i2]
        actual = window[j1:j2]
        first = start + j1 + 1
        if tag == "replace" and len(claimed) == len(actual):
            for k, (c, a) in enumerate(zip(claimed, actual)):
                kind = (
                    "indentation/whitespace differs only"
                    if c.strip() == a.strip()
                    else "text differs"
                )
                problems.append(
                    f"diff has {c!r} but file line {first + k} is {a!r} ({kind})"
                )
        elif tag == "delete":
            problems.append(f"diff has extra line(s) not in the file: {claimed!r}")
        elif tag == "insert":
            problems.append(
                f"diff dropped real file line(s) starting at line {first}: {actual!r}"
            )
        else:
            problems.append(
                f"diff has {claimed!r} where the file (from line {first}) has {actual!r}"
            )
    if not problems:
        return None
    shown = window[: _NEAREST_BLOCK_MAX_SHOWN]
    block = "\n".join(f"{start + n + 1}: {line}" for n, line in enumerate(shown))
    more = "" if len(window) <= len(shown) else "\n..."
    return (
        f"Closest real block starts at line {start + 1}. Differences: "
        + "; ".join(problems[:5])
        + f".\nReal text there:\n{block}{more}"
    )


def _check_content_fidelity(
    diff: str, current_content: str, *, position_trustworthy: bool
) -> str | None:
    """Milestone 35/36's pre-flight content-fidelity gate: rejects a
    diff BEFORE git apply ever runs when its transcription of the
    file's content is unfaithful -- but deliberately does NOT reject a
    diff whose content is faithful and merely mis-positioned, per
    Milestone 35/36's own diagnosis: replaying the task's own real
    successful patch (an 8-line header miscount, confirmed via direct
    git apply replay) showed a wrong position with faithful content
    still applies cleanly -- git's own matching tolerates it. Position
    error alone must not become a new false-positive rejection here.

    Two modes, chosen by the caller based on whether the model's own
    most recent read of this region was numbered (`position_trustworthy`):

    - **Numbered read (strict, position-anchored):** the model was just
      shown exact line numbers for this exact region. Any context/
      removed line that doesn't match the file's real content AT THE
      POSITION THE DIFF ITSELF CLAIMS is a genuine transcription fault
      -- it had perfect information and still got it wrong. This is
      what caught the two real dropped/misaligned-content captures
      (Milestone 35's b0ae5a10, c1cef787), both preceded by a numbered
      read that plainly contradicts what the diff went on to claim.

    - **Unnumbered (or no tracked read at all) read (lenient,
      position-independent):** the model had no reliable position
      signal to begin with, so a position mismatch alone proves
      nothing about content fidelity -- this is exactly the real
      successful case's own shape (a scoped-but-unnumbered read, header
      off by 8, content correct). Instead, check whether each hunk's
      old-side line sequence appears as a contiguous, exact block
      *anywhere* in the file. If it does, the transcription is faithful
      even though the claimed position isn't -- let it through to git
      apply (which may still apply it, per the same evidence). If it
      appears nowhere at all, the content itself is fabricated or
      garbled, independent of position, and is rejected here.

    Distinct error code (CONTENT_MISMATCH) from the generic
    PATCH_CONFLICT git apply itself raises, so the two failure shapes
    stay separately diagnosable going forward.
    """
    real_lines = current_content.splitlines()

    if position_trustworthy:
        mismatch = _find_context_mismatch(diff, current_content)
        if mismatch is None:
            return None
        line_no, marker, expected, actual = mismatch
        claim = "unchanged context" if marker == " " else "a line to remove"
        message = (
            f"line {line_no} of the diff assumes (as {claim}): {expected!r} "
            f"-- but the file's real current line {line_no} is: {actual!r}. "
            "You were just shown this exact region, numbered, moments ago "
            "-- this diff's content does not match what you saw. Re-read "
            "this exact region fresh (with_line_numbers=true) and "
            "transcribe it exactly into your diff, rather than from "
            "memory or an earlier attempt."
        )
        # Detail only: the first hunk that isn't an exact block anywhere
        # is the one worth explaining; a faithful-but-misplaced hunk
        # gets the message above unchanged.
        for seq in _hunk_old_side_sequences(diff):
            if not _sequence_appears(seq, real_lines):
                detail = _describe_transcription_error(seq, real_lines)
                if detail is not None:
                    message = f"{message}\n{detail}"
                break
        return message

    for seq in _hunk_old_side_sequences(diff):
        if not _sequence_appears(seq, real_lines):
            # The decision (reject) is position-independent -- but the
            # message doesn't have to be vague about it. Reuse the
            # position-anchored walk purely for diagnostic detail (which
            # line, what was claimed, what's really there), same
            # specificity M17 originally built _diagnose_context_mismatch
            # for -- it just isn't what DECIDED this rejection.
            anchored = _find_context_mismatch(diff, current_content)
            if anchored is not None:
                line_no, marker, expected, actual = anchored
                claim = "unchanged context" if marker == " " else "a line to remove"
                detail = (
                    f"line {line_no} of the diff assumes (as {claim}): "
                    f"{expected!r} -- but the file's real current line "
                    f"{line_no} is: {actual!r}. "
                )
            else:
                detail = ""
            preview = seq[0] if seq else ""
            message = (
                f"{detail}A hunk's context/removed lines (starting "
                f"{preview!r}) do not appear as a contiguous, exact block "
                "ANYWHERE in the file's real current content -- regardless "
                "of position. This diff's content is fabricated or "
                "garbled, not just mis-positioned. Re-read the region "
                "you're actually targeting (with_line_numbers=true) and "
                "construct a new diff that transcribes its real content "
                "exactly."
            )
            nearest = _describe_transcription_error(seq, real_lines)
            return f"{message}\n{nearest}" if nearest is not None else message
    return None


def _find_diff_format_defect(diff: str) -> str | None:
    """Milestone 36's pre-flight format gate, checked before either the
    freshness gate or the content-fidelity check above: purely
    syntactic, never reads file content. Every line inside a hunk body
    must start with ' ' (context), '-' (removed), '+' (added), or '\\'
    (the standard "\\ No newline at end of file" marker) -- and must
    never be truly empty, since a blank CONTEXT line has to be
    represented as a single space, not zero characters.

    Real historical case (Milestone 35's diagnostic capture
    `d3799905`, task 637c1bda): every context line was missing its
    leading marker character entirely -- indistinguishable from a
    well-formed marker+content line by a pure first-character check
    alone when the real content itself starts with whitespace (Python
    indentation), which is exactly why this diff's blank line (which
    should be a single space, ' ') being fully empty is the reliable,
    unambiguous signal this check catches -- not an attempt to detect
    every possible missing-marker line by content inspection (that
    residual case is what _check_content_fidelity catches next, since
    a markerless line's stripped-by-one-character "content" won't
    match the real file either).
    """
    lines = diff.splitlines()
    in_hunk = False
    for lineno, line in enumerate(lines, start=1):
        if line.startswith("@@"):
            in_hunk = True
            continue
        if line.startswith(("diff --git", "index ", "--- ", "+++ ")):
            in_hunk = False
            continue
        if not in_hunk:
            continue
        if line == "":
            return (
                f"line {lineno} of the diff is empty inside a hunk body -- "
                "a blank context line must be a single space (' '), never "
                "zero characters. Malformed diff, rejected before ever "
                "reaching git apply."
            )
        if line[0] not in " -+\\":
            return (
                f"line {lineno} of the diff ({line!r}) does not start with "
                "a valid unified-diff marker (' ' for context, '-' for "
                "removed, '+' for added). Malformed diff, rejected before "
                "ever reaching git apply."
            )
    return None


# ---------------------------------------------------------------------
# Milestone 30 / spec's diagnosis-vs-mechanical-output gap, attempt 2:
# forced-fresh-read enforcement. Milestone 17 diagnosed the remaining
# (phantom-context) failure precisely -- the real historical case
# (task 09d7b2d1-b039-4e2f-9e9b-282bd2e4a2e5, still in amop_dev) shows
# every retry WAS preceded by a same-turn read_file, always covering
# lines 14-20; the diff's real target range was 14-23. A same-turn-read
# PRESENCE check would have passed that case right through unchanged --
# it has to be COVERAGE: does the most recent read of this path actually
# span every old-file line the diff's hunks touch, not merely "was there
# a read at all." This is what closes the gap Milestone 17 left open by
# name ("force a fresh, immediately-preceding read_file of the exact
# target region... never letting the Coder construct a diff against
# content read several turns -- or a prior failed retry attempt --
# earlier").
#
# Enforced entirely inside patch_file() below, gated on
# ctx.coder_read_tracking being non-None (see ToolContext's own
# docstring for why: off by default, on only for a real CoderAgent
# turn) -- never a prompt instruction, never advisory.
# ---------------------------------------------------------------------


def _diff_target_range(diff: str) -> tuple[int, int] | None:
    """The union of old-file line numbers (1-indexed, inclusive) this
    diff's hunks touch -- reuses the exact same line-accounting walk
    _diagnose_context_mismatch already does (old_line_no incremented
    per context/removed line), extracted into its own function since
    that one needs a real git-apply rejection and file content to
    compare against, and this one runs BEFORE any application is
    attempted, on the diff's own structure alone.

    A pure-insertion hunk (only '+' lines, zero old-file lines
    consumed) still touches its declared anchor position -- the point
    the new lines are inserted after/before -- so that position alone
    counts as the touched range for that hunk, not an empty range.

    Returns None for a diff with no parseable '@@' header at all
    (malformed input; let _normalize_diff_headers/git apply's own error
    handling deal with that, this gate isn't the place to invent a
    second error for it).
    """
    lines = diff.splitlines()
    touched_min: int | None = None
    touched_max: int | None = None
    i = 0
    while i < len(lines):
        m = _HUNK_HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        hunk_start = int(m.group(1))
        old_line_no = hunk_start
        i += 1
        while i < len(lines) and not lines[i].startswith("@@ "):
            line = lines[i]
            i += 1
            if not line or line[0] not in " -":
                continue  # '+' lines don't consume an old-file line number
            old_line_no += 1
        hunk_end = max(old_line_no - 1, hunk_start)
        touched_min = hunk_start if touched_min is None else min(touched_min, hunk_start)
        touched_max = hunk_end if touched_max is None else max(touched_max, hunk_end)
    if touched_min is None:
        return None
    return touched_min, touched_max


# Same defaults read_file() itself falls back to when it silently
# narrows a whole-file or oversized-range request (see that function,
# above) -- reused here rather than re-derived, so "what range did the
# caller actually get SHOWN" can never drift from what read_file itself
# actually decided to return.
_UNBOUNDED_END = 1_000_000_000


def read_file_effective_range(
    start_line: int | None, end_line: int | None, result: ToolResult
) -> tuple[int, int] | None:
    """What range of a file a read_file call actually SHOWED the caller,
    given its arguments and the ToolResult it produced -- not always the
    same as the range requested: a whole-file read on a large file, or
    an oversized explicit range, gets silently narrowed to
    _SCOPED_READ_DEFAULT_LINES (read_file's own behavior, unchanged by
    this milestone). Crediting the requested-but-not-actually-shown
    range would defeat the entire point of a FRESH-read requirement.

    Returns None for a failed read (nothing was shown, so nothing should
    be credited) or a non-string output (shouldn't happen for read_file,
    defensive).
    """
    if not result.success or not isinstance(result.output, str):
        return None
    truncated = result.output.startswith("[NOTE:")
    if start_line is None and end_line is None:
        if truncated:
            return 1, _SCOPED_READ_DEFAULT_LINES
        return 1, _UNBOUNDED_END
    lo = max(1, start_line or 1)
    if truncated:
        return lo, lo + _SCOPED_READ_DEFAULT_LINES - 1
    hi = end_line if end_line is not None else _UNBOUNDED_END
    return lo, hi


def _read_covers_target(
    read_range: tuple[int, int] | None, target_range: tuple[int, int]
) -> bool:
    if read_range is None:
        return False
    read_lo, read_hi = read_range
    target_lo, target_hi = target_range
    return read_lo <= target_lo and target_hi <= read_hi


def _truncate_for_capture(text: str) -> str:
    if len(text) <= _DIAGNOSTICS_MAX_CONTENT_CHARS:
        return text
    omitted = len(text) - _DIAGNOSTICS_MAX_CONTENT_CHARS
    return f"{text[:_DIAGNOSTICS_MAX_CONTENT_CHARS]}...[truncated {omitted} chars]"


def _capture_unexplained_patch_conflict(
    ctx: ToolContext,
    *,
    path: str,
    diff: str,
    current_content: str,
    target_range: tuple[int, int] | None,
    read_range: tuple[int, int] | None,
    stage: str,
    message: str,
) -> None:
    """Not a fix -- a trail. By the time patch_file reaches either of its
    two PATCH_CONFLICT returns, the Milestone 30 gate above has ALREADY
    let this attempt through (a stale/insufficient read is rejected
    earlier, as STALE_READ, before git apply is ever invoked). So a
    PATCH_CONFLICT reaching this point, during a real CoderAgent turn,
    is specifically the residual case Milestone 30's own commit message
    named but never itself investigated: "leaving a separate diff-
    construction failure mode isolated rather than hidden behind it" --
    referred to since as the "correct region, wrong transcription" gap.

    `agent_actions` (Milestone 23) already durably records this call's
    `diff` argument and a 2000-char-capped summary of the failure
    message -- real durability, but not enough to diagnose *why* a
    diff with fresh, sufficient context still failed to apply, because
    it never captures the file's actual content being compared against.
    This writes a companion snapshot with both sides of that comparison,
    uncapped up to a generous bound, to a location that outlives the
    task's own scratch dir (Milestone 25/31 clean those up).

    Best-effort and never fatal, same discipline as
    audit/actions.py's record_action: a diagnostics-write failure must
    never turn an already-failed (but otherwise normal) tool call into a
    crashed one.
    """
    try:
        DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
        task_id = getattr(ctx, "task_id", None)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        filename = f"patch_conflict_{task_id or 'no-task'}_{stamp}_{uuid.uuid4().hex[:8]}.json"
        payload = {
            "captured_at": datetime.now(UTC).isoformat(),
            "task_id": str(task_id) if task_id is not None else None,
            "agent_name": getattr(ctx, "agent_name", None),
            "path": path,
            "git_apply_stage": stage,
            "conflict_message": message,
            "target_range": list(target_range) if target_range else None,
            "read_range": list(read_range) if read_range else None,
            "diff": _truncate_for_capture(diff),
            "current_content": _truncate_for_capture(current_content),
        }
        (DIAGNOSTICS_DIR / filename).write_text(json.dumps(payload, indent=2))
    except Exception:  # noqa: BLE001 -- diagnostics capture is never allowed to be fatal
        pass


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
        "by a space, removed lines by '-', added lines by '+' -- every "
        "such line needs its marker character, even a blank context "
        "line (a single space, never truly empty). Three checks run "
        "before git apply ever sees the diff: error_code=MALFORMED_DIFF "
        "if a hunk line is missing its marker; error_code=STALE_READ if "
        "the most recent read_file on this path doesn't cover the "
        "diff's target lines, or was an unnumbered whole-file read (call "
        "read_file again with with_line_numbers=true); "
        "error_code=CONTENT_MISMATCH if a context or removed line "
        "doesn't match the file's real current content at that line. "
        "error_code=PATCH_CONFLICT means it passed all three checks but "
        "git apply itself still rejected it. Any of these: re-read the "
        "current file/region first, then retry with a fresh diff; never "
        "resubmit the same one unchanged. Use write_file instead for a "
        "brand new file or a genuinely trivial single-line change."
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

    # Milestone 36, check 3 of 3: diff-format validation. Runs first of
    # all three pre-flight checks below, and needs no file access at
    # all -- purely syntactic, so it's the cheapest and most decisive
    # rejection to reach first when it applies.
    format_defect = _find_diff_format_defect(diff)
    if format_defect is not None:
        return ToolResult(
            success=False, error_code="MALFORMED_DIFF", message=format_defect
        )

    # Milestone 30: forced-fresh-read gate (coverage), extended by
    # Milestone 36, check 1 of 3 (numbered). Only active during a real
    # CoderAgent turn (ctx.coder_read_tracking is not None -- see
    # ToolContext's own docstring); every other caller, including every
    # test that calls patch_file directly, is unaffected. Checked before
    # touching the sandbox at all -- a stale-context attempt never
    # reaches git apply, matching Milestone 17's own named fix: "reject
    # it before it ever reaches git apply."
    #
    # target_range/read_range initialized here (not just inside the
    # branches below) so the still-unexplained-conflict capture further
    # down always has a defined value to report, including the
    # gate-inactive case (ctx.coder_read_tracking is None).
    target_range: tuple[int, int] | None = None
    read_range: tuple[int, int] | None = None
    if ctx.coder_read_tracking is not None:
        target_range = _diff_target_range(diff)
        if target_range is not None:
            read_range = ctx.coder_read_tracking.get(path)
            if not _read_covers_target(read_range, target_range):
                target_lo, target_hi = target_range
                seen = (
                    f"lines {read_range[0]}-{read_range[1]}"
                    if read_range is not None
                    else "no prior read_file call on this path this attempt"
                )
                return ToolResult(
                    success=False,
                    error_code="STALE_READ",
                    message=(
                        f"this diff touches lines {target_lo}-{target_hi} of "
                        f"{path!r}, but the most recent read_file on it "
                        f"covered {seen} -- that doesn't fully cover the "
                        "diff's target range, so it was rejected before "
                        "ever reaching git apply. Call read_file again with "
                        f"start_line<={target_lo}, end_line>={target_hi} "
                        "(with_line_numbers=true if you need exact line "
                        "numbers for the '@@' header) to see the file's "
                        "real, current content there, then construct a new "
                        "diff against what you actually just saw."
                    ),
                )
            # Milestone 36, check 1: coverage alone isn't enough for an
            # unscoped whole-file read (read_range's own sentinel value
            # for "no start_line/end_line given at all" -- see
            # read_file_effective_range) -- it trivially "covers"
            # anything, but two real captured cases (Milestone 35's
            # ee31f113/d3799905) show that's not the same as the model
            # actually knowing where things are. A deliberately scoped
            # read is treated differently even when also unnumbered --
            # it's a bounded, intentional request, not a guess across an
            # arbitrarily long, uncounted file (and Milestone 35's own
            # successful 8-line-miscount replay came from exactly this
            # scoped-unnumbered shape, which must keep working).
            if read_range == (1, _UNBOUNDED_END) and not (
                ctx.coder_read_numbered or {}
            ).get(path, False):
                return ToolResult(
                    success=False,
                    error_code="STALE_READ",
                    message=(
                        f"the most recent read_file on {path!r} covered the "
                        "whole file, but as an unnumbered, unscoped "
                        "whole-file read -- not enough for patch_file to "
                        "trust the position this diff claims. Call "
                        "read_file again with with_line_numbers=true "
                        "(scoped to the region you're editing, or the "
                        "whole file if it's small) so you know the real "
                        "line numbers before constructing the diff's '@@' "
                        "header."
                    ),
                )

    container_path = _container_path(target, ctx.scratch_dir)
    relative_path = container_path.removeprefix("/workspace/")

    try:
        current_content = await asyncio.to_thread(ctx.sandbox.read_file, container_path)
    except FileNotFoundError:
        return ToolResult(
            success=False,
            error_code="NOT_FOUND",
            message=f"No such file: {path} -- use write_file to create a new file",
        )
    except Exception as exc:
        return ToolResult(success=False, error_code="READ_ERROR", message=str(exc))

    # Milestone 36, check 2 of 3: content fidelity. Runs after the gate
    # above (which needs no file content) but still before git apply --
    # the actual decisive factor per Milestone 35/36's diagnosis (see
    # _check_content_fidelity's own docstring). Distinct error code from
    # PATCH_CONFLICT so this failure shape stays separately diagnosable
    # in agent_actions going forward, rather than blending into git
    # apply's generic rejection.
    #
    # position_trustworthy: strict, position-anchored checking only
    # when we KNOW the model just saw accurate line numbers for this
    # exact, covering region -- never for an unnumbered read (gate-
    # active or not), where a position mismatch alone proves nothing
    # (the real successful 8-line-miscount case has exactly this shape:
    # a scoped-but-unnumbered read, wrong header, faithful content).
    position_trustworthy = bool(
        ctx.coder_read_tracking is not None
        and (ctx.coder_read_numbered or {}).get(path, False)
    )
    fidelity_defect = _check_content_fidelity(
        diff, current_content, position_trustworthy=position_trustworthy
    )
    if fidelity_defect is not None:
        return ToolResult(
            success=False, error_code="CONTENT_MISMATCH", message=fidelity_defect
        )

    normalized = _normalize_diff_headers(diff, relative_path)
    patch_path = f"/tmp/amop-patch-{uuid.uuid4().hex}.diff"

    def _conflict_message(stderr: str, stdout: str) -> str:
        base = (stderr or stdout or "diff did not apply cleanly").strip()
        # Milestone 17: append a precise explanation when one is findable,
        # rather than leaving the agent to guess from git's own message
        # (which never names a line or shows real content).
        detail = _diagnose_context_mismatch(normalized, current_content)
        return f"{base}\n{detail}" if detail else base

    try:
        await asyncio.to_thread(ctx.sandbox.write_file, patch_path, normalized)

        check = await asyncio.to_thread(
            ctx.sandbox.exec_run, f"git apply --check --recount {patch_path}"
        )
        if check.exit_code != 0:
            conflict_message = _conflict_message(check.stderr, check.stdout)
            if ctx.coder_read_tracking is not None:
                _capture_unexplained_patch_conflict(
                    ctx,
                    path=path,
                    diff=diff,
                    current_content=current_content,
                    target_range=target_range,
                    read_range=read_range,
                    stage="check",
                    message=conflict_message,
                )
            return ToolResult(
                success=False,
                error_code="PATCH_CONFLICT",
                message=conflict_message,
            )

        applied = await asyncio.to_thread(
            ctx.sandbox.exec_run, f"git apply --recount {patch_path}"
        )
        if applied.exit_code != 0:
            conflict_message = _conflict_message(applied.stderr, applied.stdout)
            if ctx.coder_read_tracking is not None:
                _capture_unexplained_patch_conflict(
                    ctx,
                    path=path,
                    diff=diff,
                    current_content=current_content,
                    target_range=target_range,
                    read_range=read_range,
                    stage="apply",
                    message=conflict_message,
                )
            return ToolResult(
                success=False,
                error_code="PATCH_CONFLICT",
                message=conflict_message,
            )
    except Exception as exc:
        return ToolResult(success=False, error_code="PATCH_ERROR", message=str(exc))
    finally:
        await asyncio.to_thread(ctx.sandbox.exec_run, f"rm -f {patch_path}")

    output = await _micro_commit_after_edit(
        ctx, "patch_file", path, f"applied patch to {path}"
    )
    # Milestone 30: content just changed -- the tracked read is now
    # stale relative to the new file content. Cleared rather than left
    # around: a second patch_file on the same path later in this same
    # attempt must re-read post-edit content, not reuse a pre-edit
    # range that happened to numerically cover the new target too.
    if ctx.coder_read_tracking is not None:
        ctx.coder_read_tracking.pop(path, None)
    return ToolResult(success=True, output=output)


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

    Milestone 25 note on that claim: it was only ever true of this
    function. The invocation side (the run_tests body below) was a
    single hardcoded pytest command with no detection logic at all --
    this docstring's own "runner-agnostic per 8.6" was aspirational
    until _parse_jest_json (below) gave a second report format feeding
    the exact same normalized shape.
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


def _parse_jest_json(json_text: str) -> dict:
    """Normalize Jest's `--json` output into the exact same Section 8.6
    shape _parse_junit_xml produces. This is the actual runner-
    agnosticism fix: two structurally different report formats (JUnit
    XML for pytest, Jest's own JSON for Jest) feeding one shape, so
    nothing downstream (the orchestrator's ground-truth check, the
    flaky-test re-run logic) needs to know which runner produced it.

    `--testLocationInResults` (passed alongside `--json` in run_tests
    below) is what makes `location.line` populated -- without it Jest
    leaves `location: null` on every assertion and the normalized
    `line` field would always be None, a real loss relative to
    pytest's junit-xml (which always carries a line number).
    """
    data = json.loads(json_text)
    failures = []
    for suite in data.get("testResults", []):
        file_path = suite.get("name")
        for assertion in suite.get("assertionResults", []):
            if assertion.get("status") != "failed":
                continue
            location = assertion.get("location") or {}
            messages = assertion.get("failureMessages") or []
            failures.append(
                {
                    "test_name": assertion.get("fullName") or assertion.get("title", ""),
                    "message": (messages[0] if messages else "")[:500],
                    "file": file_path,
                    "line": location.get("line"),
                }
            )
    return {
        "passed": data.get("numPassedTests", 0),
        "failed": data.get("numFailedTests", 0),
        "skipped": data.get("numPendingTests", 0),
        "failures": failures,
    }


@tool(
    name="run_tests",
    description=(
        "Run the repository's test suite inside the sandbox (pytest for "
        "Python repos, Jest for JavaScript) and return structured "
        "results. Optionally pass 'path' to run only the tests at that "
        "path, and/or 'test_name' to run only tests matching that name. "
        "Does not modify any source file."
    ),
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}, "test_name": {"type": "string"}},
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
async def run_tests(
    ctx: ToolContext, path: str | None = None, test_name: str | None = None
) -> ToolResult:
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

    # Milestone 25: the actual runner-dispatch fix. ctx.stack is
    # resolved once in chain.py (detect_stack, right after materialize,
    # before the sandbox is even created -- see indexer.py) and passed
    # in as plain data, same D-8 pattern as permission_overrides. Python
    # path below is byte-for-byte the pre-Milestone-25 command.
    # Milestone 26: selecting ONE test is runner-specific, and getting it
    # wrong is dangerous rather than merely broken. Before this, callers
    # built a pytest nodeid (`file::test_name`) and passed it as `path`.
    # pytest understands that natively; Jest does not -- it takes
    # positional args as path regexes, so the `::name` suffix made the
    # path resolve to /workspace and Jest silently ran the ENTIRE SUITE.
    # The orchestrator's flaky-test carve-out
    # (_rerun_failing_tests_against_base) then read "the whole suite
    # fails on base" as "this individual test fails on base too",
    # excluded the real failure as environmental noise, and forced
    # all_passed=True on a red suite. Name filtering therefore goes
    # through each runner's own flag, never through the path.
    name_filter = ""
    if test_name:
        # -k (pytest) / -t (jest). Both take the name as a pattern, so a
        # name containing regex/expression metacharacters could still
        # over-match -- acceptable here because the caller uses this to
        # re-check a test it already knows failed, and an over-match can
        # only make the re-run look MORE failing, which the fail-safe
        # default already treats as "not conclusively flaky".
        flag = "-t" if ctx.stack in JEST_STACKS else "-k"
        name_filter = f" {flag} {shlex.quote(test_name)}"

    if ctx.stack in JEST_STACKS:
        report_path = f"/tmp/amop-jest-{uuid.uuid4().hex}.json"
        jest_target = "" if target == "." else f" {target}"
        command = (
            f"cd /workspace && jest --json --testLocationInResults "
            f"--outputFile={report_path}{jest_target}{name_filter}"
        )
    else:
        report_path = f"/tmp/amop-junit-{uuid.uuid4().hex}.xml"
        command = (
            f"cd /workspace && python -m pytest {target}{name_filter} "
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
        report_text = await asyncio.to_thread(ctx.sandbox.read_file, report_path)
        if ctx.stack in JEST_STACKS:
            summary = _parse_jest_json(report_text)
        else:
            summary = _parse_junit_xml(report_text)
    except (FileNotFoundError, ET.ParseError, json.JSONDecodeError):
        # The runner couldn't even produce a report (collection error,
        # no tests found, no config file, ...). Surface the raw output
        # as evidence instead of pretending we got a clean zero-failure
        # result.
        runner_name = "jest" if ctx.stack in JEST_STACKS else "pytest"
        return ToolResult(
            success=False,
            error_code="NO_TEST_REPORT",
            message=(result.stdout + result.stderr)[-2000:]
            or f"{runner_name} produced no report",
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


# ---------------------------------------------------------------------
# Milestone 14 tools: measurement for the Optimizer agent (Section 6.6).
#
# Both are read-only: they observe how the code behaves, they don't
# change it. That matters for permissions -- Optimizer runs `suggestor`,
# but an Investigator-style read-only caller must be able to profile
# without being denied for attempting a mutation.
# ---------------------------------------------------------------------

# Section 6.6: "profile before hypothesizing (never guess at a
# bottleneck without a profile in evidence)".
BENCHMARK_TIMEOUT_SECONDS = float(
    os.environ.get("AMOP_BENCHMARK_TIMEOUT_SECONDS", "120")
)
DEFAULT_BENCHMARK_ITERATIONS = 5
MAX_BENCHMARK_ITERATIONS = 25


def _benchmark_script(container_path: str, iterations: int) -> str:
    """Time one entry point N times and report the MEDIAN, not the mean.

    Median because this runs on a developer laptop sharing a CPU with
    Docker, an editor, and (in this project's case) a 9GB language model
    -- a single outlier scheduling stall would otherwise move the mean
    enough to fake a regression or fake an improvement. Section 6.6
    explicitly warns against shipping a "noisy-benchmark change", and a
    threshold applied to a mean of 5 runs on a busy machine is exactly
    that.

    Also reports min/max so a caller can see the spread rather than
    trusting one number: a median that improved while the spread covers
    the difference is not evidence of anything.
    """
    return (
        "python - <<'AMOP_BENCH'\n"
        "import json, runpy, statistics, sys, time\n"
        "times = []\n"
        f"for _ in range({iterations}):\n"
        "    start = time.perf_counter()\n"
        f"    runpy.run_path({container_path!r}, run_name='__main__')\n"
        "    times.append((time.perf_counter() - start) * 1000.0)\n"
        "print('AMOP_BENCH_JSON' + json.dumps({\n"
        "    'median_ms': statistics.median(times),\n"
        "    'min_ms': min(times),\n"
        "    'max_ms': max(times),\n"
        "    'iterations': len(times),\n"
        "}))\n"
        "AMOP_BENCH\n"
    )


def _extract_marked_json(stdout: str, marker: str) -> dict | None:
    """Pull our own JSON line out of stdout.

    The benchmarked code prints whatever it likes, so the result is
    tagged with a marker rather than assuming the last line is ours --
    a target that ends by printing its own summary would otherwise
    silently become the "measurement".
    """
    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            try:
                return json.loads(line[len(marker) :])
            except json.JSONDecodeError:
                return None
    return None


@tool(
    name="run_benchmark",
    description=(
        "Time a Python entry point inside the workspace, running it "
        "several times and reporting the median wall-clock milliseconds "
        "(plus min/max so you can see the noise). Give the path to a "
        "script with a __main__ block that does the work, e.g. "
        "'benchmark.py'. Read-only -- it runs the code, it does not "
        "change it. Use this BEFORE and AFTER an optimization, against "
        "the same entry point, to show whether the change actually helped."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "iterations": {"type": "integer"},
        },
        "required": ["path"],
    },
    mutating=False,
    timeout_seconds=BENCHMARK_TIMEOUT_SECONDS + 10,
)
async def run_benchmark(
    path: str, ctx: ToolContext, iterations: int = DEFAULT_BENCHMARK_ITERATIONS
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
    iterations = max(1, min(int(iterations), MAX_BENCHMARK_ITERATIONS))
    container_path = _container_path(target, ctx.scratch_dir)

    command = f"cd /workspace && " + _benchmark_script(container_path, iterations)
    try:
        result = await asyncio.to_thread(
            ctx.sandbox.exec_run, command, BENCHMARK_TIMEOUT_SECONDS
        )
    except Exception as exc:
        return ToolResult(success=False, error_code="RUN_ERROR", message=str(exc))

    if result.timed_out:
        return ToolResult(
            success=False,
            error_code="TIMEOUT",
            message=f"benchmark exceeded {BENCHMARK_TIMEOUT_SECONDS}s",
        )

    payload = _extract_marked_json(result.stdout, "AMOP_BENCH_JSON")
    if payload is None:
        return ToolResult(
            success=False,
            error_code="BENCHMARK_FAILED",
            message=(result.stdout + result.stderr)[-2000:]
            or "benchmark produced no measurement",
        )
    return ToolResult(success=True, output=payload)


@tool(
    name="profile_code",
    description=(
        "Profile a Python entry point inside the workspace with cProfile "
        "and return the functions taking the most cumulative time. Give "
        "the path to a script with a __main__ block, e.g. 'benchmark.py'. "
        "Returns two rankings: 'top' by cumulative time (which "
        "includes everything a function calls) and 'top_by_self_time' "
        "by time spent IN the function itself. For finding a hot spot, "
        "read top_by_self_time first -- cumulative time credits every "
        "caller up the stack, so it flatters wrappers. Read-only. Use "
        "this FIRST, before deciding what to optimize -- a bottleneck "
        "you guessed at is not a bottleneck you found."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "top_n": {"type": "integer"},
        },
        "required": ["path"],
    },
    mutating=False,
    timeout_seconds=BENCHMARK_TIMEOUT_SECONDS + 10,
)
async def profile_code(path: str, ctx: ToolContext, top_n: int = 15) -> ToolResult:
    """Spec 6.6 names this tool `profile_endpoint`.

    Renamed deliberately, not quietly: `profile_endpoint` presumes an
    HTTP service, and nothing in this project's scope serves one. What a
    repo here actually has is a Python entry point, so this profiles
    that. The capability 6.6 depends on -- "never guess at a bottleneck
    without a profile in evidence" -- is unchanged; only the assumption
    about what is being profiled is.
    """
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
    top_n = max(1, min(int(top_n), 50))
    container_path = _container_path(target, ctx.scratch_dir)

    command = (
        "cd /workspace && python - <<'AMOP_PROF'\n"
        "import cProfile, pstats, io, json, runpy\n"
        "profiler = cProfile.Profile()\n"
        "profiler.enable()\n"
        f"runpy.run_path({container_path!r}, run_name='__main__')\n"
        "profiler.disable()\n"
        "buf = io.StringIO()\n"
        "stats = pstats.Stats(profiler, stream=buf).sort_stats('cumulative')\n"
        "rows = []\n"
        "for func, (cc, nc, tt, ct, callers) in stats.stats.items():\n"
        "    filename = func[0]\n"
        # The profiler's own scaffolding -- runpy frames and the exec
        # that launches them -- has, by construction, a cumulative time
        # covering the ENTIRE run, so sorting by cumtime puts it above
        # every real function. Live-confirmed the hard way: the first
        # real Optimizer run got a top-5 of nothing but runpy frames and
        # concluded "No bottleneck found in the profile". Dropping these
        # is not cosmetic -- it is the difference between a profile that
        # names the hot function and one that hides it.
        "    if filename.startswith('<frozen') or filename in ('~', '<string>'):\n"
        "        continue\n"
        "    rows.append({'function': f'{func[0]}:{func[1]}({func[2]})',\n"
        "                 'calls': nc, 'tottime_s': round(tt, 6),\n"
        "                 'cumtime_s': round(ct, 6)})\n"
        "by_cum = sorted(rows, key=lambda r: r['cumtime_s'], reverse=True)\n"
        # Self time (excluding subcalls) is what actually identifies a
        # hot spot; cumulative time credits every caller up the stack.
        "by_self = sorted(rows, key=lambda r: r['tottime_s'], reverse=True)\n"
        f"print('AMOP_PROF_JSON' + json.dumps({{'top': by_cum[:{top_n}],\n"
        f"                                      'top_by_self_time': by_self[:{top_n}]}}))\n"
        "AMOP_PROF\n"
    )
    try:
        result = await asyncio.to_thread(
            ctx.sandbox.exec_run, command, BENCHMARK_TIMEOUT_SECONDS
        )
    except Exception as exc:
        return ToolResult(success=False, error_code="RUN_ERROR", message=str(exc))

    if result.timed_out:
        return ToolResult(
            success=False,
            error_code="TIMEOUT",
            message=f"profile exceeded {BENCHMARK_TIMEOUT_SECONDS}s",
        )

    payload = _extract_marked_json(result.stdout, "AMOP_PROF_JSON")
    if payload is None:
        return ToolResult(
            success=False,
            error_code="PROFILE_FAILED",
            message=(result.stdout + result.stderr)[-2000:]
            or "profile produced no output",
        )
    return ToolResult(success=True, output=payload)
