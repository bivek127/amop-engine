"""Monitoring Tools -- spec Section 8.7.

search_logs / get_metrics / get_error_rate / list_recent_errors: thin,
read-only wrappers over "the configured log/metric backends (Section
3.4's Prometheus, or file-tail/syslog per original Section 7.3)".

This dev setup runs neither Prometheus nor any other real metrics
backend -- there is no config.example.yaml anywhere in this codebase
(config is env-var only, the existing AMOP_* convention), and nothing
in amop/ ever references Prometheus outside the spec's own architecture
diagram. Building get_metrics against a metrics endpoint that doesn't
exist would be exactly the fake backend this module must not be. These
four tools are file-tail based instead -- the spec's own named
fallback -- reading a single operator-configured log file
(AMOP_LOG_SOURCE_PATH), never a path supplied by the caller/agent, so
there is no path-traversal surface here at all (contrast
filesystem.py's read_file, which takes a caller-supplied path and needs
resolve_within_scratch for exactly that reason).

Log line format this module understands (see tests/fixtures/
monitored_service/app.py, the real running sample app these tools are
demonstrated against): one event per line,
  "<ISO-8601 UTC timestamp> <INFO|WARNING|ERROR> <message>"
A line that doesn't match is skipped, not an error -- a real log file
may reasonably contain other lines (blank lines, a startup banner), and
one unparseable line must not make every read fail.

These tools never touch ctx.sandbox and behave identically whether or
not a task/sandbox exists -- same host-side, non-sandboxed shape as
tools/advisories.py's check_advisories (spec 8.7 calls this
`requires_repo_context: false`; this codebase's ToolSpec has no such
field to set, so the contract is expressed by never reading ctx.sandbox
here, not by a metadata flag).
"""

import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from amop.tools.registry import ToolContext, ToolResult, tool

_LOG_LINE_RE = re.compile(r"^(\S+)\s+(INFO|WARNING|ERROR)\s+(.*)$")
_LEVELS = ("INFO", "WARNING", "ERROR")


def _log_source_path() -> Path | None:
    raw = os.environ.get("AMOP_LOG_SOURCE_PATH")
    return Path(raw) if raw else None


class LogLine:
    __slots__ = ("timestamp", "level", "message")

    def __init__(self, timestamp: datetime, level: str, message: str) -> None:
        self.timestamp = timestamp
        self.level = level
        self.message = message


def _parse_line(raw: str) -> LogLine | None:
    match = _LOG_LINE_RE.match(raw.rstrip("\n"))
    if not match:
        return None
    ts_raw, level, message = match.groups()
    try:
        timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return LogLine(timestamp=timestamp, level=level, message=message)


def _read_all_lines(path: Path) -> list[LogLine]:
    lines: list[LogLine] = []
    with path.open("r", errors="replace") as fh:
        for raw in fh:
            parsed = _parse_line(raw)
            if parsed is not None:
                lines.append(parsed)
    return lines


def _unavailable(path: Path | None) -> ToolResult:
    detail = (
        "no log source configured -- set AMOP_LOG_SOURCE_PATH to a real log file"
        if path is None
        else f"configured log source {str(path)!r} does not exist or is not readable"
    )
    return ToolResult(
        success=False,
        error_code="LOG_SOURCE_UNAVAILABLE",
        message=f"{detail}. This is NOT a statement that there are no errors.",
    )


def _in_window(lines: list[LogLine], window_minutes: float) -> list[LogLine]:
    cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)
    return [ln for ln in lines if ln.timestamp >= cutoff]


@tool(
    name="search_logs",
    description=(
        "Search the configured log source for lines whose message contains "
        "`query` (case-insensitive substring), within the last "
        "`since_minutes` minutes (default 60). Read-only, file-tail based "
        "(no Prometheus is running in this setup -- see module docstring). "
        "Returns matching lines with their parsed timestamp and level."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "since_minutes": {"type": "number"},
        },
        "required": ["query"],
    },
    mutating=False,
)
async def search_logs(
    query: str, ctx: ToolContext, since_minutes: float = 60
) -> ToolResult:
    path = _log_source_path()
    if path is None or not path.is_file():
        return _unavailable(path)
    lines = _in_window(_read_all_lines(path), since_minutes)
    needle = query.lower()
    matches = [ln for ln in lines if needle in ln.message.lower()]
    return ToolResult(
        success=True,
        output={
            "since_minutes": since_minutes,
            "query": query,
            "matches": [
                {
                    "timestamp": ln.timestamp.isoformat(),
                    "level": ln.level,
                    "message": ln.message,
                }
                for ln in matches
            ],
        },
    )


@tool(
    name="get_error_rate",
    description=(
        "Compute the ERROR-level log line rate as a percentage of all log "
        "lines in the trailing `window_minutes` window (default 5) -- the "
        "metric Watcher's threshold rule (spec 6.1) evaluates every poll "
        "cycle. Read-only."
    ),
    parameters={
        "type": "object",
        "properties": {"window_minutes": {"type": "number"}},
        "required": [],
    },
    mutating=False,
)
async def get_error_rate(ctx: ToolContext, window_minutes: float = 5) -> ToolResult:
    path = _log_source_path()
    if path is None or not path.is_file():
        return _unavailable(path)
    lines = _in_window(_read_all_lines(path), window_minutes)
    total = len(lines)
    errors = sum(1 for ln in lines if ln.level == "ERROR")
    rate = (errors / total * 100) if total else 0.0
    return ToolResult(
        success=True,
        output={
            "window_minutes": window_minutes,
            "total": total,
            "errors": errors,
            "error_rate_pct": rate,
        },
    )


@tool(
    name="get_metrics",
    description=(
        "Return a log-derived metrics snapshot for the trailing "
        "`window_minutes` window (default 60): counts by level, total "
        "volume, throughput (lines/minute), and error_rate_pct. There is no "
        "real metrics backend in this setup (see module docstring) -- every "
        "number here is computed directly from real log lines, not a "
        "simulated or fabricated source. Read-only."
    ),
    parameters={
        "type": "object",
        "properties": {"window_minutes": {"type": "number"}},
        "required": [],
    },
    mutating=False,
)
async def get_metrics(ctx: ToolContext, window_minutes: float = 60) -> ToolResult:
    path = _log_source_path()
    if path is None or not path.is_file():
        return _unavailable(path)
    lines = _in_window(_read_all_lines(path), window_minutes)
    total = len(lines)
    by_level = {level: sum(1 for ln in lines if ln.level == level) for level in _LEVELS}
    rate = (by_level["ERROR"] / total * 100) if total else 0.0
    return ToolResult(
        success=True,
        output={
            "window_minutes": window_minutes,
            "total_lines": total,
            "by_level": by_level,
            "error_rate_pct": rate,
            "lines_per_minute": (total / window_minutes) if window_minutes else 0.0,
        },
    )


@tool(
    name="list_recent_errors",
    description=(
        "Return the most recent ERROR-level log lines (default 20, newest "
        "first) from the configured log source, across the whole file (not "
        "windowed). Read-only."
    ),
    parameters={
        "type": "object",
        "properties": {"limit": {"type": "number"}},
        "required": [],
    },
    mutating=False,
)
async def list_recent_errors(ctx: ToolContext, limit: int = 20) -> ToolResult:
    path = _log_source_path()
    if path is None or not path.is_file():
        return _unavailable(path)
    errors = [ln for ln in _read_all_lines(path) if ln.level == "ERROR"]
    errors.sort(key=lambda ln: ln.timestamp, reverse=True)
    return ToolResult(
        success=True,
        output=[
            {"timestamp": ln.timestamp.isoformat(), "message": ln.message}
            for ln in errors[:limit]
        ],
    )
