"""Advisory lookup — spec Section 6.7's `check_advisories`, the input
side of the Dependency Updater agent.

Host-side, not sandboxed, and that is not a shortcut: the sandbox runs
with `network_mode="none"` (Section 9.3, sandbox/manager.py), so nothing
inside it can reach an advisory feed at all. Milestone 6 hit the same
wall for `git push` and established the precedent -- the network call
happens host-side, the sandbox stays sealed. The manifest itself is
still read through the sandbox, so the agent can only ever ask about a
file inside its own workspace.

Source is OSV (osv.dev), which Section 6.7 names directly ("advisory feed
match (OSV/GitHub Security Advisories)"). Its query API needs no auth
token, which matters here: an unauthenticated read-only lookup is the
one external call in this codebase that doesn't widen the credential
surface at all.

Offline behavior is a deliberate, explicit `ADVISORY_LOOKUP_FAILED`
rather than an empty result. "No advisories found" and "I could not
reach the advisory database" are opposite claims, and quietly returning
the former for the latter would tell an autonomous agent that a
vulnerable dependency is clean.
"""

import asyncio
import os
import re

import httpx

from amop.safety.engine import resolve_within_scratch
from amop.tools.registry import ToolContext, ToolResult, tool

OSV_QUERY_URL = os.environ.get("AMOP_OSV_URL", "https://api.osv.dev/v1/query")
OSV_TIMEOUT_SECONDS = float(os.environ.get("AMOP_OSV_TIMEOUT_SECONDS", "10"))

# Only pinned requirements are checkable: an advisory is a statement
# about a specific version, so "requests" with no version tells us
# nothing to look up. Unpinned lines are reported back as skipped rather
# than silently dropped.
_PINNED_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*==\s*([A-Za-z0-9._+!-]+)\s*$")


def parse_requirements(content: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Split a requirements.txt into (pinned, unpinnable).

    Pure and separately testable -- the parsing is the part most likely
    to be wrong, and it shouldn't need a network round trip to exercise.
    """
    pinned: list[tuple[str, str]] = []
    skipped: list[str] = []
    for raw in content.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _PINNED_RE.match(line)
        if match:
            pinned.append((match.group(1), match.group(2)))
        else:
            skipped.append(line)
    return pinned, skipped


async def _query_osv(client: httpx.AsyncClient, name: str, version: str) -> list[dict]:
    response = await client.post(
        OSV_QUERY_URL,
        json={"package": {"name": name, "ecosystem": "PyPI"}, "version": version},
    )
    response.raise_for_status()
    return response.json().get("vulns", []) or []


def _fixed_versions(vuln: dict, package: str) -> list[str]:
    """Versions OSV says the fix landed in, for the given package.

    Reported as a list because a CVE affecting multiple release branches
    genuinely has several fixed versions, and picking one here would be
    the tool deciding an upgrade target that belongs to the agent (and
    ultimately to the test suite).
    """
    fixed: list[str] = []
    for affected in vuln.get("affected", []) or []:
        if (affected.get("package") or {}).get("name", "").lower() != package.lower():
            continue
        for rng in affected.get("ranges", []) or []:
            for event in rng.get("events", []) or []:
                if "fixed" in event:
                    fixed.append(event["fixed"])
    return sorted(set(fixed))


@tool(
    name="check_advisories",
    description=(
        "Check a Python requirements file for known security advisories "
        "(CVEs) against the OSV database. Give the path to the manifest "
        "inside the workspace, e.g. 'requirements.txt'. Returns one entry "
        "per vulnerable pinned dependency, each with its advisory ids, "
        "summary, and the version(s) the fix landed in. Read-only -- it "
        "never edits the manifest; use write_file or patch_file for that."
    ),
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    mutating=False,
    timeout_seconds=60,
)
async def check_advisories(path: str, ctx: ToolContext) -> ToolResult:
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

    relative = str(target.relative_to(ctx.scratch_dir.resolve()))
    container_path = f"/workspace/{relative}"
    try:
        content = await asyncio.to_thread(ctx.sandbox.read_file, container_path)
    except FileNotFoundError:
        return ToolResult(
            success=False, error_code="NOT_FOUND", message=f"No such file: {path}"
        )
    except Exception as exc:
        return ToolResult(success=False, error_code="READ_ERROR", message=str(exc))

    pinned, skipped = parse_requirements(content)
    if not pinned:
        return ToolResult(
            success=True,
            output={
                "manifest": relative,
                "checked": 0,
                "skipped_unpinned": skipped,
                "advisories": [],
            },
        )

    advisories = []
    try:
        async with httpx.AsyncClient(timeout=OSV_TIMEOUT_SECONDS) as client:
            for name, version in pinned:
                for vuln in await _query_osv(client, name, version):
                    advisories.append(
                        {
                            "package": name,
                            "installed_version": version,
                            "id": vuln.get("id", ""),
                            "aliases": vuln.get("aliases", []) or [],
                            "summary": (vuln.get("summary") or "")[:300],
                            "fixed_versions": _fixed_versions(vuln, name),
                        }
                    )
    except Exception as exc:
        # See module docstring: an unreachable feed must never look like
        # a clean bill of health.
        return ToolResult(
            success=False,
            error_code="ADVISORY_LOOKUP_FAILED",
            message=(
                f"could not reach the advisory database ({exc}) -- this is "
                "NOT a statement that the dependencies are clean"
            ),
        )

    return ToolResult(
        success=True,
        output={
            "manifest": relative,
            "checked": len(pinned),
            "skipped_unpinned": skipped,
            "advisories": advisories,
        },
    )
