"""Protected infrastructure paths — spec Section 12.2.1, Design Decision
D-12.

No agent, regardless of permission mode — including `autonomous` — may
*write* to a CI/CD-config path. This exists as its own check, separate
from the command blacklist (`safety/blacklist.py`, Section 12.4):
the blacklist stops a dangerous *command* (`rm -rf`); it has no opinion
on a dangerous *target* reached via an entirely ordinary, already-
permitted tool call (`write_file`, `patch_file`). A modified CI workflow
file is a supply-chain vector distinct from any single command — it
doesn't need `rm -rf` to be dangerous, it just needs to alter what runs
automatically on the next push, which can exfiltrate secrets (Section
12.5) through a channel the Safety Engine doesn't otherwise inspect.
Path-based protection catches this regardless of which tool or command
got there.

`safety/engine.py`'s own module docstring already named
`protected_path_match()` as deferred from Milestone 2 -- this is that
deferred piece, not new surface area.
"""

import os
from pathlib import Path

from amop.safety.engine import resolve_within_scratch
from amop.tools.registry import ToolContext, ToolSpec

# The hardcoded baseline (spec's own list, verbatim). A plain module
# constant, never derived from config or any override mechanism -- that
# is precisely what makes it a baseline rather than a default. Directory
# entries carry a trailing "/" (match anything nested under them);
# `Jenkinsfile` is a bare filename (matched by basename, see
# _matches_entry below).
DEFAULT_PROTECTED_PATHS: tuple[str, ...] = (
    ".github/workflows/",
    ".gitlab-ci/",
    ".git/hooks/",
    ".circleci/",
    "Jenkinsfile",
)


def protected_paths() -> tuple[str, ...]:
    """The baseline PLUS whatever `safety.protected_paths_extra` (Section
    21) adds via `AMOP_PROTECTED_PATHS_EXTRA` (comma-separated, matching
    the established AMOP_* convention -- circuit_breakers.py,
    concurrency.py). Extension only: there is no parameter or code path
    here that can remove a baseline entry, by construction rather than
    by a runtime check.
    """
    extra = tuple(
        p.strip()
        for p in os.environ.get("AMOP_PROTECTED_PATHS_EXTRA", "").split(",")
        if p.strip()
    )
    return DEFAULT_PROTECTED_PATHS + extra


def _matches_entry(relative: str, entry: str) -> bool:
    if entry.endswith("/"):
        prefix = entry.rstrip("/")
        return relative == prefix or relative.startswith(prefix + "/")
    # A bare filename (Jenkinsfile): matches by basename anywhere in the
    # tree, not just at repo root. A nested subdir/Jenkinsfile is still
    # a real CI/CD config file -- the conservative reading protects more
    # surface, which is the right default for a hardcoded safety floor.
    return relative == entry or relative.endswith("/" + entry)


def protected_path_match(tool: ToolSpec, args: dict, ctx: ToolContext) -> bool:
    """True if this call's path argument targets a protected path.

    Deliberately checked on the SAME resolved path `path_restricted()`
    already computes (same traversal/symlink handling via
    `resolve_within_scratch`, not reimplemented) -- a call that already
    failed `path_restricted()` returns False here (nothing to add; it's
    already denied), not an error.
    """
    path_arg = args.get("path")
    if path_arg is None:
        return False
    resolved = resolve_within_scratch(path_arg, ctx.scratch_dir)
    if resolved is None:
        return False
    try:
        relative = resolved.relative_to(Path(ctx.scratch_dir).resolve()).as_posix()
    except ValueError:
        return False
    return any(_matches_entry(relative, entry) for entry in protected_paths())
