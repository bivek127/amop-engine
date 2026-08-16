"""Command Blacklist — spec Section 12.4, Design Decision D-9.

"the blacklist is enforced as a static regex/AST-pattern check at the
tool layer (12.2), before the sandbox ever receives the string... it is
not config-removable below the hardcoded baseline -- the baseline ships
in code, not YAML."

No shell-executing tool exists yet (run_shell is a later milestone) --
this module is built now, standalone and directly testable, so it's
ready the moment one is added (Safety Engine's blacklist_match() wires
it in, see safety/engine.py).

Deliberately NOT covered here (documented gap, not an oversight):
secret-value matching (spec's "any string containing a known secret
value, cross-checked against 9.4's redaction set") depends on Section
9.4's sandbox secret-injection/redaction set, which doesn't exist without
a sandbox (Milestone 3+).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class BlacklistMatch:
    pattern_name: str
    reason: str


_RM_RF_RE = re.compile(
    r"\brm\s+"
    r"(-\w*r\w*f\w*\b"  # -rf, -Rf, etc (order r-then-f, extra flags allowed)
    r"|-\w*f\w*r\w*\b"  # -fr, -Fr, etc (order f-then-r)
    r"|-[rRf]\s+-[rRf]\b"  # -r -f / -f -r as separate short flags
    r"|--recursive\s+--force\b"
    r"|--force\s+--recursive\b)"
)

_FORCE_PUSH_RE = re.compile(r"\bgit\s+push\b.*(--force\b|(?<!\S)-f(?!\S))")
_FORCE_WITH_LEASE_RE = re.compile(r"--force-with-lease\b")

_SUDO_RE = re.compile(r"\bsudo\b")

_CHMOD_777_RE = re.compile(r"\bchmod\s+777\b")

_CHOWN_RE = re.compile(r"\bchown\b\s+\S+\s+(\S+)")

_CURL_WGET_PIPE_SHELL_RE = re.compile(r"\b(curl|wget)\b.*\|\s*\b(sh|bash|zsh)\b")

_DD_RE = re.compile(r"\bdd\b")

_MKFS_RE = re.compile(r"\bmkfs\b")

# Classic `:(){ :|:& };:` fork bomb, whitespace-tolerant.
_FORK_BOMB_RE = re.compile(r":\s*\(\s*\)\s*\{[^}]*:\s*\|\s*:&?[^}]*\}\s*;\s*:")


def _check_force_push(command: str) -> bool:
    if not _FORCE_PUSH_RE.search(command):
        return False
    # --force-with-lease is the spec-endorsed safe alternative (Section
    # 8.4/8.5.1) -- must not be caught by the bare --force pattern above.
    return not _FORCE_WITH_LEASE_RE.search(command)


def _check_unsafe_chown(command: str) -> bool:
    match = _CHOWN_RE.search(command)
    if not match:
        return False
    target = match.group(1)
    return not target.startswith("/workspace")


# (pattern_name, reason, checker) -- the hardcoded baseline. Order matters
# only for which reason string surfaces first when multiple match.
_BASELINE: list[tuple[str, str, Callable[[str], bool]]] = [
    ("rm_rf", "recursive force delete (rm -rf)", lambda c: bool(_RM_RF_RE.search(c))),
    ("force_push", "force push to a remote (git push --force)", _check_force_push),
    ("sudo", "privilege escalation (sudo)", lambda c: bool(_SUDO_RE.search(c))),
    (
        "chmod_777",
        "world-writable permission change (chmod 777)",
        lambda c: bool(_CHMOD_777_RE.search(c)),
    ),
    (
        "chown_outside_workspace",
        "ownership change targeting a path outside /workspace",
        _check_unsafe_chown,
    ),
    (
        "curl_wget_pipe_shell",
        "remote script piped directly into a shell",
        lambda c: bool(_CURL_WGET_PIPE_SHELL_RE.search(c)),
    ),
    ("dd", "raw block-device write (dd)", lambda c: bool(_DD_RE.search(c))),
    ("mkfs", "filesystem creation (mkfs)", lambda c: bool(_MKFS_RE.search(c))),
    (
        "fork_bomb",
        "fork-bomb pattern",
        lambda c: bool(_FORK_BOMB_RE.search(c)),
    ),
]


def is_blacklisted(command: str) -> BlacklistMatch | None:
    """Check a shell command string against the hardcoded baseline.
    Returns the first matching pattern, or None if the command is clean.
    """
    for pattern_name, reason, checker in _BASELINE:
        if checker(command):
            return BlacklistMatch(pattern_name=pattern_name, reason=reason)
    return None
