"""Pre-PR Secret Scanning — spec Section 29.1 ("Must-Have Before Phase 1
Code"), Section 12.5.1's attack table ("Secret leakage into a commit/PR" ->
"Pre-PR secret scan is a hard gate blocking create_pull_request" ->
enforced at "Tool Registry middleware").

Section 12.5 already covers secret *injection* (9.4: tokens reach the
sandbox as container env vars, never as tool arguments) and command-level
*redaction* (12.4: run_shell's blacklist cross-checks against known secret
values). Neither of those stops a Coder from *reading* a real secret via a
legitimate tool call and then *writing* it back out in generated code --
that's the gap this module closes, immediately before create_pull_request
ever reaches the GitHub API (amop/tools/github.py).

Per CLAUDE.md's Milestone 6 scope, this is a baseline regex set -- "doesn't
need to be exhaustive" -- not gitleaks/trufflehog running in a sandboxed
container (spec's original suggestion, explicitly descoped). A hit is a
hard block, same severity as a Safety Engine DENY (safety/engine.py's
Decision.deny), never a warning fed back to the agent to "reconsider".

Mirrors safety/blacklist.py's exact shape: a hardcoded _BASELINE list of
(pattern_name, reason, compiled_regex) tuples, checked by a pure function
with zero LLM/DB/network dependency, directly unit-testable in isolation
(Section 19.4). Not config-removable below the hardcoded baseline, same
rationale as the command blacklist -- this baseline ships in code, not
YAML, and secret_scan.py is not registered into safety/engine.py's
evaluate() pipeline; it's called directly by create_pull_request as a hard
pre-flight gate (see amop/tools/github.py), since it applies to one
specific tool's diff argument, not to every tool call's args the way the
blacklist/path-restriction checks do.

Scope: only lines *added* by this diff (`+` prefix, excluding the `+++
b/file` diff header) are scanned. A secret merely shown as diff *context*,
or one being *removed*, isn't a new leak this PR introduces -- flagging
those would be noisy and wouldn't catch anything the repo doesn't already
have committed. Documented choice, not an oversight.
"""

import re
from dataclasses import dataclass

_AWS_ACCESS_KEY_RE = re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")
_AWS_SECRET_ASSIGN_RE = re.compile(
    r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}['\"]?"
)
_GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")
_GITHUB_FINE_GRAINED_TOKEN_RE = re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")
_PRIVATE_KEY_HEADER_RE = re.compile(
    r"-----BEGIN (RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
)
_SLACK_TOKEN_RE = re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")
_STRIPE_LIVE_KEY_RE = re.compile(r"\bsk_live_[A-Za-z0-9]{16,}\b")
_GOOGLE_API_KEY_RE = re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")
# Generic catch-all: any `key/secret/token/password`-shaped assignment to a
# long quoted string. Deliberately broad -- this is what catches an
# arbitrary planted test key that doesn't match a specific vendor format,
# at the cost of being the one pattern most likely to false-positive on a
# long non-secret string. Kept last so vendor-specific patterns (better
# reasons) win when both would match.
_GENERIC_SECRET_ASSIGN_RE = re.compile(
    r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*"
    r"['\"][A-Za-z0-9\-_/+=]{16,}['\"]"
)

# (pattern_name, reason, compiled_regex) -- the hardcoded baseline. Order
# matters only for which reason string surfaces first when multiple match.
_BASELINE: list[tuple[str, str, re.Pattern]] = [
    ("aws_access_key_id", "AWS access key id", _AWS_ACCESS_KEY_RE),
    ("aws_secret_access_key", "AWS secret access key assignment", _AWS_SECRET_ASSIGN_RE),
    ("github_token", "GitHub personal access / app token", _GITHUB_TOKEN_RE),
    (
        "github_fine_grained_token",
        "GitHub fine-grained personal access token",
        _GITHUB_FINE_GRAINED_TOKEN_RE,
    ),
    ("private_key", "private key material", _PRIVATE_KEY_HEADER_RE),
    ("slack_token", "Slack token", _SLACK_TOKEN_RE),
    ("stripe_live_key", "Stripe live secret key", _STRIPE_LIVE_KEY_RE),
    ("google_api_key", "Google API key", _GOOGLE_API_KEY_RE),
    (
        "generic_secret_assignment",
        "generic key/secret/token/password-shaped assignment",
        _GENERIC_SECRET_ASSIGN_RE,
    ),
]


@dataclass
class SecretMatch:
    pattern_name: str
    reason: str
    line_number: int | None = None

    # Deliberately no `matched_text` field: the whole point is this result
    # can flow into a ToolResult.message / CLI output / log line without
    # re-leaking the secret it just caught.


def _added_lines(diff_text: str) -> list[tuple[int, str]]:
    """Return (line_number_within_diff, content) for every unified-diff
    line that represents newly added content -- a leading '+' that isn't
    the '+++ b/file' header line."""
    added = []
    for i, line in enumerate(diff_text.splitlines(), start=1):
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            added.append((i, line[1:]))
    return added


def scan_for_secrets(diff_text: str) -> list[SecretMatch]:
    """Scan a unified diff's added lines against the baseline. Returns
    every match found (not just the first), so a caller can report the
    full set of hits, e.g. to a human reviewing why a PR was blocked."""
    matches: list[SecretMatch] = []
    for line_number, content in _added_lines(diff_text):
        for pattern_name, reason, pattern in _BASELINE:
            if pattern.search(content):
                matches.append(
                    SecretMatch(
                        pattern_name=pattern_name,
                        reason=reason,
                        line_number=line_number,
                    )
                )
    return matches


# ---------------------------------------------------------------------
# Milestone 23: redaction for the agent_actions audit trail.
# ---------------------------------------------------------------------

REDACTED = "[REDACTED]"


def redact_secrets(text: str) -> str:
    """Replace anything matching the baseline secret patterns.

    Reuses `_BASELINE` -- the SAME pattern set the pre-PR gate above
    uses -- deliberately, rather than defining a second one. Two pattern
    sets drift: a pattern added for the PR gate would silently not
    protect the audit trail, and the failure would be invisible until a
    secret was already persisted.

    Different scope from `scan_for_secrets()` above, though, and the
    difference matters: that function only inspects a diff's ADDED
    lines, because a secret shown as context isn't a new leak. Here
    every byte is in scope -- `agent_actions.arguments` is arbitrary
    tool input (a file's whole contents via write_file, a diff via
    patch_file), with no diff structure to reason about and no reason to
    exempt any part of it.

    Section 12.5: audit arguments are stored with secrets redacted.
    Baseline-quality, not exhaustive -- the same honest limitation
    `_BASELINE` carries for the PR gate. It reduces what lands in the
    audit trail; it does not guarantee nothing sensitive ever does.
    """
    for _pattern_name, _reason, pattern in _BASELINE:
        text = pattern.sub(REDACTED, text)
    return text


def redact_arguments(arguments: dict | None) -> dict | None:
    """Redact secrets from a tool call's arguments, preserving shape.

    Recurses through nested dicts/lists so a secret nested inside a
    structured argument is caught too. Non-string leaves pass through
    untouched -- a secret has to be text to match a pattern, and coercing
    ints/bools to strings would corrupt the stored arguments for no gain.
    """
    if arguments is None:
        return None

    def _walk(value):
        if isinstance(value, str):
            return redact_secrets(value)
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v) for v in value]
        return value

    return _walk(arguments)
