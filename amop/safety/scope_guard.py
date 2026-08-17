"""Diff-Size Cap — spec Section 29.1 ("Must-Have Before Phase 1 Code").

This is a *different, new* mechanism from Section 6.3.9's existing
file-scope guard (git diff --stat against Coder's declared file list,
scope_justification, independently re-checked by the Reviewer -- already
implemented via orchestrator/chain.py's out_of_scope_files() /
enforce_review_checks()). Section 29.1's own words: this "extends Section
6.3.9's scope guard with a hard line-of-code ceiling" -- it doesn't
duplicate or replace the file-list check, it adds a second, independent
one on top of it.

If a task's diff exceeds coder.max_loc_per_task (default ~150 changed
lines), Coder aborts rather than attempting the oversized single-shot
edit, handing back to the orchestrator with
CodeChangeReport.status == "needs_decomposition" (agents/handoffs.py).
Full DAG-based sub-task decomposition is the correct long-term answer
(spec Section 29.6, later phase); this milestone ships the abort-and-report
half only.

Hard cap enforced in code -- not a prompt instruction to "keep changes
small" -- checked from two call sites: orchestrator/chain.py's main loop
(right after Coder's turn, before ever calling go(TESTING, ...)) and
tools/github.py's create_pull_request (as a hard gate immediately before
the real GitHub API call, per Section 12.5.1's enforcement table). Pure
function, zero I/O, directly unit-testable in isolation (Section 19.4),
same as safety/secret_scan.py and safety/blacklist.py. Not registered into
safety/engine.py -- there's no plugin/registry system for safety checks in
this codebase; each new check is called directly by its call sites.
"""

import os

DEFAULT_MAX_LOC_PER_TASK = 150

# AMOP_* env var convention (AMOP_SCRATCH_DIR, AMOP_TEST_TIMEOUT_SECONDS,
# ...). Corresponds to the spec's config-style name coder.max_loc_per_task.
MAX_LOC_PER_TASK = int(
    os.environ.get("AMOP_MAX_LOC_PER_TASK", str(DEFAULT_MAX_LOC_PER_TASK))
)


def changed_line_count(diff_text: str) -> int:
    """Count of added-or-removed lines in a unified diff -- every line
    starting with '+' or '-', excluding the '+++ b/file' / '--- a/file'
    file-header lines, which aren't content changes."""
    count = 0
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count


def over_cap(diff_text: str, max_loc: int = MAX_LOC_PER_TASK) -> bool:
    """True if `diff_text` changes more than `max_loc` lines."""
    return changed_line_count(diff_text) > max_loc
