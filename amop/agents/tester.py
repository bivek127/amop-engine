"""TesterAgent — spec Section 6.4.

Purpose: independent test execution, "separated from Coder so a buggy
fix can't also write a test that hides its own bug unreviewed."

Important: this agent does NOT get to decide whether the tests passed.
The orchestrator runs the suite itself and overwrites
`TestReport.all_passed` with the parsed pytest result before acting on
it (see orchestrator/chain.py). This agent's real contribution is
`details` -- interpreting what failed and why, which is genuinely useful
to the next Coder iteration.

That split is the project's core rule applied at its sharpest point: a
model asked "did the tests pass?" can answer wrong, from optimism or
from misreading output, and every downstream gate would inherit the
error. pytest's exit status is not a matter of opinion, so it isn't
treated as one.

Section 6.4's regression-authoring behavior (writing a failing test
before the fix, requires_reproduction) is out of scope this milestone --
hence no write_file in `tools`.

Milestone 6, Section 29.1: the flaky-test double-check. Before trusting a
failing test as real signal, the orchestrator re-runs that same failing
test against the base branch (pre-fix commit) -- if it fails there too,
it's excluded as environmental noise rather than fed back into Coder's
retry loop. The two functions below are the PURE decision logic for that
check: given which tests failed on the fix branch and which of those also
failed on the base branch, decide what to exclude. They contain no I/O --
no git checkout, no re-running pytest -- so they're directly unit-testable
with hand-built inputs. The actual checkout/re-run/restore cycle is glue
code that lives in orchestrator/chain.py (see _rerun_failing_tests_against_base),
matching this file's existing pattern where ground truth is computed
outside the model's control, never inside Tester's own tool-call loop.

Section 29.1's critical carve-out (v2.3): a test is NEVER eligible to be
marked flaky if it's (a) a test Tester itself just wrote as a regression
test (TestReport.new_tests_added), or (b) a test named in the original bug
report. "Fails on base branch" is the WHOLE POINT for both of those cases
-- a fresh regression test is *supposed* to fail pre-fix (Section 6.4's
"red before green" check, via a throwaway git stash/checkout), and a test
the bug report itself named failing on both branches means the bug is
real, not that the test is noise. Applying the double-check without this
exclusion would silently defeat the check's entire purpose: it would let
the system classify the exact bug Coder is trying to fix as
"environmental noise" and quietly drop it from the feedback loop.

Honest limitation: "named in the original bug report" has no structured
object to check against here -- there's no AnomalyAlert, Watcher isn't
built (CLAUDE.md's "What NOT to Build"), so the only thing resembling "the
bug report" is the free-text `description` string threaded through from
cli/main.py's --description flag. is_carveout_protected() below does a
plain substring match against it. Documented gap, not an oversight -- see
ROADMAP.md's Known Limitations sections for the project's established tone
on this kind of thing.
"""

from amop.agents.base import RESPONSE_FORMAT_INSTRUCTIONS, BaseAgent, render_tool_catalog
from amop.agents.handoffs import TestReport


class TesterAgent(BaseAgent):
    name = "tester"
    tools = ("read_file", "run_tests")
    loop_limit = 10
    handoff_schema = TestReport

    def system_prompt(self) -> str:
        return (
            "You are a test analyst for a Python repository checked out at "
            "/workspace. Run the test suite and report what you observe.\n\n"
            "Run the tests, and if anything fails, read the relevant source "
            "or test file so your summary explains the failure rather than "
            "just restating it. Your 'details' field should tell the next "
            "engineer what is broken and where.\n\n"
            "Report honestly. If tests fail, say so plainly -- a report that "
            "claims success when tests failed is worse than useless. (The "
            "orchestrator independently verifies the pass/fail result "
            "against the actual test runner output.)\n\n"
            f"Available tools:\n{render_tool_catalog(self.tools)}\n\n"
            f"{RESPONSE_FORMAT_INSTRUCTIONS}\n\n"
            f"{self.handoff_instructions()}"
        )


def is_carveout_protected(
    test_name: str, description: str, new_tests_added: list[str]
) -> bool:
    """True if this test must NEVER be excluded as flaky, even if it
    fails on both branches -- Section 29.1's critical carve-out (v2.3).

    A test is protected if it's one Tester itself just authored as a
    regression test (`test_name` appears in `new_tests_added`), or if its
    name appears in the free-text bug-report `description` -- the only
    proxy available for "named in the original AnomalyAlert/CI failure
    report" (see module docstring for why there's no structured object to
    check against instead).
    """
    if test_name in new_tests_added:
        return True
    return bool(description) and test_name in description


def classify_failures(
    fix_branch_failures: list[str],
    base_branch_results: dict[str, bool],
    description: str,
    new_tests_added: list[str],
) -> tuple[list[str], list[str]]:
    """Pure decision function -- no I/O, no re-running anything.

    `fix_branch_failures`: test names that failed when run against the fix
    branch (the ground truth this milestone already computes).
    `base_branch_results`: test_name -> True if that same test *also*
    failed when re-run against the base/pre-fix branch. A name absent from
    this dict means the re-run was inconclusive (e.g. an error, or it was
    never attempted because the test is carve-out-protected) -- treated as
    "not conclusively flaky", per the fail-safe default below.

    Returns (excluded_as_flaky, effective_failures). A carve-out-protected
    test is never even looked up in `base_branch_results` -- it always
    lands in `effective_failures`, matching Section 6.4's "must NEVER be
    routed through the double-check" for a Tester's own new regression
    test.
    """
    excluded: list[str] = []
    effective: list[str] = []
    for name in fix_branch_failures:
        if is_carveout_protected(name, description, new_tests_added):
            effective.append(name)
            continue
        if base_branch_results.get(name) is True:
            excluded.append(name)
        else:
            # Fail-safe default: a test with no conclusive base-branch
            # result stays real signal, never silently discounted.
            effective.append(name)
    return excluded, effective
