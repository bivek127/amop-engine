"""ReviewerAgent — spec Section 6.5.

Purpose: the last automated gate before a human sees a diff. Checks the
change "independently, not by re-reading Coder's self-report" — which is
why its inputs are the real diff (via get_diff, straight from git) and
the orchestrator-verified TestReport, never Coder's own account of what
it did.

Read-only by construction: `tools` has no write_file, so a Reviewer
cannot "helpfully" fix what it objects to. Section 6.5 gives it observer
permissions for the same reason.

Milestone 4 keeps the checklist to the two items CLAUDE.md asks for
(does it look like a real fix; is it scoped to the affected files)
rather than 6.5's full five. The scope half is additionally re-checked
mechanically by the orchestrator, since "the diff only touches the files
the root cause named" is a fact about the diff, not a judgment call.
"""

from amop.agents.base import RESPONSE_FORMAT_INSTRUCTIONS, BaseAgent, render_tool_catalog
from amop.agents.handoffs import ReviewVerdict


class ReviewerAgent(BaseAgent):
    name = "reviewer"
    tools = ("read_file", "get_diff")
    loop_limit = 8
    handoff_schema = ReviewVerdict

    def system_prompt(self) -> str:
        return (
            "You are a code reviewer. A fix has been proposed for a bug in "
            "a Python repository at /workspace, and the test suite has "
            "already been run against it. Decide whether the change should "
            "be approved.\n\n"
            "Fetch the diff and review it against the stated root cause, "
            "working through these checks explicitly:\n"
            "  1. Does the ORIGINAL BUG REPORT actually match this root "
            "cause and this diff? A fix can be correct in isolation and "
            "still address a completely different problem than the one "
            "reported. If the change does not explain the reported "
            "symptom, set addresses_reported_symptom to false and reject "
            "-- even if the code change itself looks good and the tests "
            "pass.\n"
            "  2. Does the diff actually address the stated root cause -- "
            "not merely 'does it look plausible'?\n"
            "  3. Is the change scoped to the affected files, with no "
            "unrelated edits smuggled in?\n"
            "  4. Does it break anything obvious in the surrounding code, "
            "or silently change behavior the bug report didn't ask about?\n\n"
            "Approve only if the change is a real fix for the reported root "
            "cause. Reject if it is a workaround, if it edits tests to make "
            "them pass rather than fixing the code under test, or if it "
            "touches files unrelated to the root cause. When you reject, "
            "every finding must be specific and actionable -- the Coder "
            "receives them verbatim and has to act on them.\n\n"
            "You cannot edit files. Your job is the verdict, not the fix.\n\n"
            f"Available tools:\n{render_tool_catalog(self.tools)}\n\n"
            f"{RESPONSE_FORMAT_INSTRUCTIONS}\n\n"
            f"{self.handoff_instructions()}"
        )
