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
            "already been run against it and PASSED. Decide whether the "
            "change should be approved.\n\n"
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
            "touches files unrelated to the root cause.\n\n"
            "EVIDENCE REQUIREMENT FOR ANY REJECTION -- this is the part "
            "reviewers get wrong most often, so follow it exactly:\n"
            "  - A passing test suite plus a diff confined to the affected "
            "files is real evidence the change works. It does not "
            "automatically mean the change is right, but it means a "
            "rejection needs to point at something specific the tests "
            "didn't catch -- not just a restated doubt.\n"
            "  - If you reject, `rejection_reason` and at least one entry "
            "in `findings` MUST quote the exact code you are objecting to, "
            "copied verbatim from the diff (e.g. `return (urgency * "
            "URGENCY_WEIGHT + impact * IMPACT_WEIGHT) / (1 + effort)`), "
            "together with its file and line number, plus a concrete "
            "statement of what is wrong with THAT code and, where "
            "possible, what it should say instead.\n"
            "  - Generic phrases with no quoted code -- 'does not address "
            "the stated root cause', 'introduces a new formula that does "
            "not align with the bug report', 'alters behavior in an "
            "unexpected way', 'does not explain the reported symptom' -- "
            "are NOT acceptable as a rejection on their own. If that is "
            "all you can say, you have not actually located a problem in "
            "the diff, and the correct verdict is approved=true, not a "
            "rejection you cannot back up with the code itself.\n"
            "  - 'A different formula/approach than I would have written' "
            "is not a defect. Multiple different diffs can each correctly "
            "fix the same root cause; judge the one in front of you against "
            "the bug report and root cause, not against an approach you "
            "imagined instead.\n"
            "  - If tests pass and the diff is confined to the affected "
            "files, you MUST also fill in `counterexample_claim`: a "
            "STRUCTURED, EXECUTABLE counterexample (see its schema "
            "description below for the exact format), plus the prose "
            "`counterexample` for a human to read.\n"
            "  - Your `counterexample_claim` IS ACTUALLY EXECUTED against "
            "the real code before your verdict is applied. Both the "
            "`expect` values you state and the `claim` relation you "
            "assert are compared with what the code really returns. A "
            "claim that does not hold up is detected and your rejection "
            "is converted to an approval. Stating something plausible "
            "but untrue is strictly worse than stating nothing -- it will "
            "be caught either way, and a true claim is the only thing "
            "that makes a rejection stick.\n"
            "  - Do the arithmetic before you write it down. If you claim "
            "one input outranks another, compute both values and check "
            "that the comparison you are asserting is actually true of "
            "those numbers.\n"
            "  - This is mechanically enforced -- a rejection of a "
            "passing, in-scope diff whose claim is absent, un-runnable, "
            "or contradicted by the code is automatically converted to an "
            "approval before a human ever sees it, regardless of what you "
            "set `approved` to. If you genuinely cannot construct a true "
            "one, the diff is not demonstrably wrong and the correct "
            "verdict is approved=true.\n\n"
            "Every finding must be specific and actionable -- the Coder "
            "receives them verbatim and has to act on them.\n\n"
            "You cannot edit files. Your job is the verdict, not the fix.\n\n"
            f"Available tools:\n{render_tool_catalog(self.tools)}\n\n"
            f"{RESPONSE_FORMAT_INSTRUCTIONS}\n\n"
            f"{self.handoff_instructions()}"
        )
