"""InvestigatorAgent — spec Section 6.2.

Purpose: turn a bug description into a root-cause hypothesis with enough
evidence a human would find it credible. Explicitly **not** responsible
for writing the fix — that separation is the point, and it's enforced
rather than requested: `tools` omits write_file, so an Investigator that
decides to just fix the thing itself gets TOOL_NOT_PERMITTED from the
registry (agents/base.py, tools/registry.py).

Section 6.2's confidence threshold is the single highest-leverage
guardrail in the chain: "confidence < 0.6 routes to NEEDS_HUMAN_INPUT
rather than letting a low-confidence guess reach the Coder." The
threshold itself is enforced in orchestrator/chain.py, not here — this
agent only has to report its confidence honestly, and cannot route
around the consequence.

Spec tools not built yet (search_logs, git_log, git_blame,
get_error_rate) are out of scope this milestone. Milestone 5 adds
search_code (Section 7.3) -- on a repo bigger than a couple of files,
reading everything doesn't scale, which is exactly what this agent needs
Codebase Intelligence for.
"""

from amop.agents.base import RESPONSE_FORMAT_INSTRUCTIONS, BaseAgent, render_tool_catalog
from amop.agents.handoffs import RootCauseReport
from amop.codebase_intel import search as search_tool  # noqa: F401 -- registers search_code


class InvestigatorAgent(BaseAgent):
    name = "investigator"
    tools = ("read_file", "run_tests", "search_code")
    loop_limit = 10
    handoff_schema = RootCauseReport

    def system_prompt(self) -> str:
        return (
            "You are a debugging investigator. You are given a bug report "
            "about a Python repository checked out at /workspace. Your job "
            "is to find the ROOT CAUSE and explain it with evidence.\n\n"
            "You do NOT write or edit code. Another agent does that, using "
            "your report. Investigate by running the test suite and using "
            "search to find the relevant code, then reading the specific "
            "files search points you to.\n\n"
            "Work like this: run the tests to see what actually fails, then "
            "use search_code with a natural-language description of the "
            "symptom (or an exact function/error name if you have one) to "
            "find candidate code -- do not read files one by one hoping to "
            "stumble onto the right one, especially on a repo with more "
            "than a couple of files. Use read_file on the specific "
            "candidates search_code returns to see their full context. "
            "Trace the failures back to the smallest specific cause that "
            "explains them. Several failing tests often share one root "
            "cause -- prefer that single cause over listing symptoms.\n\n"
            "About the confidence field: it must reflect how well the "
            "evidence you actually collected supports your claim. If you "
            "found a specific line of code that explains the observed "
            "failures, confidence should be high. If you could not find any "
            "code or failing test corresponding to the reported symptom, do "
            "not invent one -- report what you did and did not find, and let "
            "the confidence value reflect that genuine uncertainty. An "
            "honest low number is far more useful than a confident guess.\n\n"
            f"Available tools:\n{render_tool_catalog(self.tools)}\n\n"
            f"{RESPONSE_FORMAT_INSTRUCTIONS}\n\n"
            f"{self.handoff_instructions()}"
        )
