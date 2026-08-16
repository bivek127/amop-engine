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
