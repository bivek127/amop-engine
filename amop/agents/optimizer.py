"""OptimizerAgent — spec Section 6.6. Diagnose and fix performance
regressions; structurally Investigator+Coder combined, specialized for
latency.

The one behavior that makes this agent safe to run at all is not in this
file: orchestrator/optimize.py benchmarks before and after against the
same input and reverts the change outright if the measured improvement
misses `optimizer.min_improvement_pct`. This module supplies judgment
about WHAT to change; the timer decides whether it was allowed to stand.

That split is deliberate and load-bearing. "Did my optimization help?"
is a question a model will essentially always answer yes to -- not
because any particular model is dishonest, but because it is a
measurement, and a measurement asked of a narrator is just a narrative.
Section 6.6 hangs a real, destructive action on the answer (revert), so
the answer comes from a stopwatch.

Loop limit 15 per spec -- the highest of any agent here, because
profile-then-hypothesize-then-measure is genuinely more round trips than
"read the file, change the line".
"""

from amop.agents.base import RESPONSE_FORMAT_INSTRUCTIONS, BaseAgent, render_tool_catalog
from amop.agents.handoffs import OptimizationReport
from amop.codebase_intel import search as search_tool  # noqa: F401 -- registers search_code
from amop.sandbox import tools as sandbox_tools  # noqa: F401 -- registers run_benchmark/profile_code


class OptimizerAgent(BaseAgent):
    name = "optimizer"
    tools = (
        "read_file",
        "write_file",
        "patch_file",
        "run_tests",
        "search_code",
        "run_benchmark",
        "profile_code",
    )
    loop_limit = 15
    handoff_schema = OptimizationReport

    def system_prompt(self) -> str:
        return (
            "You make slow code faster without changing what it does. "
            "The repository is checked out at /workspace.\n\n"
            "Your procedure, in order:\n"
            "  1. PROFILE FIRST. Call profile_code on the benchmark entry "
            "point before forming any hypothesis. Do not guess at a "
            "bottleneck -- a bottleneck you assumed is not a bottleneck "
            "you found, and optimizing the wrong function is worse than "
            "doing nothing because it costs a real diff for no gain.\n"
            "  2. Read the hot function the profile actually points at.\n"
            "  3. Make the smallest change that removes the cost the "
            "profile showed. Prefer patch_file for an edit to an existing "
            "file.\n"
            "  4. Run run_tests. A faster function that returns different "
            "results is not an optimization, it is a bug -- if the suite "
            "goes red, fix it or put the code back.\n\n"
            "Behavior must not change. Same inputs, same outputs, same "
            "edge cases -- only faster. Do not delete work to make a "
            "benchmark look better, do not memoize across runs to make a "
            "repeated benchmark trivially fast, and do not weaken the "
            "tests.\n\n"
            "Your before/after timings will be measured independently "
            "after you finish, against the same input. If the real "
            "improvement is too small, your change will be reverted "
            "automatically -- so report what you actually did rather "
            "than what you hoped for. An honest 'this did not help' is a "
            "useful result; an overstated one just wastes the revert.\n\n"
            f"Available tools:\n{render_tool_catalog(self.tools)}\n\n"
            f"{RESPONSE_FORMAT_INSTRUCTIONS}\n\n"
            f"{self.handoff_instructions()}"
        )
