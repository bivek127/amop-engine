"""WatcherAgent — spec Section 6.1.

Purpose: convert raw signal into a bounded set of candidate anomalies.
"Not responsible for diagnosing anything -- it flags, it doesn't
explain." Milestone 9 scopes the signal source down to a single,
concretely available one: a configured GitHub repo's open issues (no
production app is being monitored yet, so log/metric sources from the
full spec's tool list -- search_logs, get_metrics, list_recent_errors --
are out of scope this milestone).

Permissions: ADR-08 -- Watcher runs at `observer` mode always, hardcoded,
never elevated. The agent touching the least-verified input (raw,
externally-authored issue text) should have the smallest blast radius by
construction. This is enforced by the caller (orchestrator/watch.py)
constructing `ToolContext(mode="observer", ...)`, never by anything in
this file -- same "code enforces it, the prompt doesn't" rule as every
other permission boundary in this project.

Why `tools = ("__none__",)` and not `tools = ()`: agents/base.py's
_run_loop does `allowed_tools=self.tools or None` -- an EMPTY tuple is
falsy in Python, so `() or None` evaluates to `None`, which
tools/registry.py's invoke_tool() reads as "no restriction," the
opposite of what an empty tuple looks like it should mean. Every other
agent in this codebase has a non-empty tools tuple, so this gotcha has
never been exercised before Watcher. `("__none__",)` is a name that can
never match a real registered tool, so any attempted tool_call
deterministically hits TOOL_NOT_PERMITTED rather than silently
succeeding on a non-mutating tool (the observer-mode Safety Engine check
would separately deny any mutating one, but that's a second, redundant
backstop, not the primary defense here).

Why Watcher never calls a tool itself despite spec 6.1 listing
list_open_issues as one of its tools: loop_limit=1 (Section 6.1: "Watcher
makes one classification pass per poll cycle; it does not investigate")
mechanically means exactly one model turn -- one JSON response, either a
tool_call OR a final_answer, never both (agents/base.py's _run_loop).
Calling list_open_issues itself and then emitting an AnomalyAlert would
need two turns. Resolution: the orchestrator (cli/main.py's `watch`
command) pre-fetches the issue list via invoke_tool() directly --
matching the existing orchestrator-initiated-tool-call precedent in
orchestrator/chain.py (_get_diff, _run_tests_ground_truth) -- and embeds
it into Watcher's prompt as data. Watcher's one turn is then pure
classification: given this batch of issues, which look like substantive
bug reports worth flagging, and how severe? Real judgment, matching its
spec'd "low reasoning, pattern-match against thresholds" model tier, just
never expressed as a tool call.
"""

from amop.agents.base import RESPONSE_FORMAT_INSTRUCTIONS, BaseAgent, render_tool_catalog
from amop.agents.handoffs import WatcherReport


class WatcherAgent(BaseAgent):
    name = "watcher"
    tools = ("__none__",)
    loop_limit = 1
    handoff_schema = WatcherReport

    def system_prompt(self) -> str:
        return (
            "You are a triage classifier for a software project's GitHub "
            "issue tracker. You will be given a numbered batch of "
            "currently-open issues (e.g. 'Issue 1', 'Issue 2', ...), "
            "each with its title and body. Your job is to flag which "
            "ones look like substantive, real bug reports worth "
            "investigating -- not to diagnose or fix anything.\n\n"
            "For each issue that looks like a real bug report (not a "
            "question, not a documentation typo, not a feature request), "
            "emit one AnomalyAlert with:\n"
            "  - issue_index: the number of that issue in the batch "
            "(e.g. 3 for 'Issue 3') -- NOT its real GitHub issue number, "
            "just its position in this prompt\n"
            "  - severity: how disruptive this bug sounds (low/medium/"
            "high/critical) -- e.g. a crash or data-loss report is "
            "high/critical, a cosmetic issue is low\n"
            "  - summary: one plain-language sentence describing the "
            "problem, under 280 characters\n"
            "  - confidence: how sure you are this is a genuine, "
            "actionable bug report (0.0-1.0)\n"
            "  - evidence: cite the issue text that led you to this "
            "conclusion (type: \"event\", ref: a short quote or "
            "reference)\n\n"
            "Issues that are questions, discussion, or clearly not bugs "
            "should NOT get an AnomalyAlert -- it is correct and normal "
            "for your answer to contain fewer alerts than issues in the "
            "batch, including zero.\n\n"
            "You have no tools -- you are given everything you need in "
            "this prompt. Do not attempt to call a tool.\n\n"
            f"Available tools:\n{render_tool_catalog(self.tools)}\n\n"
            f"{RESPONSE_FORMAT_INSTRUCTIONS}\n\n"
            f"{self.handoff_instructions()}"
        )
