"""ReporterAgent — spec Section 6.8. Summarization and notification
only; never touches code, never opens PRs.

That guarantee is structural, not prompted. `tools = ("__none__",)`
means every tool call this agent could attempt is refused by
invoke_tool()'s allowlist check before it reaches any tool body -- the
same construction WatcherAgent uses, and for the same reason: "the agent
is instructed not to X" is a suggestion, not a safety property
(CLAUDE.md's own rule). An agent that cannot name a permitted tool
cannot edit a file no matter what a prompt injection talks it into.

Why `("__none__",)` and not `()`: agents/base.py's `_run_loop` passes
`allowed_tools=self.tools or None`, and an empty tuple is falsy in
Python -- `() or None` is None, which invoke_tool() reads as "no
restriction at all", the exact opposite of the intent. A name that can
never match a registered tool gets the intended deny. (Same trap
documented in agents/watcher.py.)

Section 6.8 also lists `send_telegram_message` and a `write_file`
scoped to WEEKLY_REPORT.md. Both are deliberately out of scope this
milestone (CLAUDE.md's "What NOT to Build" names the Telegram bot
explicitly), so Reporter has no write path of any kind -- the report is
returned to the caller, which prints it.

Loop limit 3, per spec -- generous for an agent that gets everything in
its prompt and needs exactly one turn to answer.
"""

from amop.agents.base import RESPONSE_FORMAT_INSTRUCTIONS, BaseAgent, render_tool_catalog
from amop.agents.handoffs import ReportSummary


class ReporterAgent(BaseAgent):
    name = "reporter"
    tools = ("__none__",)
    loop_limit = 3
    handoff_schema = ReportSummary

    def system_prompt(self) -> str:
        return (
            "You are a reporting assistant for an autonomous software "
            "maintenance system. You are given a window of activity -- "
            "verified counts and a list of the tasks that ran -- and you "
            "write the short human-readable summary of it.\n\n"
            "The counts you are given are already verified against the "
            "database. Echo them back exactly as given; do not "
            "recalculate them, do not adjust them, and do not infer "
            "different numbers from the task list. They will be checked "
            "against the database again after you answer, and yours "
            "will be replaced if they disagree -- so there is nothing to "
            "gain by changing them.\n\n"
            "Your actual job is `top_issues`: reading the task list, say "
            "what a human should know about this window. Recurring "
            "failures, notable fixes, anything that looks like a pattern "
            "rather than a one-off. Be specific and concrete -- name the "
            "actual bugs and files involved rather than writing "
            "generalities. If nothing notable happened, say that plainly "
            "in one bullet rather than padding the list. Never invent a "
            "task that is not in the list you were shown.\n\n"
            "You have no tools -- everything you need is in this prompt. "
            "Do not attempt to call a tool.\n\n"
            f"Available tools:\n{render_tool_catalog(self.tools)}\n\n"
            f"{RESPONSE_FORMAT_INSTRUCTIONS}\n\n"
            f"{self.handoff_instructions()}"
        )
