"""CoderAgent — spec Section 6.3. Given a root-cause plan, produce a
minimal, correct, in-scope diff.

History: Milestone 2 gave this agent the first real tool-calling loop;
Milestone 3 moved its tools into a container; Milestone 4 lifted the
loop itself into BaseAgent (agents/base.py) now that four agents share
it. What stays here is what's genuinely Coder-specific: its prompt, its
tool allowlist, and the standalone sandbox lifecycle.

Ollama tool-calling convention: spec Section 11.1 assigns the "structured
JSON in the prompt" translation to OllamaProvider itself. That's
implemented at the agent level instead -- models/ollama.py and BaseLLM
stay untouched, since the protocol only needs to shape the prompt and
parse response.content.

No handoff_schema (deliberate). Section 6.3.9 says the Reviewer "is not
allowed to trust Coder's self-report alone", so the orchestrator builds
CodeChangeReport from git ground truth (branch, sha, changed files) via
sandbox/repo.py rather than from whatever this agent claims it did. The
model's final answer stays free-form prose, which is also why Milestone
0 and 2's tests still pass against it unchanged.
"""

import asyncio
import os
import uuid

from amop.agents.base import (
    RESPONSE_FORMAT_INSTRUCTIONS,
    AgentResult,
    BaseAgent,
    render_tool_catalog,
)
from amop.codebase_intel import search as search_tool  # noqa: F401 -- registers search_code
from amop.sandbox import tools as sandbox_tools  # noqa: F401 -- registers the sandboxed tools
from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext


class CoderAgent(BaseAgent):
    name = "coder"
    tools = ("read_file", "write_file", "patch_file", "run_tests", "search_code")
    loop_limit = 10

    def __init__(
        self, model, ctx: ToolContext | None = None, task_id: str | None = None
    ) -> None:
        # Section 12.1: single global mode this milestone (no per-repo/
        # per-agent precedence yet). Defaults to "suggestor" -- "observer"
        # would block every write and make tool use pointless to demo.
        super().__init__(
            model,
            ctx
            or ToolContext(
                agent_name=self.name,
                scratch_dir=sandbox_tools.SCRATCH_DIR,
                mode=os.environ.get("AMOP_PERMISSION_MODE", "suggestor"),
            ),
        )
        # Sandbox container label (Section 9.7.1) -- not a persisted Task
        # id necessarily; callers that have one (cli/main.py) should pass
        # it, everything else gets a fresh one per agent instance.
        self._task_id = task_id or uuid.uuid4().hex
        self._sandbox_manager: SandboxManager | None = None
        self.last_container_id: str | None = None

    def system_prompt(self) -> str:
        return (
            "You are a code assistant with access to tools for reading and "
            "writing files inside a scratch workspace. Given a description "
            "of a change, use the tools available to make it happen, then "
            "give a final answer summarizing what you did.\n\n"
            "If your task already names specific affected file(s), start "
            "by calling read_file directly on those -- they came from a "
            "real investigation of this repo, don't spend a turn "
            "re-discovering what you were already told. On a repo with "
            "more than a couple of files and no known target file, use "
            "search_code with a natural-language or exact-symbol query "
            "to find the specific code to change, rather than reading "
            "files one by one, then read_file on the candidates it "
            "returns before editing. If search_code doesn't return "
            "anything relevant, that is not a reason to give up -- fall "
            "back to read_file on any affected file paths you were "
            "given before concluding there's nothing to change.\n\n"
            "If a plain read_file (no start_line/end_line) on a large "
            "file comes back showing only its first ~60 lines with a "
            "[NOTE: ...] at the top -- that is expected, not the whole "
            "file and not an error. Don't conclude anything about the "
            "rest of the file from what you were shown. Use search_code "
            "to find the specific lines you actually need, then read_file "
            "with start_line/end_line to see them.\n\n"
            "To make an edit: prefer patch_file over write_file for any "
            "EXISTING file -- it applies a unified diff to just the "
            "lines you're changing, so you never need the whole file in "
            "context, only the region you're editing plus a couple of "
            "lines of surrounding context on each side. Locate that "
            "region with search_code's returned line range, or a "
            "targeted read_file(path, start_line, end_line) -- add "
            "with_line_numbers=true on that read if you need to be sure "
            "of the exact starting line for the diff's '@@' header (the "
            "'N: ' prefix that adds is for your reference only -- never "
            "put it in the actual diff body, that has to match the "
            "file's real content exactly). Reserve write_file for a "
            "brand new file, or a genuinely trivial single-line change. "
            "If patch_file comes back PATCH_CONFLICT, the diff didn't "
            "apply cleanly -- re-read the current region (it may have "
            "changed) and retry with a fresh diff; never resubmit the "
            "same one unchanged. Do NOT fall back to write_file after a "
            "PATCH_CONFLICT unless you actually have the file's full, "
            "correct content to write -- writing back only the small "
            "fragment you last read would destroy the rest of the file. "
            "If write_file itself comes back SUSPICIOUS_SHRINK, that "
            "means you just tried exactly that: re-read the file and use "
            "patch_file for a targeted change instead of retrying "
            "write_file.\n\n"
            f"Available tools:\n{render_tool_catalog(self.tools)}\n\n"
            f"{RESPONSE_FORMAT_INSTRUCTIONS}"
        )

    async def run(self, prompt: str) -> AgentResult:
        """Section 9.1's lifecycle around the shared reasoning loop.

        When the caller already supplied a sandbox (the Milestone 4
        chain does, because Section 9.1 requires one container per *task*
        shared across agents -- Tester has to see the branch Coder
        created), this just runs the loop and leaves lifecycle to the
        owner. Standalone (Milestones 0/2/3, and the CLI's single-agent
        path) it creates and destroys its own container, always in
        `finally` so an exception can't leak one.

        Milestone 30: (re)activates the forced-fresh-read gate fresh for
        THIS attempt, every time -- ctx.coder_read_tracking = {} here,
        not merely at __init__. ctx is the SAME long-lived object the
        whole chain shares (Investigator/Tester/Reviewer included), and
        this same CoderAgent instance is reused across retry attempts
        within one task (chain.py's CODING loop), each a fresh
        conversation via a fresh run() call -- a stale range tracked
        from an earlier attempt's now-irrelevant messages must never
        silently satisfy this attempt's own gate.
        """
        self.ctx.coder_read_tracking = {}
        if self.ctx.sandbox is not None:
            self.last_container_id = self.ctx.sandbox.short_id
            return await self._run_loop(prompt)

        if self._sandbox_manager is None:
            try:
                self._sandbox_manager = await asyncio.to_thread(SandboxManager)
            except Exception as exc:
                return AgentResult(
                    success=False,
                    output="",
                    error=f"sandbox unavailable: {exc}",
                    iterations_used=0,
                    tool_calls=[],
                )

        try:
            sandbox = await asyncio.to_thread(
                self._sandbox_manager.create, self._task_id, self.ctx.scratch_dir
            )
        except Exception as exc:
            return AgentResult(
                success=False,
                output="",
                error=f"sandbox unavailable: {exc}",
                iterations_used=0,
                tool_calls=[],
            )

        self.ctx.sandbox = sandbox
        self.last_container_id = sandbox.short_id
        try:
            return await self._run_loop(prompt)
        finally:
            # Milestone 31, opt-in -- same reasoning as run_fix/
            # resume_fix: this standalone path owns its scratch dir for
            # its whole lifetime (no caller supplied a sandbox, so
            # nothing outside this method could still need it), and
            # leaving it around was a real, previously-unnoticed leak --
            # every run of this exact path (Milestone 0's own real
            # standalone-agent test, and any future no-ctx caller)
            # created a fresh directory under the real default
            # SCRATCH_DIR and never removed it. Confirmed safe: the one
            # existing test on this path never reads scratch-dir content
            # after run() returns.
            await asyncio.to_thread(
                self._sandbox_manager.destroy, self._task_id, remove_scratch_dir=True
            )
            self.ctx.sandbox = None
