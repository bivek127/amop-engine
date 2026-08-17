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
    tools = ("read_file", "write_file", "run_tests", "search_code")
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
        """
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
            await asyncio.to_thread(self._sandbox_manager.destroy, self._task_id)
            self.ctx.sandbox = None
