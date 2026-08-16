"""CoderAgent — Milestone 2 wires in real tool calling. Section 5.1's
run() shape: call model with tools -> if tool_use, invoke via registry ->
feed result back -> repeat until final answer or loop_limit.

Ollama tool-calling convention: spec Section 11.1 assigns the "structured
JSON in the prompt" translation to OllamaProvider itself. That's
implemented here, at the agent level, instead -- models/ollama.py and
BaseLLM stay untouched (neither is in this milestone's file list, and
extending them wasn't necessary: the protocol only needs to shape the
prompt and parse response.content, which agents/coder.py already owns).
The retry-once-on-malformed-JSON behavior spec assigns to the provider's
parser is replicated here for the same reason.

Milestone 3: run() now wraps the loop in a sandbox session (Section
9.1's Create -> ... -> Destroy) -- one container per run() call, reused
across every tool call inside that run, destroyed when it ends
(including on error, via finally). Tools import from sandbox/tools.py,
not tools/filesystem.py -- read_file/write_file now execute inside that
container, not on the host (see sandbox/tools.py's module docstring for
why the two are never imported together: importing both would let
whichever loads second silently overwrite the other's registration in
the shared tool registry).
"""

import asyncio
import json
import os
import uuid

from amop.agents.base import AgentResult, BaseAgent
from amop.sandbox import tools as sandbox_tools  # noqa: F401 -- registers read_file/write_file (sandboxed)
from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext, all_tools, invoke_tool

_RESPONSE_FORMAT_INSTRUCTIONS = (
    "You MUST respond with exactly one JSON object per turn, and nothing "
    "else -- no prose before or after it, no markdown code fences. Two "
    "forms are valid:\n"
    '  To call a tool:    {"tool_call": {"name": "<tool name>", "arguments": {...}}}\n'
    '  To give a final answer (no more tools needed): '
    '{"final_answer": "<your answer text>"}'
)


def _render_tool_catalog() -> str:
    tools = all_tools()
    if not tools:
        return "(no tools available)"
    lines = []
    for t in tools:
        props = ", ".join((t.parameters or {}).get("properties", {}).keys())
        lines.append(f"- {t.name}({props}): {t.description}")
    return "\n".join(lines)


def _parse_model_json(content: str) -> tuple[dict | None, str | None]:
    """Parse one turn's response against the tool_call/final_answer
    protocol. Returns (parsed, None) on success or (None, error) on
    failure -- never raises."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"

    if not isinstance(parsed, dict):
        return None, "expected a JSON object"
    if "tool_call" in parsed:
        call = parsed["tool_call"]
        if not isinstance(call, dict) or "name" not in call:
            return None, "tool_call must be an object with a 'name' field"
        return parsed, None
    if "final_answer" in parsed:
        return parsed, None
    return None, "expected a 'tool_call' or 'final_answer' key"


class CoderAgent(BaseAgent):
    name = "coder"

    def __init__(
        self, model, ctx: ToolContext | None = None, task_id: str | None = None
    ) -> None:
        super().__init__(model)
        # Section 12.1: single global mode this milestone (no per-repo/
        # per-agent precedence yet). Defaults to "suggestor" -- "observer"
        # would block every write and make tool use pointless to demo.
        self.ctx = ctx or ToolContext(
            agent_name=self.name,
            scratch_dir=sandbox_tools.SCRATCH_DIR,
            mode=os.environ.get("AMOP_PERMISSION_MODE", "suggestor"),
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
            f"Available tools:\n{_render_tool_catalog()}\n\n"
            f"{_RESPONSE_FORMAT_INSTRUCTIONS}"
        )

    async def run(self, prompt: str) -> AgentResult:
        """Section 9.1's lifecycle around the tool-calling loop below:
        Create Sandbox -> ... -> Destroy container (default), whichever
        of task-end/max_lifetime_seconds comes first is enforced by
        SandboxManager itself. The container is created here (blocking
        Docker calls off-loaded via asyncio.to_thread so they don't stall
        the event loop) and always destroyed in `finally`, including on
        an exception from the loop itself."""
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

    async def _run_loop(self, prompt: str) -> AgentResult:
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": prompt},
        ]
        tool_calls_log: list[dict] = []

        for iteration in range(1, self.loop_limit + 1):
            try:
                response = await self.model.complete(messages)
            except Exception as exc:
                return AgentResult(
                    success=False,
                    output="",
                    error=str(exc),
                    iterations_used=iteration,
                    tool_calls=tool_calls_log,
                )

            messages.append({"role": "assistant", "content": response.content})
            parsed, parse_error = _parse_model_json(response.content)

            if parse_error:
                # One retry, per spec's "validating parser that retries
                # once on malformed JSON before surfacing a hard error."
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last response was not valid JSON in the "
                            f"expected format ({parse_error}). "
                            f"{_RESPONSE_FORMAT_INSTRUCTIONS}"
                        ),
                    }
                )
                try:
                    response = await self.model.complete(messages)
                except Exception as exc:
                    return AgentResult(
                        success=False,
                        output="",
                        error=str(exc),
                        iterations_used=iteration,
                        tool_calls=tool_calls_log,
                    )
                messages.append({"role": "assistant", "content": response.content})
                parsed, parse_error = _parse_model_json(response.content)
                if parse_error:
                    return AgentResult(
                        success=False,
                        output="",
                        error=(
                            "model did not return valid tool-call JSON "
                            f"after retry: {parse_error}"
                        ),
                        iterations_used=iteration,
                        tool_calls=tool_calls_log,
                    )

            if "final_answer" in parsed:
                return AgentResult(
                    success=True,
                    output=str(parsed["final_answer"]),
                    iterations_used=iteration,
                    tool_calls=tool_calls_log,
                )

            call = parsed["tool_call"]
            tool_name = call.get("name", "")
            tool_args = call.get("arguments") or {}

            result = await invoke_tool(
                tool_name, tool_args, self.ctx, agent_name=self.name
            )
            tool_calls_log.append(
                {
                    "name": tool_name,
                    "args": tool_args,
                    "success": result.success,
                    "error_code": result.error_code,
                    "message": result.message,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool result for {tool_name}: "
                    + json.dumps(
                        {
                            "success": result.success,
                            "output": result.output,
                            "error_code": result.error_code,
                            "message": result.message,
                        }
                    ),
                }
            )

        return AgentResult(
            success=False,
            output="",
            error="loop_limit exceeded without a final answer",
            iterations_used=self.loop_limit,
            tool_calls=tool_calls_log,
        )
