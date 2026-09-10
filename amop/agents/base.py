"""BaseAgent — spec Section 5.1's contract, shared by every agent.

Milestone 4 lifts the reasoning loop up here. Through Milestone 3 it
lived in agents/coder.py because Coder was the only agent; with four
agents it becomes the thing that makes them agents rather than bespoke
scripts (5.1: "run() is the reasoning loop, identical in shape across
all agents").

Two Section 5.1 pieces arrive with it:

  tools           -- the subset of the Tool Registry this agent may call.
                     Enforced in registry.invoke_tool(), not merely
                     described in the system prompt: the Investigator is
                     stopped from writing files by code, the same way
                     the Safety Engine gates paths. A prompt saying
                     "don't write files" is a suggestion, not a boundary.
  handoff_schema  -- the pydantic model this agent's final answer must
                     validate against (Section 4.7). None means the
                     agent emits a free-form string answer, which is
                     what CoderAgent still does (see agents/coder.py).
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError

from amop.models.base import BaseLLM
from amop.audit.actions import record_model_call
from amop.tools.registry import ToolContext, all_tools, invoke_tool

RESPONSE_FORMAT_INSTRUCTIONS = (
    "You MUST respond with exactly one JSON object per turn, and nothing "
    "else -- no prose before or after it, no markdown code fences. Two "
    "forms are valid:\n"
    '  To call a tool:    {"tool_call": {"name": "<tool name>", "arguments": {...}}}\n'
    '  To give a final answer (no more tools needed): '
    '{"final_answer": <your answer>}'
)


class AgentResult(BaseModel):
    success: bool
    output: str
    error: str | None = None
    iterations_used: int
    # Milestone 2: one entry per tool call made during run(), in order --
    # {"name":, "args":, "success":, "error_code":, "message":}. Lets a
    # caller (e.g. the CLI) show which calls were ALLOWED vs DENIED
    # without needing a persisted audit table. Empty for agents that
    # don't call tools (e.g. BaseAgent's default single-call run()).
    tool_calls: list[dict] = []
    # Milestone 4: the validated handoff object (Section 4.7) when this
    # agent declares a handoff_schema. Typed loosely because each agent
    # returns a different pydantic model; the orchestrator narrows it.
    handoff: Any = None


def render_tool_catalog(allowed: tuple[str, ...] | None) -> str:
    """Render the tool list for a system prompt. Only ever shows tools
    the agent is actually permitted to call, so the prompt can't tempt a
    model toward a call that invoke_tool() will just reject."""
    tools = [t for t in all_tools() if allowed is None or t.name in allowed]
    if not tools:
        return "(no tools available)"
    lines = []
    for t in tools:
        props = ", ".join((t.parameters or {}).get("properties", {}).keys())
        lines.append(f"- {t.name}({props}): {t.description}")
    return "\n".join(lines)


def parse_model_json(content: str) -> tuple[dict | None, str | None]:
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


class BaseAgent(ABC):
    name: str
    model: BaseLLM
    loop_limit: int = 10
    tools: tuple[str, ...] = ()
    handoff_schema: type[BaseModel] | None = None

    def __init__(self, model: BaseLLM, ctx: ToolContext | None = None) -> None:
        self.model = model
        self.ctx = ctx

    @abstractmethod
    def system_prompt(self) -> str: ...

    def build_initial_messages(self, prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": prompt},
        ]

    async def _complete_and_record(self, messages: list[dict]):
        """One model call, with its token usage written to the audit
        trail (spec 11.3) instead of thrown away.

        Providers have always captured `input_tokens`/`output_tokens`;
        until now this loop discarded them, which is why Section 20.1's
        tokens-per-task metric had nothing to aggregate and
        `estimated_cost_usd` stayed permanently NULL. Recording happens
        here, in the one place every agent's completions funnel through,
        rather than at each of the three call sites below -- a fourth
        call site added later gets the accounting for free.

        The write never raises (see audit/actions.py) and never blocks
        the completion: a run must not fail because its bookkeeping did.
        """
        started = time.monotonic()
        response = await self.model.complete(messages)
        latency_ms = int((time.monotonic() - started) * 1000)

        if self.ctx is not None:
            await record_model_call(
                self.ctx,
                agent_name=self.name,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=latency_ms,
            )
        return response

    def handoff_instructions(self) -> str:
        """Render this agent's handoff schema into prompt text. Derived
        from the pydantic model itself so the instructions can never
        drift from what validate_handoff() actually enforces."""
        if self.handoff_schema is None:
            return ""
        fields = json.dumps(self.handoff_schema.model_json_schema(), indent=2)
        return (
            "Your final_answer MUST be a JSON object matching this schema "
            f"exactly:\n{fields}\n"
            'Example shape: {"final_answer": { ...fields above... }}'
        )

    def validate_handoff(self, raw: Any) -> BaseModel:
        """Parse/validate an agent's final answer against its handoff
        schema (Section 5.1). Raises ValidationError on failure --
        Section 4.7 treats that as an agent failure, not a silent
        pass-through, and the orchestrator surfaces it as one."""
        if self.handoff_schema is None:
            raise RuntimeError(f"{self.name} declares no handoff_schema")
        if isinstance(raw, str):
            raw = json.loads(raw)
        return self.handoff_schema.model_validate(raw)

    async def run(self, prompt: str) -> AgentResult:
        return await self._run_loop(prompt)

    async def _run_loop(self, prompt: str) -> AgentResult:
        """Section 5.1's loop: call model -> if tool_use, invoke via the
        registry (gated by the Safety Engine) -> feed the result back ->
        repeat until a valid handoff, a failure, or loop_limit."""
        messages = self.build_initial_messages(prompt)
        tool_calls_log: list[dict] = []

        for iteration in range(1, self.loop_limit + 1):
            try:
                response = await self._complete_and_record(messages)
            except Exception as exc:
                return AgentResult(
                    success=False,
                    output="",
                    error=str(exc),
                    iterations_used=iteration,
                    tool_calls=tool_calls_log,
                )

            messages.append({"role": "assistant", "content": response.content})
            parsed, parse_error = parse_model_json(response.content)

            if parse_error:
                # One retry, per spec's "validating parser that retries
                # once on malformed JSON before surfacing a hard error."
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last response was not valid JSON in the "
                            f"expected format ({parse_error}). "
                            f"{RESPONSE_FORMAT_INSTRUCTIONS}"
                        ),
                    }
                )
                try:
                    response = await self._complete_and_record(messages)
                except Exception as exc:
                    return AgentResult(
                        success=False,
                        output="",
                        error=str(exc),
                        iterations_used=iteration,
                        tool_calls=tool_calls_log,
                    )
                messages.append({"role": "assistant", "content": response.content})
                parsed, parse_error = parse_model_json(response.content)
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
                answer = parsed["final_answer"]
                if self.handoff_schema is None:
                    return AgentResult(
                        success=True,
                        output=str(answer),
                        iterations_used=iteration,
                        tool_calls=tool_calls_log,
                    )
                try:
                    handoff = self.validate_handoff(answer)
                except (ValidationError, json.JSONDecodeError, TypeError) as exc:
                    # Same one-retry courtesy the malformed-JSON path
                    # gets: a schema miss from a local model is usually a
                    # fixable formatting slip, and a retry here is much
                    # cheaper than failing the whole task. If it misses
                    # twice it's a real agent failure (Section 4.7).
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your final_answer did not match the required "
                                f"schema: {exc}. {self.handoff_instructions()}"
                            ),
                        }
                    )
                    try:
                        response = await self._complete_and_record(messages)
                        messages.append(
                            {"role": "assistant", "content": response.content}
                        )
                        retry_parsed, retry_error = parse_model_json(response.content)
                        if retry_error or "final_answer" not in retry_parsed:
                            raise ValueError(retry_error or "expected a final_answer")
                        handoff = self.validate_handoff(retry_parsed["final_answer"])
                    except Exception as retry_exc:
                        return AgentResult(
                            success=False,
                            output="",
                            error=(
                                f"{self.name} handoff failed schema validation "
                                f"after retry: {retry_exc}"
                            ),
                            iterations_used=iteration,
                            tool_calls=tool_calls_log,
                        )
                return AgentResult(
                    success=True,
                    output=handoff.model_dump_json(),
                    iterations_used=iteration,
                    tool_calls=tool_calls_log,
                    handoff=handoff,
                )

            call = parsed["tool_call"]
            tool_name = call.get("name", "")
            tool_args = call.get("arguments") or {}

            result = await invoke_tool(
                tool_name,
                tool_args,
                self.ctx,
                agent_name=self.name,
                allowed_tools=self.tools or None,
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
