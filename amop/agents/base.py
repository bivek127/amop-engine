from abc import ABC, abstractmethod

from pydantic import BaseModel

from amop.models.base import BaseLLM


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


class BaseAgent(ABC):
    name: str
    model: BaseLLM
    loop_limit: int = 10

    def __init__(self, model: BaseLLM) -> None:
        self.model = model

    @abstractmethod
    def system_prompt(self) -> str: ...

    async def run(self, prompt: str) -> AgentResult:
        """
        The reasoning loop:
        1. Build messages (system prompt + user prompt)
        2. Call self.model.complete(messages)
        3. Return the response
        No tool calling yet — just text in, text out.
        """
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.model.complete(messages)
        except Exception as exc:
            return AgentResult(
                success=False,
                output="",
                error=str(exc),
                iterations_used=1,
            )

        return AgentResult(
            success=True,
            output=response.content,
            iterations_used=1,
        )
