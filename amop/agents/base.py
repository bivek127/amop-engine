from abc import ABC, abstractmethod

from pydantic import BaseModel

from amop.models.base import BaseLLM


class AgentResult(BaseModel):
    success: bool
    output: str
    error: str | None = None
    iterations_used: int


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
