from abc import ABC, abstractmethod

from pydantic import BaseModel


class ModelResponse(BaseModel):
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str


class BaseLLM(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        """Send messages to the model and get a response."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings. Stub is fine for now."""
        ...
