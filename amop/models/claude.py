from amop.models.base import BaseLLM, ModelResponse


class ClaudeProvider(BaseLLM):
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        raise NotImplementedError("ClaudeProvider not implemented yet")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("ClaudeProvider not implemented yet")
