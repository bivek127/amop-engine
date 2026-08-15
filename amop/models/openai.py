from amop.models.base import BaseLLM, ModelResponse


class OpenAIProvider(BaseLLM):
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        raise NotImplementedError("OpenAIProvider not implemented yet")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("OpenAIProvider not implemented yet")
