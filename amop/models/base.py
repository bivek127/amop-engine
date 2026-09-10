from abc import ABC, abstractmethod

from pydantic import BaseModel


class ModelResponse(BaseModel):
    """One completion's result, normalized across providers.

    Cache-token fields (Design Decision D-16, spec 11.3.1): frontier
    providers bill prompt-cache reads at a fraction of the base input
    rate, so collapsing them into a single `input_tokens` count "can
    overestimate real spend by an order of magnitude." That matters
    beyond a cosmetic dashboard number -- it feeds Section 12.3's
    max_cost_per_day_usd breaker, which would then trip and drop the
    whole platform into global observer mode over money that was never
    actually spent.

    D-16 puts the responsibility for surfacing these distinctly on each
    provider implementation rather than on shared cost logic, which is
    why they live here on the shared response shape but are parsed
    per-provider. Both default to None: providers that don't report
    caching (Ollama, which is local and free) simply leave them unset,
    and every pre-existing caller is unaffected.
    """

    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    # Input tokens billed at the cache-WRITE rate (typically a premium
    # over base input) and the cache-READ rate (typically ~10x cheaper).
    # `input_tokens` stays the uncached count so the three never
    # double-count each other.
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
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
