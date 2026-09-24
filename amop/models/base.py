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
    per-provider. All three default to None: providers that don't report
    caching (Ollama, which is local and free) simply leave the two cache
    fields unset, and every pre-existing caller is unaffected.

    Milestone 32 (spec 11.3/11.3.1): `uncached_input_tokens` is the field
    name spec 11.3.1's own cost formula uses, so it's named that here
    rather than the more generic `input_tokens` this field was
    originally called -- Anthropic's `usage.input_tokens` already
    excludes cached tokens by the API's own design, and OpenAI's
    provider now subtracts `cached_tokens` out of `prompt_tokens` before
    setting this (models/openai.py), so it means the same "never
    double-counted with the two cache fields below" thing for every
    provider, not just Claude's.
    """

    content: str
    uncached_input_tokens: int | None = None
    output_tokens: int | None = None
    # Input tokens billed at the cache-WRITE rate (typically a premium
    # over base input) and the cache-READ rate (typically ~10x cheaper).
    # `uncached_input_tokens` excludes both, so the three never
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
