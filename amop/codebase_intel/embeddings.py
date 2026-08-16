"""Embedding generation — spec Section 7.5, Design Decision D-6: code
and prose share the same embedding model rather than separate specialized
encoders ("v1 optimizes for one fewer moving part over marginal
retrieval-quality gains").

A thin module-level wrapper rather than a class: indexing (indexer.py)
and query-time search (search.py) both call the exact same function with
the exact same default model, which is what actually satisfies D-6 --
there's no separate code-path where the two could drift to different
models.
"""

from amop.models.base import BaseLLM
from amop.models.ollama import OllamaProvider

# A fresh OllamaProvider per call would still work (it's a thin HTTP
# client wrapper, not a stateful connection), but a module-level default
# avoids constructing one repeatedly across a single indexing run's many
# calls. Note: OllamaProvider's own `model` constructor arg selects the
# *chat* model (used by complete()) -- embed() takes its own `model`
# param, defaulted to DEFAULT_EMBED_MODEL in ollama.py, and that default
# is what actually governs embedding calls here regardless of how this
# instance was constructed.
_default_provider = OllamaProvider()


async def embed_texts(
    texts: list[str], provider: BaseLLM | None = None
) -> list[list[float]]:
    """Embed a batch of texts with the shared code/prose model.
    `provider` is overridable (tests can inject a fake BaseLLM), but
    defaults to the real Ollama-backed one -- so indexer.py and
    search.py never have to duplicate model selection."""
    if not texts:
        return []
    llm = provider or _default_provider
    return await llm.embed(texts)
