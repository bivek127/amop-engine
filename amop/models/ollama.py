import httpx

from amop.models.base import BaseLLM, ModelResponse

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:14b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# Milestone 8: real breaking point, measured live against a running Ollama
# with the real pyinvoke/invoke chunk corpus (1,127 chunks, 619,482 total
# chars) -- not assumed from documentation, since Ollama's own /api/embed
# docs state no aggregate batch-size limit at all. Empirical result: ITEM
# COUNT is the dominant crash driver, not aggregate char count --
# 1,000 small items (49K total chars) crashed Ollama's tokenizer
# subprocess; 30 huge items (122K total chars, ~4K chars/item) did not;
# 612 naturally-sized items (420K total chars) succeeded, 618 failed.
# Binary search narrowed the real boundary to 612 (safe) / 618 (fail)
# items. EMBED_MAX_ITEMS_PER_REQUEST is set to roughly half that boundary
# for a real safety margin. EMBED_MAX_CHARS_PER_REQUEST is a secondary,
# defensive cap (item count already dominates in practice) kept well
# under the 420K-char point that was confirmed safe at high item count.
EMBED_MAX_ITEMS_PER_REQUEST = 300
EMBED_MAX_CHARS_PER_REQUEST = 200_000


class OllamaProvider(BaseLLM):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
    ) -> None:
        self.model = model
        self.base_url = base_url

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        url = f"{self.base_url}/api/chat"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Is Ollama running? Try `ollama serve`."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama returned an error ({exc.response.status_code}): "
                f"{exc.response.text}"
            ) from exc

        data = response.json()
        message = data.get("message", {})
        content = message.get("content", "")

        return ModelResponse(
            content=content,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            model=self.model,
        )

    async def embed(
        self, texts: list[str], model: str = DEFAULT_EMBED_MODEL
    ) -> list[list[float]]:
        """Section 7.5: code and prose share the same embedding model
        (Design Decision D-6) -- `model` defaults to the one model
        codebase_intel/embeddings.py uses for both indexing and query
        time, but stays overridable rather than hardcoded, since embed()
        is BaseLLM's general-purpose method, not codebase_intel-specific.

        Milestone 8: NOT one HTTP call regardless of len(texts) -- that
        claim was true at Milestone 5's fixture scale (15 chunks) and
        false at real-repo scale (Milestone 7: 1,127 chunks in one
        request crashed Ollama's own tokenizer subprocess). texts is
        split into ordered sub-batches via batch_by_budget() and sent as
        one sequential POST per sub-batch, then concatenated back into a
        single list matching the original order -- callers (indexer.py's
        zip(all_chunks, embeddings, strict=True)) depend on getting back
        exactly len(texts) embeddings in the same order, regardless of
        how many requests that took under the hood.
        """
        url = f"{self.base_url}/api/embed"
        all_embeddings: list[list[float]] = []

        for batch in batch_by_budget(
            texts, EMBED_MAX_CHARS_PER_REQUEST, EMBED_MAX_ITEMS_PER_REQUEST
        ):
            payload = {"model": model, "input": batch}
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"Could not connect to Ollama at {self.base_url}. "
                    "Is Ollama running? Try `ollama serve`."
                ) from exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    raise RuntimeError(
                        f"Ollama has no model {model!r} pulled. Try "
                        f"`ollama pull {model}`."
                    ) from exc
                raise RuntimeError(
                    f"Ollama returned an error ({exc.response.status_code}): "
                    f"{exc.response.text}"
                ) from exc

            data = response.json()
            embeddings = data.get("embeddings")
            if not embeddings:
                raise RuntimeError(f"Ollama /api/embed returned no embeddings: {data}")
            all_embeddings.extend(embeddings)

        return all_embeddings


def batch_by_budget(
    texts: list[str], max_chars: int, max_items: int
) -> list[list[str]]:
    """Split `texts` into ordered sub-batches, each under `max_chars`
    combined length AND `max_items` count -- whichever limit is hit
    first closes the current batch. Public (not `_`-prefixed): meant to
    be unit-tested directly, same convention as search.py's
    reciprocal_rank_fusion.

    Order-preserving: concatenating the sub-batches back together
    reconstructs `texts` exactly, since callers (OllamaProvider.embed(),
    and transitively indexer.py's zip(..., strict=True)) depend on a
    strict 1:1 positional correspondence between input texts and output
    embeddings.

    A single text that alone exceeds `max_chars` still gets emitted as
    its own singleton batch -- never dropped, never a zero-progress
    infinite loop. (Milestone 8's indexer.py-level truncation is meant to
    keep this case rare, but this function makes no assumption that
    upstream truncation happened -- it has to be correct on its own.)
    """
    if not texts:
        return []

    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0

    for text in texts:
        text_len = len(text)
        would_exceed_chars = current and current_chars + text_len > max_chars
        would_exceed_items = current and len(current) >= max_items
        if would_exceed_chars or would_exceed_items:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += text_len

    if current:
        batches.append(current)

    return batches
