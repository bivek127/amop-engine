import httpx

from amop.models.base import BaseLLM, ModelResponse

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:14b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


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

        Ollama's /api/embed accepts a batch `input` list directly, so
        this is one HTTP call regardless of len(texts), not len(texts)
        calls.
        """
        payload = {"model": model, "input": texts}
        url = f"{self.base_url}/api/embed"

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
        return embeddings
