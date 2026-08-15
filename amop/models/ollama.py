import httpx

from amop.models.base import BaseLLM, ModelResponse

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:14b"


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

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("OllamaProvider.embed is not implemented yet")
