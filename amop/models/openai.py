"""OpenAIProvider — spec Section 11.1, real Chat Completions API with
function calling.

Written and tested against mocked responses only: this project has an
Anthropic key and an Anthropic spend cap, not an OpenAI one, so nothing
here has been pointed at the real API. That is a deliberate limit on
financial surface, not an oversight -- and it is stated plainly rather
than left for someone to discover when their first real call fails.

The value of writing it anyway is that OpenAI's function-calling shape
is genuinely different from Anthropic's `tool_use` blocks:

  - tools nest under {"type": "function", "function": {...}}, where
    Anthropic takes a flat {name, description, input_schema}
  - the schema key is `parameters`, not `input_schema`
  - calls come back on message.tool_calls[], not as content blocks
  - arguments arrive as a JSON *string* that must be parsed, where
    Anthropic gives an already-structured object
  - the system prompt is an ordinary role="system" message rather than
    a top-level parameter
  - usage keys are prompt_tokens/completion_tokens, not
    input_tokens/output_tokens

Having both means the shared layer above is provably provider-neutral
rather than Anthropic's shape wearing a thin coat of paint.
"""

import json
import os

import httpx

from amop.models.base import BaseLLM, ModelResponse
from amop.models.claude import (
    MAX_TRANSIENT_RETRIES,
    RETRY_STATUS_CODES,
    MissingAPIKeyError,
)

DEFAULT_MODEL = "gpt-4o"
API_URL = "https://api.openai.com/v1/chat/completions"


def _tools_to_openai(tools: list[dict]) -> list[dict]:
    """Map the project's ToolSpec-shaped dicts to OpenAI's `tools`.

    Same one-general-mapping rule as the Anthropic side -- no per-tool
    special cases -- but a visibly different target shape (see this
    module's docstring)."""
    mapped = []
    for tool in tools:
        mapped.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return mapped


def _content_from_message(message: dict) -> str:
    """Normalize one OpenAI message into the project's JSON-in-content
    protocol, exactly as the Anthropic provider does -- so the agent loop
    cannot tell which provider answered.

    OpenAI hands back tool-call arguments as a JSON *string*, so it gets
    parsed here. A malformed one is not silently swallowed: it falls
    through to `{}` arguments, which the Safety Engine and the tool's own
    schema validation will then reject loudly, rather than being
    reshaped into something that looks valid.
    """
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        call = tool_calls[0]
        function = call.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        return json.dumps(
            {"tool_call": {"name": function.get("name", ""), "arguments": arguments}}
        )
    return message.get("content") or ""


class OpenAIProvider(BaseLLM):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        if not self.api_key:
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is not set. Note that this provider has never "
                "been exercised against the real API -- see this module's "
                "docstring."
            )

        payload: dict = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = _tools_to_openai(tools)

        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

        data = await self._post_with_bounded_retry(payload, headers)

        choices = data.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        usage = data.get("usage") or {}
        # OpenAI reports cached input under prompt_tokens_details; absent
        # on models or accounts without prompt caching, hence the .get
        # chain rather than an assumed key (D-16 still applies: it must
        # be surfaced distinctly, not folded into prompt_tokens).
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")

        return ModelResponse(
            content=_content_from_message(message),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            cache_read_input_tokens=cached,
            model=data.get("model") or self.model,
        )

    async def _post_with_bounded_retry(self, payload: dict, headers: dict) -> dict:
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for _attempt in range(MAX_TRANSIENT_RETRIES + 1):
                try:
                    response = await client.post(API_URL, json=payload, headers=headers)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status not in RETRY_STATUS_CODES:
                        raise RuntimeError(
                            f"OpenAI API error ({status}): {exc.response.text[:500]}"
                        ) from exc
                    last_error = exc
                except httpx.RequestError as exc:
                    last_error = exc

        raise RuntimeError(
            f"OpenAI API unreachable after {MAX_TRANSIENT_RETRIES + 1} "
            f"attempts: {last_error}"
        ) from last_error

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """OpenAI does have an embeddings API, but codebase_intel is
        committed to one embedding model across indexing and query time
        (D-6, Ollama's nomic-embed-text). Mixing embedding models would
        put vectors from two different spaces in the same pgvector
        column, where cosine similarity between them is meaningless --
        so this raises rather than offering a foot-gun."""
        raise NotImplementedError(
            "Embeddings deliberately stay on OllamaProvider (D-6): mixing "
            "embedding models would put incomparable vectors in one index."
        )
