"""ClaudeProvider — spec Section 11.1, real Anthropic Messages API.

A stub since Milestone 0. This is the first provider in the project that
costs real money per call, which drives two deliberate choices:

**Raw httpx, not the `anthropic` SDK.** The SDK is good, but it retries
internally by default, and a retry is a billed call. This milestone
promised an exact call count, and an exact count is only auditable if
every HTTP request is one this file issues explicitly. Retries here are
opt-in, bounded, and only for errors that are genuinely transient (429 /
5xx) -- never for a 4xx that would just fail again at the same cost.
It also matches OllamaProvider's existing raw-httpx style, so both
providers read the same way.

**Native tool_use on the wire, the project's JSON protocol in the
return value.** The Anthropic API is called with real `tools` and
returns real `tool_use` content blocks (what the milestone brief asks
for). Those are then translated back into the same
`{"tool_call": {"name", "arguments"}}` JSON-in-content convention
agents/base.py's parse_model_json already speaks, so the agent loop is
completely unchanged and one agent implementation works across every
provider. Provider-specific wire formats stay inside providers.

Worth noting what that translation removes: with native tool_use, the
JSON the agent loop parses is *constructed here from structured API
fields*, not hoped for from a model's free-text output. The ~1-in-5
malformed-JSON failures logged against the local model are structurally
impossible on this path -- not because the model is better, but because
the format is no longer the model's job.
"""

import json
import os

import httpx

from amop.models.base import BaseLLM, ModelResponse

DEFAULT_MODEL = "claude-sonnet-5"
API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096

# Bounded, explicit, and only for genuinely transient failures. Each
# retry is a billed call, so this is deliberately small and never
# applies to 4xx (other than 429), which would fail identically twice.
MAX_TRANSIENT_RETRIES = 2
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class MissingAPIKeyError(RuntimeError):
    """Raised when no API key is configured. Deliberately its own type:
    the router (models/router.py) treats it as "this provider is not
    usable at all" rather than as a transient error worth failing over
    for, since falling back on a missing key would silently mask a
    misconfiguration."""


def _tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Map the project's ToolSpec-shaped dicts to Anthropic's `tools`.

    One general mapping over the shared shape, never a per-tool special
    case -- a tool the registry gains later must work here with no edit.
    The registry's `parameters` is already a JSON Schema object, which is
    exactly what `input_schema` wants, so this is a rename plus a
    guaranteed-present schema rather than a translation.
    """
    mapped = []
    for tool in tools:
        mapped.append(
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return mapped


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Anthropic takes the system prompt as a top-level parameter, not as
    a message with role="system" (unlike Ollama/OpenAI). Pull any system
    messages out and join them; leave the rest in order."""
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    return ("\n\n".join(system_parts) if system_parts else None), rest


def _content_from_blocks(blocks: list[dict]) -> str:
    """Turn Anthropic's content blocks into the single string the agent
    loop expects.

    A `tool_use` block becomes the project's own tool-call JSON, so
    parse_model_json sees exactly what it sees from Ollama. Plain text
    passes through untouched -- when the model is giving its final
    answer it will already be emitting the `{"final_answer": ...}` JSON
    the shared system prompt asked for.

    If both appear, the tool call wins: it is the actionable half, and
    any accompanying prose is the model narrating what it is about to
    do, which the agent loop has no slot for.
    """
    for block in blocks:
        if block.get("type") == "tool_use":
            return json.dumps(
                {
                    "tool_call": {
                        "name": block.get("name", ""),
                        "arguments": block.get("input") or {},
                    }
                }
            )
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


class ClaudeProvider(BaseLLM):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        # Read at construction, not per call, so a run can't start
        # against a key that disappears halfway through.
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        if not self.api_key:
            raise MissingAPIKeyError(
                "ANTHROPIC_API_KEY is not set. Add it to .env (which is "
                "gitignored) or the environment before using ClaudeProvider."
            )

        system, chat_messages = _split_system(messages)
        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": chat_messages,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = _tools_to_anthropic(tools)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

        data = await self._post_with_bounded_retry(payload, headers)

        usage = data.get("usage") or {}
        return ModelResponse(
            content=_content_from_blocks(data.get("content") or []),
            # D-16: these three are kept distinct rather than summed --
            # see ModelResponse's docstring for why collapsing them
            # misprices a run badly enough to trip a cost breaker.
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
            cache_read_input_tokens=usage.get("cache_read_input_tokens"),
            model=data.get("model") or self.model,
        )

    async def _post_with_bounded_retry(self, payload: dict, headers: dict) -> dict:
        """Every attempt here is a billed call, so the loop is explicit
        and small. Non-retryable statuses raise immediately rather than
        paying repeatedly for the same rejection."""
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(MAX_TRANSIENT_RETRIES + 1):
                try:
                    response = await client.post(API_URL, json=payload, headers=headers)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status not in RETRY_STATUS_CODES:
                        raise RuntimeError(
                            f"Anthropic API error ({status}): {exc.response.text[:500]}"
                        ) from exc
                    last_error = exc
                except httpx.RequestError as exc:  # connect/read/timeout
                    last_error = exc

        raise RuntimeError(
            f"Anthropic API unreachable after {MAX_TRANSIENT_RETRIES + 1} "
            f"attempts: {last_error}"
        ) from last_error

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Anthropic has no embeddings endpoint. This stays an honest
        raise rather than a silent fallback: codebase_intel already
        embeds via Ollama's nomic-embed-text (Design Decision D-6), and
        quietly redirecting to a different provider here would hide
        which model actually produced a vector."""
        raise NotImplementedError(
            "Anthropic exposes no embeddings API -- use OllamaProvider.embed() "
            "(codebase_intel/embeddings.py already does)."
        )
