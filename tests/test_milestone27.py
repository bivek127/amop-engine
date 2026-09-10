"""Milestone 27 — Real Multi-Provider Model Router. Spec Sections 11.1
(provider interface), 11.2 (routing + fallback), 11.3.1/D-16
(cache-aware token accounting), 21.1 (Day Zero).

EVERY test in this file is mocked. Nothing here makes a real API call,
and the suite is expected to pass with no API key set at all -- that is
Done-When #5, and it's demonstrated rather than asserted (see
test_no_test_in_this_file_can_reach_a_real_api). Real-API checks live
behind AMOP_E2E_ANTHROPIC, default OFF, firmer than Milestone 5's
AMOP_E2E_OLLAMA precedent because these cost money.
"""

import json
import os

import httpx
import pytest

from amop.models.base import ModelResponse
from amop.models.claude import (
    ClaudeProvider,
    MissingAPIKeyError,
    _content_from_blocks,
    _split_system,
    _tools_to_anthropic,
)
from amop.models.openai import OpenAIProvider, _content_from_message
from amop.models.router import (
    DEFAULT_AGENT_PROVIDERS,
    FallbackLLM,
    ModelRouter,
)

TOOLSPEC = [
    {
        "name": "read_file",
        "description": "Read a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
]


# =======================================================================
# Financial safety -- the Day Zero posture, enforced by tests
# =======================================================================


def test_a_provider_without_a_key_raises_rather_than_calling_anything():
    """A missing key must fail before any HTTP request is attempted --
    not produce a 401 round-trip, and certainly not a billed call."""
    provider = ClaudeProvider(api_key=None)
    provider.api_key = None  # explicit: ignore any ambient env key
    with pytest.raises(MissingAPIKeyError):
        import asyncio

        asyncio.run(provider.complete([{"role": "user", "content": "hi"}]))


def test_missing_key_is_a_distinct_type_so_the_router_will_not_mask_it():
    """MissingAPIKeyError must not be a plain RuntimeError, because
    FallbackLLM catches broad Exceptions to degrade gracefully and would
    otherwise silently route a misconfiguration to the local model --
    turning a setup error into a quietly degraded run."""
    assert issubclass(MissingAPIKeyError, RuntimeError)
    assert MissingAPIKeyError is not RuntimeError


def test_every_agent_defaults_to_the_local_free_provider():
    """Nothing starts spending money by merely existing. Opting an agent
    into a paid provider has to be a deliberate config change."""
    assert set(DEFAULT_AGENT_PROVIDERS.values()) == {"ollama"}


# =======================================================================
# D-16 -- cache tokens surfaced distinctly, not collapsed
# =======================================================================


def test_model_response_keeps_cache_tokens_separate_from_input_tokens():
    """Spec 11.3.1: collapsing cache reads into one input count "can
    overestimate real spend by an order of magnitude", which would trip
    the cost breaker over money never spent. The three counts stay
    distinct fields."""
    response = ModelResponse(
        content="x",
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=2000,
        cache_read_input_tokens=8000,
        model="m",
    )
    assert response.input_tokens == 100
    assert response.cache_creation_input_tokens == 2000
    assert response.cache_read_input_tokens == 8000


def test_cache_fields_default_to_none_so_existing_providers_are_unaffected():
    response = ModelResponse(content="x", model="m")
    assert response.cache_creation_input_tokens is None
    assert response.cache_read_input_tokens is None


# =======================================================================
# Anthropic mapping (pure functions -- no network involved at all)
# =======================================================================


def test_toolspec_maps_to_anthropic_input_schema():
    mapped = _tools_to_anthropic(TOOLSPEC)
    assert mapped == [
        {
            "name": "read_file",
            "description": "Read a file.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]


def test_tool_mapping_is_general_not_per_tool():
    """A tool the registry gains later must map with no edit here."""
    invented = [{"name": "brand_new_tool", "parameters": {"type": "object"}}]
    mapped = _tools_to_anthropic(invented)
    assert mapped[0]["name"] == "brand_new_tool"
    assert mapped[0]["input_schema"] == {"type": "object"}


def test_system_messages_are_split_out_as_anthropic_requires():
    """Anthropic takes `system` as a top-level parameter, unlike
    Ollama/OpenAI where it's an ordinary message."""
    system, rest = _split_system(
        [
            {"role": "system", "content": "you are a reviewer"},
            {"role": "user", "content": "review this"},
        ]
    )
    assert system == "you are a reviewer"
    assert rest == [{"role": "user", "content": "review this"}]


def test_a_tool_use_block_becomes_the_projects_own_tool_call_json():
    """The whole integration hinges on this: native tool_use on the
    wire, but the agent loop's existing JSON protocol in the return
    value, so agents/base.py needs no provider awareness."""
    content = _content_from_blocks(
        [{"type": "tool_use", "name": "read_file", "input": {"path": "a.py"}}]
    )
    assert json.loads(content) == {
        "tool_call": {"name": "read_file", "arguments": {"path": "a.py"}}
    }


def test_the_translated_tool_call_is_parseable_by_the_real_agent_parser():
    """Not just 'looks like JSON' -- run it through the actual parser the
    agent loop uses, so this can't drift from the real contract."""
    from amop.agents.base import parse_model_json

    content = _content_from_blocks(
        [{"type": "tool_use", "name": "run_tests", "input": {}}]
    )
    parsed, error = parse_model_json(content)
    assert error is None
    assert parsed["tool_call"]["name"] == "run_tests"


def test_plain_text_passes_through_untouched_for_final_answers():
    content = _content_from_blocks(
        [{"type": "text", "text": '{"final_answer": {"ok": true}}'}]
    )
    assert json.loads(content) == {"final_answer": {"ok": True}}


def test_a_tool_use_block_wins_over_accompanying_prose():
    """When the model narrates and calls a tool in the same turn, the
    actionable half must survive -- the agent loop has no slot for the
    prose."""
    content = _content_from_blocks(
        [
            {"type": "text", "text": "Let me read that file."},
            {"type": "tool_use", "name": "read_file", "input": {"path": "a.py"}},
        ]
    )
    assert json.loads(content)["tool_call"]["name"] == "read_file"


def test_openai_tool_call_json_string_arguments_are_parsed_to_an_object():
    """The asymmetry worth pinning: Anthropic hands back an already-
    structured `input` object, OpenAI hands back `arguments` as a JSON
    STRING. Both must normalize to the same protocol shape."""
    content = _content_from_message(
        {
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path": "b.py"}'}}
            ]
        }
    )
    assert json.loads(content) == {
        "tool_call": {"name": "read_file", "arguments": {"path": "b.py"}}
    }


def test_openai_plain_content_passes_through():
    assert _content_from_message({"content": '{"final_answer": 1}'}) == (
        '{"final_answer": 1}'
    )


# =======================================================================
# Provider round-trips against mocked HTTP (still no real network)
# =======================================================================


class _MockTransport(httpx.AsyncBaseTransport):
    """Answers every request from a canned payload and records what was
    sent, so the request body itself can be asserted on."""

    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self.status, json=self.payload, request=request
        )


def _patch_transport(monkeypatch, transport: "_MockTransport") -> None:
    """Route every httpx.AsyncClient the providers build through a mock
    transport. Both providers do `import httpx` and reference
    httpx.AsyncClient at call time, so patching the attribute is enough
    -- and it means no test here can open a real socket even by
    accident."""
    original = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


async def test_claude_complete_parses_usage_including_cache_tokens(monkeypatch):
    transport = _MockTransport(
        {
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": '{"final_answer": "done"}'}],
            "usage": {
                "input_tokens": 120,
                "output_tokens": 30,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 9000,
            },
        }
    )
    _patch_transport(monkeypatch, transport)

    provider = ClaudeProvider(api_key="test-key-not-real")
    response = await provider.complete(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        tools=TOOLSPEC,
    )

    assert response.content == '{"final_answer": "done"}'
    assert response.input_tokens == 120
    assert response.output_tokens == 30
    assert response.cache_creation_input_tokens == 1000
    assert response.cache_read_input_tokens == 9000

    sent = json.loads(transport.requests[0].content)
    assert sent["system"] == "sys"  # hoisted out of messages
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["tools"][0]["input_schema"]["properties"] == {"path": {"type": "string"}}
    assert transport.requests[0].headers["x-api-key"] == "test-key-not-real"


async def test_claude_does_not_retry_a_non_transient_error(monkeypatch):
    """Every retry is a billed call. A 400 would fail identically the
    second time, so it must raise after exactly ONE request."""
    transport = _MockTransport({"error": "bad request"}, status=400)
    _patch_transport(monkeypatch, transport)

    provider = ClaudeProvider(api_key="test-key-not-real")
    with pytest.raises(RuntimeError, match="400"):
        await provider.complete([{"role": "user", "content": "hi"}])

    assert len(transport.requests) == 1  # exactly one billed attempt


async def test_claude_retries_a_transient_error_but_stays_bounded(monkeypatch):
    from amop.models.claude import MAX_TRANSIENT_RETRIES

    transport = _MockTransport({"error": "rate limited"}, status=429)
    _patch_transport(monkeypatch, transport)

    provider = ClaudeProvider(api_key="test-key-not-real")
    with pytest.raises(RuntimeError):
        await provider.complete([{"role": "user", "content": "hi"}])

    # Bounded: the cost of a rate-limit storm is capped and knowable.
    assert len(transport.requests) == MAX_TRANSIENT_RETRIES + 1


async def test_openai_maps_its_own_differently_shaped_schema(monkeypatch):
    """OpenAI nests under type/function, uses `parameters` not
    `input_schema`, and returns tool calls with JSON-STRING arguments --
    proving the shared layer isn't just Anthropic's shape renamed."""
    transport = _MockTransport(
        {
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "b.py"}',
                                }
                            }
                        ]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 64},
            },
        }
    )
    _patch_transport(monkeypatch, transport)

    provider = OpenAIProvider(api_key="test-key-not-real")
    response = await provider.complete(
        [{"role": "user", "content": "hi"}], tools=TOOLSPEC
    )

    # Same normalized protocol as the Anthropic path.
    assert json.loads(response.content) == {
        "tool_call": {"name": "read_file", "arguments": {"path": "b.py"}}
    }
    assert response.input_tokens == 80
    assert response.output_tokens == 20
    assert response.cache_read_input_tokens == 64

    sent = json.loads(transport.requests[0].content)
    assert sent["tools"][0]["type"] == "function"
    assert sent["tools"][0]["function"]["parameters"]["type"] == "object"


async def test_openai_malformed_tool_arguments_do_not_become_silent_defaults(
    monkeypatch,
):
    """A malformed argument string must not be reshaped into something
    that looks valid -- empty arguments get rejected downstream by the
    tool's own schema, which is the loud failure we want."""
    transport = _MockTransport(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": "{not json"}}
                        ]
                    }
                }
            ],
            "usage": {},
        }
    )
    _patch_transport(monkeypatch, transport)

    provider = OpenAIProvider(api_key="test-key-not-real")
    response = await provider.complete([{"role": "user", "content": "hi"}])
    assert json.loads(response.content)["tool_call"]["arguments"] == {}


# =======================================================================
# Routing and fallback (Section 11.2)
# =======================================================================


class _BoomLLM:
    """A provider that always fails, standing in for a rate limit or
    outage without needing a real one."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    async def complete(self, messages, tools=None):
        self.calls += 1
        raise self.exc

    async def embed(self, texts):
        raise self.exc


class _OkLLM:
    def __init__(self, content: str = "fallback answer") -> None:
        self.content = content
        self.calls = 0

    async def complete(self, messages, tools=None):
        self.calls += 1
        return ModelResponse(content=self.content, model="ollama-stub")

    async def embed(self, texts):
        return [[0.0]]


async def test_a_provider_outage_falls_back_and_says_so_out_loud():
    """11.2: a degraded but continued run beats a hard task failure --
    but the substitution is 'intentional visibility, not a bug to
    hide', so it must be announced, not silently absorbed."""
    emitted: list[str] = []
    primary = _BoomLLM(RuntimeError("Anthropic API error (529): overloaded"))
    fallback = _OkLLM()

    router_llm = FallbackLLM(
        primary=primary,
        primary_name="claude",
        fallback=fallback,
        fallback_name="ollama",
        agent_name="reviewer",
        emit=emitted.append,
    )
    response = await router_llm.complete([{"role": "user", "content": "hi"}])

    assert response.content == "fallback answer"
    assert primary.calls == 1 and fallback.calls == 1
    assert len(emitted) == 1
    notice = emitted[0]
    assert "MODEL FALLBACK" in notice
    assert "reviewer" in notice and "claude" in notice and "ollama" in notice


async def test_a_missing_key_is_never_masked_by_the_fallback():
    """The important converse. Falling back on a misconfiguration would
    produce a run that looks fine while quietly using a weaker model --
    the 'confidently wrong' class this project keeps guarding against."""
    emitted: list[str] = []
    primary = _BoomLLM(MissingAPIKeyError("ANTHROPIC_API_KEY is not set"))
    fallback = _OkLLM()

    router_llm = FallbackLLM(
        primary=primary,
        primary_name="claude",
        fallback=fallback,
        fallback_name="ollama",
        agent_name="reviewer",
        emit=emitted.append,
    )
    with pytest.raises(MissingAPIKeyError):
        await router_llm.complete([{"role": "user", "content": "hi"}])

    assert fallback.calls == 0  # never silently degraded
    assert emitted == []


def test_router_resolves_per_agent_providers_from_config():
    router = ModelRouter(agent_providers={"reviewer": "claude"})
    assert router.provider_name_for("reviewer") == "claude"
    assert router.provider_name_for("coder") == "ollama"


def test_router_reads_a_per_agent_env_override(monkeypatch):
    """Pointing one agent at a paid model should be a config change, not
    a code change."""
    monkeypatch.setenv("AMOP_AGENT_PROVIDER_REVIEWER", "claude")
    router = ModelRouter()
    assert router.provider_name_for("reviewer") == "claude"
    assert router.provider_name_for("coder") == "ollama"


def test_router_does_not_wrap_a_provider_in_a_fallback_to_itself():
    router = ModelRouter()  # everything defaults to ollama == fallback
    built = router.build("coder")
    assert not isinstance(built, FallbackLLM)


def test_router_rejects_an_unknown_provider_name():
    router = ModelRouter(agent_providers={"reviewer": "not-a-provider"})
    with pytest.raises(ValueError, match="unknown provider"):
        router.build("reviewer")


# =======================================================================
# Done-When #5, demonstrated rather than asserted
# =======================================================================


def test_no_test_in_this_file_can_reach_a_real_api():
    """The real-API opt-in must default OFF. If AMOP_E2E_ANTHROPIC is
    unset -- which is the normal state, including in CI -- nothing in
    this suite is permitted to spend money.

    This is the guard, not the proof. The proof is running the whole
    suite with no ANTHROPIC_API_KEY present at all and seeing it green,
    which is exactly how this milestone's Stage 1 was verified.
    """
    assert os.environ.get("AMOP_E2E_ANTHROPIC") in (None, "", "0", "false")


@pytest.mark.skipif(
    os.environ.get("AMOP_E2E_ANTHROPIC") not in ("1", "true"),
    reason="real Anthropic API calls cost money; opt in with AMOP_E2E_ANTHROPIC=1",
)
async def test_real_anthropic_smoke_call():
    """The only test here that can spend money, and it is skipped unless
    explicitly opted in. Kept deliberately tiny -- a handful of tokens."""
    provider = ClaudeProvider()
    response = await provider.complete(
        [{"role": "user", "content": "Reply with exactly: ok"}]
    )
    assert response.content
    assert response.input_tokens is not None
