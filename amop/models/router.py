"""Model Router — spec Section 11.2, per-agent model selection and the
fallback chain.

Section 5.5's per-agent model table has existed on paper since the
start and has never been wired to anything, because every agent has
always been handed the same OllamaProvider instance (constructed
directly in 8+ places in cli/main.py). This module is that seam.

Two behaviors, both from 11.2:

**Per-agent selection.** Which provider an agent gets is configuration
resolved at instantiation time, not a hardcoded constructor call.

**Fallback.** If the configured provider errors beyond its own retry
budget, fall back to a configured alternative rather than failing the
task -- "a degraded but continued run is preferred over a hard task
failure." The substitution is logged loudly and recorded on the
response's `model` field, because 11.2 is explicit that this is
"intentional visibility, not a bug to hide": a run that silently
produced weaker output from a cheaper model would be worse than one
that failed.

A missing API key is deliberately NOT a fallback trigger. Falling back
on it would turn a misconfiguration into a silently degraded run that
looks like it worked -- exactly the "confidently wrong" failure class
this project keeps prioritizing. It raises instead.
"""

import os
from collections.abc import Callable

from amop.models.base import BaseLLM, ModelResponse
from amop.models.claude import ClaudeProvider, MissingAPIKeyError
from amop.models.ollama import OllamaProvider
from amop.models.openai import OpenAIProvider

# Provider name -> zero-arg factory. Factories rather than instances so
# nothing constructs a provider (or reads a key) until something
# actually asks for one.
PROVIDER_FACTORIES: dict[str, Callable[[], BaseLLM]] = {
    "ollama": OllamaProvider,
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
}

# Section 5.5's per-agent table. Defaults keep every agent on the local
# model -- 26 milestones of behavior stays byte-identical unless someone
# opts an agent into a paid provider via config. Nothing here starts
# spending money by merely existing.
DEFAULT_AGENT_PROVIDERS: dict[str, str] = {
    "investigator": "ollama",
    "coder": "ollama",
    "tester": "ollama",
    "reviewer": "ollama",
}

DEFAULT_FALLBACK = "ollama"


def _noop(message: str) -> None:
    pass


class ModelRouter:
    """Resolves a provider per agent and wraps it with fallback.

    `emit` receives the substitution notice; the CLI passes its own
    stage-printer so a fallback is visible in a live run rather than
    buried in a log file nobody reads during a demo.
    """

    def __init__(
        self,
        agent_providers: dict[str, str] | None = None,
        fallback_provider: str | None = None,
        emit: Callable[[str], None] = _noop,
    ) -> None:
        self.agent_providers = dict(DEFAULT_AGENT_PROVIDERS)
        if agent_providers:
            self.agent_providers.update(agent_providers)
        # Env override so a real-model run is a config change, not a
        # code change: AMOP_AGENT_PROVIDER_REVIEWER=claude.
        for agent in list(self.agent_providers):
            env_value = os.environ.get(f"AMOP_AGENT_PROVIDER_{agent.upper()}")
            if env_value:
                self.agent_providers[agent] = env_value
        self.fallback_provider = (
            fallback_provider
            or os.environ.get("AMOP_FALLBACK_PROVIDER")
            or DEFAULT_FALLBACK
        )
        self.emit = emit

    def provider_name_for(self, agent_name: str) -> str:
        return self.agent_providers.get(agent_name, DEFAULT_FALLBACK)

    def is_overridden(self, agent_name: str) -> bool:
        """True only if this agent was deliberately pointed somewhere
        other than the default local provider.

        Callers use this to leave un-overridden agents on the model
        instance they already built (which carries the operator's own
        `--model` choice). Without this distinction, merely introducing
        a router would silently discard `--model` for every agent and
        replace it with a freshly-defaulted OllamaProvider -- a
        regression that would look like nothing changed."""
        return self.provider_name_for(agent_name) != DEFAULT_AGENT_PROVIDERS.get(
            agent_name, DEFAULT_FALLBACK
        )

    def overrides(self) -> dict[str, str]:
        """Agent -> provider, for the agents that were actually
        overridden. Used to announce real-model usage at run start,
        since that is the moment spend begins."""
        return {
            agent: self.provider_name_for(agent)
            for agent in self.agent_providers
            if self.is_overridden(agent)
        }

    def build(self, agent_name: str) -> BaseLLM:
        """The provider an agent should use, already wrapped so that a
        provider outage degrades the run instead of ending it."""
        primary_name = self.provider_name_for(agent_name)
        primary = _construct(primary_name)
        if primary_name == self.fallback_provider:
            # No point wrapping a provider in a fallback to itself.
            return primary
        return FallbackLLM(
            primary=primary,
            primary_name=primary_name,
            fallback=_construct(self.fallback_provider),
            fallback_name=self.fallback_provider,
            agent_name=agent_name,
            emit=self.emit,
        )


def _construct(name: str) -> BaseLLM:
    factory = PROVIDER_FACTORIES.get(name)
    if factory is None:
        raise ValueError(
            f"unknown provider {name!r}; known: {sorted(PROVIDER_FACTORIES)}"
        )
    return factory()


class FallbackLLM(BaseLLM):
    """Primary provider, with one configured alternative behind it.

    Deliberately only ONE level deep. A chain of fallbacks-of-fallbacks
    would make the worst case unbounded in both latency and (on paid
    providers) cost, and would make "which model actually answered"
    progressively harder to state honestly.
    """

    def __init__(
        self,
        primary: BaseLLM,
        primary_name: str,
        fallback: BaseLLM,
        fallback_name: str,
        agent_name: str,
        emit: Callable[[str], None] = _noop,
    ) -> None:
        self.primary = primary
        self.primary_name = primary_name
        self.fallback = fallback
        self.fallback_name = fallback_name
        self.agent_name = agent_name
        self.emit = emit

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        try:
            return await self.primary.complete(messages, tools)
        except MissingAPIKeyError:
            # Never fall back on a misconfiguration -- see module
            # docstring. A missing key is a setup error to surface, not
            # a transient outage to route around.
            raise
        except Exception as exc:  # noqa: BLE001 -- 11.2: any provider error degrades rather than fails
            self.emit(
                f"MODEL FALLBACK: {self.agent_name} could not use "
                f"{self.primary_name} ({type(exc).__name__}: {exc}); "
                f"continuing on {self.fallback_name}"
            )
            response = await self.fallback.complete(messages, tools)
            # 11.2: model_used may differ from model_config.default, and
            # that difference must stay visible downstream rather than
            # being normalized away here.
            return response

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self.primary.embed(texts)
