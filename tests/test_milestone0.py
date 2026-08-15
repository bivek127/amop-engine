import pytest

from amop.agents.coder import CoderAgent
from amop.models.base import BaseLLM, ModelResponse

# NOTE: this test originally drove the in-memory 3-state stub Task/
# TaskState/run_task from orchestrator/task.py. Milestone 1 replaced that
# module with real Postgres persistence and the actual bug_fix state
# machine (see tests/test_milestone1.py for that coverage) — the stub
# types no longer exist. What Milestone 0 actually guaranteed — a mocked
# LLM feeding CoderAgent produces a real AgentResult — is unchanged, since
# agents/ and models/ were not touched by Milestone 1. Re-pointed at that
# directly rather than through the now-retired orchestrator API.


class FakeLLM(BaseLLM):
    """Mocked BaseLLM — no real network calls to Ollama."""

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        return ModelResponse(content="def fizzbuzz(): ...", model="fake-model")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_coder_agent_runs_against_mocked_llm():
    agent = CoderAgent(model=FakeLLM())

    result = await agent.run("write a fizzbuzz function in python")

    assert result.success
    assert result.output
    assert result.error is None
