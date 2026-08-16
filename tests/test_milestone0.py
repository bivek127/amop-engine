import json

import pytest

from amop.agents.coder import CoderAgent
from amop.models.base import BaseLLM, ModelResponse

# NOTE: this test originally drove the in-memory 3-state stub Task/
# TaskState/run_task from orchestrator/task.py. Milestone 1 replaced that
# module with real Postgres persistence and the actual bug_fix state
# machine (see tests/test_milestone1.py for that coverage) — the stub
# types no longer exist. Milestone 2 then changed CoderAgent.run()'s
# contract again: it's a real tool-calling loop now, speaking a
# {"tool_call": ...} / {"final_answer": ...} JSON protocol instead of
# returning a single raw text completion (see tests/test_milestone2.py
# for that coverage). What Milestone 0 actually guaranteed — a mocked LLM
# feeding CoderAgent produces a real AgentResult — is still what this
# tests; FakeLLM just has to speak the current protocol to do that.


class FakeLLM(BaseLLM):
    """Mocked BaseLLM — no real network calls to Ollama."""

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        return ModelResponse(
            content=json.dumps({"final_answer": "def fizzbuzz(): ..."}),
            model="fake-model",
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_coder_agent_runs_against_mocked_llm():
    agent = CoderAgent(model=FakeLLM())

    result = await agent.run("write a fizzbuzz function in python")

    assert result.success
    assert result.output
    assert result.error is None
    assert result.tool_calls == []
