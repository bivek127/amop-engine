import uuid

import pytest

from amop.agents.coder import CoderAgent
from amop.models.base import BaseLLM, ModelResponse
from amop.orchestrator.task import Task, TaskState, run_task


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
async def test_full_loop_task_created_agent_runs_task_done():
    task = Task(id=str(uuid.uuid4()), prompt="write a fizzbuzz function in python")
    assert task.state == TaskState.CREATED

    agent = CoderAgent(model=FakeLLM())
    result_task = await run_task(task, agent)

    assert result_task.state == TaskState.DONE
    assert result_task.output
    assert result_task.error is None
