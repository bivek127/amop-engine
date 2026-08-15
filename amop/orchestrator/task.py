from enum import Enum

from pydantic import BaseModel

from amop.agents.base import BaseAgent


class TaskState(str, Enum):
    CREATED = "CREATED"
    CODING = "CODING"
    DONE = "DONE"
    FAILED = "FAILED"


class Task(BaseModel):
    id: str  # uuid
    prompt: str
    state: TaskState = TaskState.CREATED
    output: str | None = None
    error: str | None = None


async def run_task(task: Task, agent: BaseAgent) -> Task:
    """CREATED -> CODING -> (run agent) -> DONE or FAILED"""
    task.state = TaskState.CODING

    result = await agent.run(task.prompt)

    if result.success:
        task.state = TaskState.DONE
        task.output = result.output
    else:
        task.state = TaskState.FAILED
        task.error = result.error

    return task
