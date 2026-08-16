"""In-memory task storage."""

from exceptions import TaskNotFoundError
from models import Task


class InMemoryTaskStore:
    """A simple dict-backed store -- no persistence, good enough for
    tests and small scripts."""

    def __init__(self):
        self._tasks: dict[str, Task] = {}

    def add(self, task: Task) -> None:
        self._tasks[task.id] = task

    def get(self, task_id: str) -> Task:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFoundError(task_id) from None

    def update(self, task: Task) -> None:
        if task.id not in self._tasks:
            raise TaskNotFoundError(task.id)
        self._tasks[task.id] = task

    def delete(self, task_id: str) -> None:
        try:
            del self._tasks[task_id]
        except KeyError:
            raise TaskNotFoundError(task_id) from None

    def list(self) -> list[Task]:
        return list(self._tasks.values())

    def count(self) -> int:
        return len(self._tasks)
