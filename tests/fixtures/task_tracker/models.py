"""Core data types for task_tracker."""

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"


# Legal status transitions -- used by validators.validate_status_transition.
ALLOWED_TRANSITIONS = {
    Status.OPEN: {Status.IN_PROGRESS, Status.BLOCKED},
    Status.IN_PROGRESS: {Status.BLOCKED, Status.DONE, Status.OPEN},
    Status.BLOCKED: {Status.IN_PROGRESS, Status.OPEN},
    Status.DONE: set(),
}


@dataclass
class Task:
    """One task in the tracker.

    urgency/impact/effort are 1-10 scales used by priority.compute_priority_score
    to rank tasks for sorting -- see that module for what each represents.
    """

    id: str
    title: str
    description: str
    urgency: int
    impact: int
    effort: int
    status: Status = Status.OPEN
    assignee: str | None = None
    tags: list[str] = field(default_factory=list)
    comments: list[dict] = field(default_factory=list)
    due_date: str | None = None

    def is_terminal(self) -> bool:
        return self.status == Status.DONE
