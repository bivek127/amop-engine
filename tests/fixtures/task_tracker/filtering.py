"""Filtering and sorting over a list of tasks."""

from models import Status
from priority import compute_priority_score


def filter_tasks(tasks: list, status: Status | None = None, assignee: str | None = None) -> list:
    """Return tasks matching the given status/assignee, if given.
    Both filters are AND'd together when both are set."""
    result = tasks
    if status is not None:
        result = [t for t in result if t.status == status]
    if assignee is not None:
        result = [t for t in result if t.assignee == assignee]
    return result


def filter_by_tag(tasks: list, tag: str) -> list:
    """Return tasks that have `tag` in their tags list."""
    return [t for t in tasks if tag in t.tags]


def sort_tasks(tasks: list, key: str = "priority") -> list:
    """Sort tasks by 'priority' (highest first) or 'title' (alphabetical).
    Unknown keys raise ValueError rather than silently no-op sorting."""
    if key == "priority":
        return sorted(
            tasks,
            key=lambda t: compute_priority_score(t.urgency, t.impact, t.effort),
            reverse=True,
        )
    if key == "title":
        return sorted(tasks, key=lambda t: t.title.lower())
    raise ValueError(f"unknown sort key: {key!r}")
