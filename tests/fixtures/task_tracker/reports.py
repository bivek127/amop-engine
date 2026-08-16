"""Summary reports over a task list."""

from datetime import date

from models import Status
from priority import priority_bucket, compute_priority_score
from scheduling import is_overdue


def summarize_by_status(tasks: list) -> dict:
    """{status_value: count} for every Status, including zero counts."""
    counts = {status.value: 0 for status in Status}
    for task in tasks:
        counts[task.status.value] += 1
    return counts


def summarize_by_priority_bucket(tasks: list) -> dict:
    """{bucket: count} using priority.priority_bucket on each task's score."""
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for task in tasks:
        score = compute_priority_score(task.urgency, task.impact, task.effort)
        counts[priority_bucket(score)] += 1
    return counts


def overdue_report(tasks: list, today: date) -> list:
    """Tasks that are overdue and not yet done."""
    return [
        task
        for task in tasks
        if not task.is_terminal() and is_overdue(task.due_date, today)
    ]
