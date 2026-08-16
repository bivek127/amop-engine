"""Scheduling helpers: turning a task's effort into a completion estimate,
and checking overdue status."""

from datetime import date

from utils.dates import add_business_days, parse_date

EFFORT_TO_DAYS = {1: 1, 2: 1, 3: 2, 4: 3, 5: 5, 6: 7, 7: 9, 8: 12, 9: 15, 10: 20}


def estimate_completion_date(start: date, effort: int) -> date:
    """Estimate when a task will finish, given its 1-10 effort score."""
    days = EFFORT_TO_DAYS.get(effort, effort * 2)
    return add_business_days(start, days)


def is_overdue(due_date: str | None, today: date) -> bool:
    """True if `due_date` (ISO string, or None) is in the past relative
    to `today`."""
    if due_date is None:
        return False
    return parse_date(due_date) < today


def days_until_due(due_date: str | None, today: date) -> int | None:
    """Days remaining until due_date, or None if there is no due date.
    Negative if already overdue."""
    if due_date is None:
        return None
    return (parse_date(due_date) - today).days
