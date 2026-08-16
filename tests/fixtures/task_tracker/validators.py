"""Field and transition validation for tasks."""

from exceptions import InvalidTransitionError, ValidationError
from models import ALLOWED_TRANSITIONS, Status


def validate_task_fields(title: str, urgency: int, impact: int, effort: int) -> None:
    """Raise ValidationError if any field is out of range or missing.
    Doesn't return anything -- callers just check for the exception."""
    if not title or not title.strip():
        raise ValidationError("title must not be empty")
    for name, value in (("urgency", urgency), ("impact", impact), ("effort", effort)):
        if not (1 <= value <= 10):
            raise ValidationError(f"{name} must be between 1 and 10, got {value}")


def validate_status_transition(current: Status, new: Status) -> None:
    """Raise InvalidTransitionError if `current -> new` isn't in
    ALLOWED_TRANSITIONS."""
    if new not in ALLOWED_TRANSITIONS.get(current, set()):
        raise InvalidTransitionError(f"cannot move a task from {current} to {new}")


def validate_assignee(assignee: str | None, team_members: list[str]) -> None:
    """Raise ValidationError if assignee is set but not a known team member."""
    if assignee is not None and assignee not in team_members:
        raise ValidationError(f"{assignee!r} is not a known team member")
