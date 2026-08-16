import pytest

from exceptions import InvalidTransitionError, ValidationError
from models import Status
from validators import validate_assignee, validate_status_transition, validate_task_fields


def test_validate_task_fields_accepts_valid_input():
    validate_task_fields("Fix bug", urgency=5, impact=5, effort=5)  # should not raise


def test_validate_task_fields_rejects_empty_title():
    with pytest.raises(ValidationError):
        validate_task_fields("", urgency=5, impact=5, effort=5)


def test_validate_task_fields_rejects_out_of_range():
    with pytest.raises(ValidationError):
        validate_task_fields("x", urgency=11, impact=5, effort=5)


def test_validate_status_transition_allows_legal_moves():
    validate_status_transition(Status.OPEN, Status.IN_PROGRESS)  # should not raise


def test_validate_status_transition_rejects_illegal_moves():
    with pytest.raises(InvalidTransitionError):
        validate_status_transition(Status.DONE, Status.OPEN)


def test_validate_assignee_rejects_unknown_member():
    with pytest.raises(ValidationError):
        validate_assignee("nobody", ["alice", "bob"])
