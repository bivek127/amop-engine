"""Custom exception types for task_tracker."""


class TaskTrackerError(Exception):
    """Base class for all task_tracker errors."""


class TaskNotFoundError(TaskTrackerError):
    """Raised when a task id doesn't exist in the store."""


class InvalidTransitionError(TaskTrackerError):
    """Raised when a status transition isn't allowed."""


class ValidationError(TaskTrackerError):
    """Raised when task field values fail validation."""
