"""Notification formatting and delivery."""

from utils.formatting import truncate


def format_notification(task, event: str) -> str:
    """Render a one-line notification for `event` (e.g. 'assigned',
    'status_changed', 'commented') on `task`."""
    title = truncate(task.title, 40)
    return f"[{event}] {title} (assignee: {task.assignee or 'unassigned'})"


def notify_assignee(task, message: str, sender=print) -> None:
    """Deliver `message` to task's assignee via `sender` (defaults to
    print -- a real deployment would inject something that actually
    sends mail/Slack/etc)."""
    if task.assignee is None:
        return
    sender(f"To {task.assignee}: {message}")


def notify_on_status_change(task, old_status, sender=print) -> None:
    """Convenience wrapper: format + send a status_changed notification."""
    message = format_notification(task, "status_changed")
    notify_assignee(task, f"{message} (was {old_status})", sender=sender)
