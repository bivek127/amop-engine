"""Task comments."""

from datetime import datetime


def add_comment(task, author: str, text: str) -> dict:
    """Append a comment dict to task.comments and return it."""
    comment = {"author": author, "text": text, "at": datetime.now().isoformat()}
    task.comments.append(comment)
    return comment


def format_comment_thread(comments: list[dict]) -> str:
    """Render a list of comment dicts as a readable thread, oldest first."""
    if not comments:
        return "(no comments)"
    lines = [f"{c['author']}: {c['text']}" for c in comments]
    return "\n".join(lines)


def latest_comment(comments: list[dict]) -> dict | None:
    """The most recently added comment, or None if there are none."""
    return comments[-1] if comments else None
