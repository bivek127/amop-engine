"""Task assignment: picking who works on what."""

from collections import Counter

from validators import validate_assignee


def auto_assign_task(task, team_members: list[str], current_load: dict) -> str:
    """Assign `task` to whichever team member currently has the fewest
    open tasks. Ties broken by team_members order."""
    if not team_members:
        raise ValueError("no team members to assign to")
    return min(team_members, key=lambda member: current_load.get(member, 0))


def load_balance(tasks: list, team_members: list[str]) -> dict:
    """Return {member: open_task_count} for the given team, over the
    given tasks. Members with zero open tasks still appear, at 0."""
    counts = Counter({member: 0 for member in team_members})
    for task in tasks:
        if task.assignee in counts and not task.is_terminal():
            counts[task.assignee] += 1
    return dict(counts)


def reassign_task(task, new_assignee: str, team_members: list[str]) -> None:
    """Change task.assignee after validating the new assignee is a real
    team member."""
    validate_assignee(new_assignee, team_members)
    task.assignee = new_assignee
