"""Priority scoring -- ranks tasks for sorting so the most important work
surfaces first.
"""

URGENCY_WEIGHT = 3
IMPACT_WEIGHT = 2
EFFORT_WEIGHT = 0.5


def compute_priority_score(urgency: int, impact: int, effort: int) -> float:
    """Score a task for sorting -- higher scores sort first.

    Combines urgency, impact, and effort into one number so tasks can be
    ranked. See URGENCY_WEIGHT/IMPACT_WEIGHT/EFFORT_WEIGHT above for the
    relative weighting.
    """
    return (urgency * URGENCY_WEIGHT + impact * IMPACT_WEIGHT) / effort


def priority_bucket(score: float) -> str:
    """Coarse label for a score, used in reports.summarize_by_status."""
    if score >= 25:
        return "critical"
    if score >= 15:
        return "high"
    if score >= 5:
        return "medium"
    return "low"


def rank_tasks(tasks: list) -> list:
    """Sort tasks by compute_priority_score, highest first."""
    return sorted(
        tasks,
        key=lambda t: compute_priority_score(t.urgency, t.impact, t.effort),
        reverse=True,
    )
