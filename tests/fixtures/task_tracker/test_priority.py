from priority import compute_priority_score, priority_bucket, rank_tasks
from models import Task


def test_priority_score_increases_with_urgency():
    low = compute_priority_score(urgency=2, impact=5, effort=5)
    high = compute_priority_score(urgency=9, impact=5, effort=5)
    assert high > low


def test_urgent_task_outranks_a_trivial_low_effort_task():
    # A genuinely urgent, high-impact, high-effort task must still beat a
    # trivial task that just happens to be cheap to do -- effort is a
    # cost that should reduce a score somewhat, not one that can flip
    # the ranking against real urgency and impact.
    urgent_and_big = compute_priority_score(urgency=9, impact=8, effort=10)
    trivial = compute_priority_score(urgency=1, impact=1, effort=1)
    assert urgent_and_big > trivial


def test_rank_tasks_puts_urgent_task_first():
    urgent = Task(id="1", title="Fix outage", description="", urgency=9, impact=8, effort=10)
    trivial = Task(id="2", title="Rename variable", description="", urgency=1, impact=1, effort=1)
    ranked = rank_tasks([trivial, urgent])
    assert ranked[0].id == "1"


def test_priority_bucket_boundaries():
    assert priority_bucket(30) == "critical"
    assert priority_bucket(20) == "high"
    assert priority_bucket(10) == "medium"
    assert priority_bucket(1) == "low"
