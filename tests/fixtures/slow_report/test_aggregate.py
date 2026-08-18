"""Correctness tests -- green before and after any optimization.

These exist so a "faster" change that breaks behavior gets caught: the
Optimizer's improvement threshold measures speed, and speed alone is a
terrible definition of success.
"""

from app.aggregate import count_unique_visitors, summarize


def test_counts_distinct_ids():
    assert count_unique_visitors(["a", "b", "a", "c", "b"]) == 3


def test_empty_input():
    assert count_unique_visitors([]) == 0


def test_all_identical():
    assert count_unique_visitors(["x"] * 50) == 1


def test_summarize_reports_both_totals():
    assert summarize(["a", "b", "a"]) == {"total_events": 3, "unique_visitors": 2}
