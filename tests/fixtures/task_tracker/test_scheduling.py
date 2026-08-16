from datetime import date

from scheduling import days_until_due, is_overdue


def test_is_overdue_true_for_past_date():
    assert is_overdue("2020-01-01", today=date(2020, 6, 1))


def test_is_overdue_false_for_future_date():
    assert not is_overdue("2030-01-01", today=date(2020, 6, 1))


def test_is_overdue_false_for_no_due_date():
    assert not is_overdue(None, today=date(2020, 6, 1))


def test_days_until_due():
    assert days_until_due("2020-06-10", today=date(2020, 6, 1)) == 9
