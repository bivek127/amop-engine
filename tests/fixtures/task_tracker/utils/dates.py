"""Date helpers used by scheduling.py."""

from datetime import date, datetime, timedelta


def parse_date(value: str) -> date:
    """Parse an ISO 'YYYY-MM-DD' string into a date."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def business_days_between(start: date, end: date) -> int:
    """Count weekdays (Mon-Fri) strictly between start and end, exclusive
    of start, inclusive of end."""
    if end <= start:
        return 0
    count = 0
    current = start
    while current < end:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return count


def add_business_days(start: date, days: int) -> date:
    """Advance `start` by `days` business days (skipping weekends)."""
    current = start
    remaining = days
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current
