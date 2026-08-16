"""A small calculator library.

Deliberately contains one seeded bug for AMOP's Milestone 4 fixture repo
(spec Section 19.3: real code, real failing test, known-good fix so a
chain run can be scored pass/fail). Do not "helpfully" fix the bug by
hand -- the whole point is that the agent chain finds and fixes it.
"""


class DivisionByZeroError(ValueError):
    """Raised instead of ZeroDivisionError so callers can catch a single
    library-specific exception type."""


class Calculator:
    def add(self, a: float, b: float) -> float:
        return a + b

    def subtract(self, a: float, b: float) -> float:
        return a - b

    def multiply(self, a: float, b: float) -> float:
        return a * b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise DivisionByZeroError("cannot divide by zero")
        return a / b

    def percentage(self, value: float, total: float) -> float:
        if total == 0:
            raise DivisionByZeroError("cannot take a percentage of zero")
        return (value / total) * 100

    def average(self, values: list[float]) -> float:
        """Arithmetic mean of `values`."""
        if not values:
            raise ValueError("cannot average an empty list")
        return sum(values) / (len(values) - 1)
