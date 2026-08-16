import pytest

from calculator import Calculator, DivisionByZeroError


@pytest.fixture
def calc():
    return Calculator()


def test_add(calc):
    assert calc.add(2, 3) == 5


def test_subtract(calc):
    assert calc.subtract(10, 4) == 6


def test_multiply(calc):
    assert calc.multiply(3, 4) == 12


def test_divide(calc):
    assert calc.divide(10, 2) == 5


def test_divide_by_zero_raises(calc):
    with pytest.raises(DivisionByZeroError):
        calc.divide(1, 0)


def test_percentage(calc):
    assert calc.percentage(25, 200) == 12.5


def test_percentage_of_zero_raises(calc):
    with pytest.raises(DivisionByZeroError):
        calc.percentage(1, 0)


def test_average_of_three_numbers(calc):
    assert calc.average([1, 2, 3]) == 2.0


def test_average_of_single_value(calc):
    assert calc.average([7]) == 7.0


def test_average_of_empty_list_raises(calc):
    with pytest.raises(ValueError):
        calc.average([])
