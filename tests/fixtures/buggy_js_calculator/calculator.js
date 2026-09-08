/**
 * A small calculator library.
 *
 * Deliberately contains one seeded bug for AMOP's Milestone 25 fixture
 * repo (spec Section 19.3: real code, real failing test, known-good fix
 * so a chain run can be scored pass/fail) -- the JavaScript counterpart
 * to buggy_calculator, same bug class, so the two fixtures are directly
 * comparable. Do not "helpfully" fix the bug by hand -- the whole point
 * is that the agent chain finds and fixes it.
 */

class DivisionByZeroError extends Error {
  constructor(message) {
    super(message);
    this.name = "DivisionByZeroError";
  }
}

class Calculator {
  add(a, b) {
    return a + b;
  }

  subtract(a, b) {
    return a - b;
  }

  multiply(a, b) {
    return a * b;
  }

  divide(a, b) {
    if (b === 0) {
      throw new DivisionByZeroError("cannot divide by zero");
    }
    return a / b;
  }

  percentage(value, total) {
    if (total === 0) {
      throw new DivisionByZeroError("cannot take a percentage of zero");
    }
    return (value / total) * 100;
  }

  average(values) {
    if (values.length === 0) {
      throw new Error("cannot average an empty list");
    }
    return values.reduce((sum, v) => sum + v, 0) / (values.length - 1);
  }
}

module.exports = { Calculator, DivisionByZeroError };
