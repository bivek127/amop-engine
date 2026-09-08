const { Calculator, DivisionByZeroError } = require("./calculator");

let calc;
beforeEach(() => {
  calc = new Calculator();
});

test("add", () => {
  expect(calc.add(2, 3)).toBe(5);
});

test("subtract", () => {
  expect(calc.subtract(10, 4)).toBe(6);
});

test("multiply", () => {
  expect(calc.multiply(3, 4)).toBe(12);
});

test("divide", () => {
  expect(calc.divide(10, 2)).toBe(5);
});

test("divide by zero raises", () => {
  expect(() => calc.divide(1, 0)).toThrow(DivisionByZeroError);
});

test("percentage", () => {
  expect(calc.percentage(25, 200)).toBe(12.5);
});

test("percentage of zero raises", () => {
  expect(() => calc.percentage(1, 0)).toThrow(DivisionByZeroError);
});

test("average of three numbers", () => {
  expect(calc.average([1, 2, 3])).toBe(2.0);
});

test("average of single value", () => {
  expect(calc.average([7])).toBe(7.0);
});

test("average of empty list raises", () => {
  expect(() => calc.average([])).toThrow();
});
