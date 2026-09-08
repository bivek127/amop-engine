# buggy_js_calculator — AMOP fixture repo

A small, self-contained JavaScript project used as AMOP's end-to-end
JavaScript fixture (Milestone 25, spec Section 19.3). The JavaScript
counterpart to `buggy_calculator` -- same seeded bug class (one method
divides by `length - 1` instead of `length`), same shape, so the two
fixtures are directly comparable. It contains **one deliberately seeded
bug**, with a test suite where two tests fail because of it.

Run its tests directly (from a materialized copy, not this directory):

```
jest
```

Expected on the unfixed baseline: 8 passed, 2 failed —
`average of three numbers` (assertion, `average([1,2,3])` returns `3`,
not the expected `2.0`) and `average of single value`
(`average([7])` returns `Infinity`, not `7.0`). Both trace to the same
single root cause (`Calculator.average`'s `values.length - 1`
divisor), which is what makes this a real investigation target rather
than a trivial one.

One deliberate, honest difference from the Python fixture: Python's
`average([7])` raises `ZeroDivisionError` (division by zero throws in
Python), while JavaScript's `7 / 0` evaluates to `Infinity` rather than
throwing -- so the JS test fails on a wrong *value*, not a raised
exception. Real language behavior, not smoothed over to force parity.

No `jest`/dependencies are declared in `package.json` -- Jest is
pre-installed globally in the Node sandbox image
(`amop/sandbox/Dockerfile.node`), the same reason none of the Python
fixtures declare `pytest` as a dependency either.

## Why there is no `.git` directory here

This is tracked as plain files. A nested `.git` inside the AMOP repo
would be stored as a gitlink (mode 160000) and these files would never
actually be committed. Instead, `amop fix` copies this directory into a
per-task scratch dir and runs `git init` + a baseline commit inside the
sandbox container, so every run starts from a byte-identical pristine
repo and nothing here is ever mutated.
