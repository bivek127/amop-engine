# buggy_calculator — AMOP fixture repo

A small, self-contained Python project used as AMOP's end-to-end fixture
(spec Section 19.3). It contains **one deliberately seeded bug**, with a
test suite where two tests fail because of it.

Run its tests directly (from a materialized copy, not this directory):

```
python -m pytest -q
```

Expected on the unfixed baseline: 8 passed, 2 failed —
`test_average_of_three_numbers` (assertion, `3.0 != 2.0`) and
`test_average_of_single_value` (`ZeroDivisionError`). Both trace to the
same single root cause, which is what makes this a real investigation
target rather than a trivial one.

## Why there is no `.git` directory here

This is tracked as plain files. A nested `.git` inside the AMOP repo
would be stored as a gitlink (mode 160000) and these files would never
actually be committed. Instead, `amop fix` copies this directory into a
per-task scratch dir and runs `git init` + a baseline commit inside the
sandbox container, so every run starts from a byte-identical pristine
repo and nothing here is ever mutated.
