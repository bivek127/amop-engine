# task_tracker — AMOP Milestone 5 fixture repo

A small, self-contained task-tracking library — deliberately larger than
`tests/fixtures/buggy_calculator/` (14 source files, 5 test files) so
Milestone 5's `search_code` tool has something real to search instead of
a repo small enough to just read every file.

## The seeded bug

`priority.compute_priority_score()` divides by `effort` instead of
subtracting a proportional cost for it. The result: a high-effort,
genuinely urgent task can score *lower* than a trivial, low-effort task
that has neither urgency nor impact — effort ends up dominating the
ranking instead of moderating it. Two tests in `test_priority.py` fail on
the unfixed baseline:

- `test_urgent_task_outranks_a_trivial_low_effort_task`
- `test_rank_tasks_puts_urgent_task_first`

Run its tests directly (from a materialized copy, not this directory):

```
python -m pytest -q
```

Expected on the unfixed baseline: those 2 tests fail, everything else
(13 tests across 5 files) passes.

## Why there is no `.git` directory here

Same reasoning as `buggy_calculator/`: tracked as plain files, `git
init` + a baseline commit happens per run inside the sandbox container,
not checked in here — see that fixture's own README for the full
rationale.
