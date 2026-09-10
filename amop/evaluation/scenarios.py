"""Benchmark scenario definitions — spec Section 20.2.

Every scenario points at a fixture that already exists in
`tests/fixtures/`, seeded by an earlier milestone. Nothing here authors
a new repo: 20.2 wants scenarios "checked into tests/fixtures/ alongside
the seeded repos (19.3)", and they already are.

**Scoring is objective, and deliberately not the chain's opinion of
itself.** Each scenario declares an `expected` outcome that is checked
against real state after the run:

  - the task's terminal state, read from Postgres (`tasks.state`)
  - for bug-fix scenarios, the fixture's OWN test suite re-run in a
    sandbox against the produced code

A chain that reports success while the fixture's tests still fail scores
as a failure here, which is the whole point -- it is the same rule
Milestone 14's Reporter established ("a model may describe the window;
it may not decide the counts") applied to benchmarking.

Note on the fixture list: Milestone 28's brief refers to a `larger_repo`
fixture from Milestone 5. No such directory exists -- Milestone 5's
fixture was `task_tracker`, which is included below. Recorded rather
than quietly substituted, same documentation-drift class as Milestone
23's schema-vs-docs finding.
"""

from dataclasses import dataclass
from pathlib import Path

FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

# Task types, matching what the CLI writes so metrics can group by them
# exactly as `tasks.task_type` stores them.
BUG_FIX = "bug_fix"
OPTIMIZATION = "optimization"
DEPENDENCY_UPDATE = "dependency_update"


@dataclass(frozen=True)
class Scenario:
    """One benchmark case.

    `baseline_failing` / `baseline_total` are the fixture's documented
    pre-fix test counts (from each fixture's own README). They are not
    decoration: the runner asserts the baseline actually reproduces
    before crediting a fix, so a fixture that drifted -- or one whose
    tests fail for an unrelated environmental reason -- is caught as a
    broken scenario instead of silently scoring every model as failing.
    """

    name: str
    fixture: str
    task_type: str
    description: str
    # What a correct fix looks like, in prose, for the report. Scoring
    # uses the machine-checkable fields below, never this string.
    expected: str
    baseline_failing: int = 0
    baseline_total: int = 0
    # Optimization/dependency scenarios need their own entry point.
    entry_point: str | None = None
    manifest: str | None = None

    @property
    def repo_path(self) -> Path:
        return FIXTURES_ROOT / self.fixture


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="calculator_average",
        fixture="buggy_calculator",
        task_type=BUG_FIX,
        description="Why does average() return the wrong value?",
        expected="average() divides by len(values), not len(values) - 1",
        baseline_failing=2,
        baseline_total=10,
    ),
    Scenario(
        name="priority_score",
        fixture="task_tracker",
        task_type=BUG_FIX,
        description=(
            "Why does a low priority task outrank an urgent one when "
            "sorted by priority score?"
        ),
        expected=(
            "compute_priority_score stops dividing by effort; urgent "
            "tasks outrank trivial ones"
        ),
        baseline_failing=2,
        baseline_total=13,
    ),
    Scenario(
        name="js_average",
        fixture="buggy_js_calculator",
        task_type=BUG_FIX,
        description="Why does averaging a list of numbers give the wrong result?",
        expected="Calculator.average divides by values.length, not length - 1",
        baseline_failing=2,
        baseline_total=10,
    ),
    Scenario(
        name="react_count_leak",
        fixture="buggy_react_cart",
        task_type=BUG_FIX,
        description=(
            "The cart summary shows a stray 0 on the page when the cart "
            "is empty. Why?"
        ),
        expected=(
            "the JSX guard becomes items.length > 0 && ..., so an empty "
            "cart renders no stray 0"
        ),
        baseline_failing=1,
        baseline_total=6,
    ),
    Scenario(
        name="slow_aggregate",
        fixture="slow_report",
        task_type=OPTIMIZATION,
        description="The aggregate report endpoint is slow.",
        expected=(
            "measured improvement clears the configured threshold and the "
            "correctness suite stays green"
        ),
        entry_point="benchmark.py",
    ),
    Scenario(
        name="vulnerable_requests",
        fixture="outdated_deps",
        task_type=DEPENDENCY_UPDATE,
        description="requests is pinned to a version with published advisories.",
        expected="requests moves off the advisory-bearing pin, tests still pass",
        manifest="requirements.txt",
    ),
)

SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}


def select(names: list[str] | None = None) -> list[Scenario]:
    """The scenarios to run. Unknown names raise rather than being
    skipped: a typo that silently shrinks the benchmark would make two
    runs quietly incomparable, which is worse than a hard error."""
    if not names:
        return list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS_BY_NAME]
    if unknown:
        raise ValueError(
            f"unknown scenario(s): {unknown}; known: {sorted(SCENARIOS_BY_NAME)}"
        )
    return [SCENARIOS_BY_NAME[n] for n in names]
