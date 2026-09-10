"""Section 20.1's metrics, computed from ground truth only.

Every function here is pure: it takes already-fetched rows and returns
numbers. No queries, no model, no I/O -- the same discipline as
`safety/engine.py`'s `evaluate()` and `agents/tester.py`'s
`classify_failures`, and for the same reason: the arithmetic that
decides whether AMOP is getting better or worse should be testable
against hand-written inputs, with no database or agent in the loop.

**What a metric may be computed from.** `TaskOutcome` is assembled by
runner.py from `tasks`, `agent_actions`, and a real re-run of the
fixture's own test suite. It deliberately carries no field sourced from
a chain's self-report. A chain that announces success while the
fixture's tests still fail scores as a failure, which is the
benchmarking form of Milestone 14's rule that a model may describe the
window but may not decide the counts.

**Metrics that cannot be measured say so.** Several of 20.1's metrics
need data a benchmark run structurally cannot produce -- a merged
GitHub PR, a human cancelling an alert, days passing so a regression
can recur. Those return `Metric.unavailable(...)` with the reason
rather than `0.0`. A fabricated zero is worse than a gap: it looks like
a measurement, it averages into comparisons, and nobody ever questions
it. This follows `agent_actions.estimated_cost_usd`'s own precedent --
"it is not silently reporting zero-cost."
"""

from dataclasses import dataclass, field
from datetime import datetime

from amop.orchestrator.state_machine import TaskState

# What counts as the chain having succeeded.
#
# NOT `state == RESOLVED`. Measured against a real run: a fixture chain
# that did everything right -- correct diagnosis, correct fix, 10/10
# tests, Reviewer APPROVED -- still ends at NEEDS_HUMAN_INPUT, because
# PR creation is unconditional after approval and opening a real GitHub
# PR against a materialized fixture fails ("no history in common with
# main"). RESOLVED and MERGED require a human approving a real PR, so no
# benchmark scenario can ever reach them.
#
# Reaching PR_CREATION is therefore the honest success signal: it means
# the Reviewer approved and every gate before it passed, which is the
# end of AMOP's *autonomous* work. Everything after is GitHub and human
# territory and says nothing about model quality. Scoring on RESOLVED
# would have marked every scenario a failure for reasons no model could
# affect -- the benchmark would have been measuring GitHub configuration.
SUCCESS_TRANSITION = TaskState.PR_CREATION.value
FAILURE_STATES = frozenset({TaskState.FAILED.value})
INTERVENTION_STATES = frozenset({TaskState.NEEDS_HUMAN_INPUT.value})


@dataclass(frozen=True)
class TaskOutcome:
    """One benchmark task's ground truth.

    `tests_pass` is the fixture's own suite re-run against the produced
    code -- the objective half of scoring. None means the check could
    not be run (e.g. the chain never produced a workspace), which is
    distinct from False and is not counted as a passing fix.
    """

    task_id: str
    scenario: str
    task_type: str
    state: str
    created_at: datetime | None = None
    resolved_at: datetime | None = None
    # Counts from agent_actions, split by row kind (see
    # audit/actions.py's MODEL_CALL_TOOL_NAME).
    tool_calls: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tests_pass: bool | None = None
    error: str | None = None
    # Monotonic seconds actually spent running this scenario,
    # measured by the runner. Preferred over the database
    # timestamps for duration: wall-clock timestamps include any
    # time the machine was ASLEEP, which is not work the model did.
    # Observed for real -- one scenario showed 47.9 wall-clock
    # minutes against 5.8 minutes of compute, because the laptop
    # slept mid-run. Comparing two models on a number that inflates
    # with the operator's sleep schedule would be meaningless.
    duration_seconds: float | None = None
    # Whether the task ever transitioned into PR_CREATION, read from
    # task_transitions. This -- not the terminal state -- is what says
    # the chain finished its autonomous work; see SUCCESS_TRANSITION.
    reached_pr_creation: bool = False

    @property
    def succeeded(self) -> bool:
        return self.reached_pr_creation

    @property
    def failed(self) -> bool:
        return self.state in FAILURE_STATES

    @property
    def needed_human(self) -> bool:
        """A genuine "an agent could not proceed" stop.

        Deliberately excludes a task that reached PR_CREATION first: that
        one asked for a human only because a real PR could not be opened
        against a throwaway fixture repo, which is an environment fact,
        not an agent giving up. Counting it here would report a ~100%
        human-intervention rate for every model regardless of how well
        it actually worked.
        """
        return self.state in INTERVENTION_STATES and not self.reached_pr_creation

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Metric:
    """A metric result, which may legitimately have no value.

    `value is None` and `unavailable_reason` set is a first-class
    outcome, not an error -- see this module's docstring.
    """

    name: str
    value: float | None
    unit: str = ""
    detail: str = ""
    unavailable_reason: str | None = None
    # The ground-truth basis, printed in the report so a human can
    # re-derive the number themselves rather than trusting it.
    basis: str = ""

    @classmethod
    def unavailable(cls, name: str, reason: str) -> "Metric":
        return cls(name=name, value=None, unavailable_reason=reason)

    @property
    def available(self) -> bool:
        return self.value is not None

    def render(self) -> str:
        if not self.available:
            return f"{self.name}: NOT MEASURABLE — {self.unavailable_reason}"
        shown = f"{self.value:.2f}{self.unit}"
        suffix = f"  ({self.detail})" if self.detail else ""
        return f"{self.name}: {shown}{suffix}"


def task_success_rate(outcomes: list[TaskOutcome]) -> Metric:
    """20.1: resolved tasks / (resolved + failed).

    The denominator is deliberately resolved+failed, not all tasks:
    a task parked at NEEDS_HUMAN_INPUT hasn't succeeded or failed, and
    folding it into either would misreport both this metric and the
    human-intervention one that exists to count it.
    """
    decided = [o for o in outcomes if o.succeeded or o.failed]
    if not decided:
        return Metric.unavailable(
            "task success rate",
            "no task reached a success or failure state in this run",
        )
    wins = sum(1 for o in decided if o.succeeded)
    return Metric(
        name="task success rate",
        value=100.0 * wins / len(decided),
        unit="%",
        detail=f"{wins}/{len(decided)} decided tasks",
        basis=(
            "task_transitions shows a transition into PR_CREATION "
            "(Reviewer approved) vs tasks.state == FAILED"
        ),
    )


def fix_success_rate(outcomes: list[TaskOutcome]) -> Metric:
    """20.1's fix success rate, via a declared local proxy.

    The spec defines it as merged PR / detected anomaly. Nothing here
    merges real PRs, so the proxy is the stricter, objectively checkable
    thing available: the task reached a success state AND the fixture's
    own test suite passes against the produced code. Labelled a proxy
    everywhere it is printed, so it is never mistaken for the spec's
    own definition.
    """
    fixable = [o for o in outcomes if o.tests_pass is not None]
    if not fixable:
        return Metric.unavailable(
            "fix success rate (proxy)",
            "no scenario produced a workspace whose tests could be re-run",
        )
    real_fixes = sum(1 for o in fixable if o.succeeded and o.tests_pass)
    return Metric(
        name="fix success rate (proxy)",
        value=100.0 * real_fixes / len(fixable),
        unit="%",
        detail=f"{real_fixes}/{len(fixable)} verified by re-running the fixture's suite",
        basis="tasks.state AND the fixture's own suite re-run in a sandbox",
    )


def human_intervention_rate(outcomes: list[TaskOutcome]) -> Metric:
    """20.1: tasks reaching NEEDS_HUMAN_INPUT / total tasks."""
    if not outcomes:
        return Metric.unavailable("human intervention rate", "no tasks in this run")
    stuck = sum(1 for o in outcomes if o.needed_human)
    return Metric(
        name="human intervention rate",
        value=100.0 * stuck / len(outcomes),
        unit="%",
        detail=f"{stuck}/{len(outcomes)} tasks",
        basis="tasks.state == NEEDS_HUMAN_INPUT",
    )


def tool_efficiency(outcomes: list[TaskOutcome]) -> Metric:
    """20.1: tool calls per successful task vs per failed task.

    The spec's own framing is what makes this worth reporting as a
    ratio: "a rising ratio signals agent drift worth investigating" --
    a model burning many more calls on the tasks it fails is behaving
    differently from one that fails fast.
    """
    succeeded = [o for o in outcomes if o.succeeded]
    failed = [o for o in outcomes if o.failed]
    if not succeeded or not failed:
        return Metric.unavailable(
            "tool efficiency",
            "needs at least one succeeded AND one failed task to form a ratio "
            f"(this run: {len(succeeded)} succeeded, {len(failed)} failed)",
        )
    per_success = sum(o.tool_calls for o in succeeded) / len(succeeded)
    per_failure = sum(o.tool_calls for o in failed) / len(failed)
    if per_success == 0:
        return Metric.unavailable(
            "tool efficiency", "successful tasks made no tool calls at all"
        )
    return Metric(
        name="tool efficiency",
        value=per_failure / per_success,
        unit="x",
        detail=(
            f"{per_failure:.1f} calls/failed vs {per_success:.1f} calls/succeeded"
        ),
        basis="agent_actions rows where tool_name != 'model.complete'",
    )


def tokens_per_task(outcomes: list[TaskOutcome]) -> Metric:
    """20.1's token usage per task, aggregated from agent_actions.

    Real only because this milestone wired provider token counts through
    the agent loop; before that they were captured and discarded. If a
    run predates that (or ran against a provider reporting no usage),
    every count is zero and this reports unavailable rather than
    claiming a genuine zero.
    """
    if not outcomes:
        return Metric.unavailable("tokens per task", "no tasks in this run")
    total = sum(o.total_tokens for o in outcomes)
    if total == 0:
        return Metric.unavailable(
            "tokens per task",
            "no model call reported token usage — the provider returned none, "
            "or these tasks predate token recording",
        )
    return Metric(
        name="tokens per task",
        value=total / len(outcomes),
        unit=" tokens",
        detail=f"{total:,} tokens across {len(outcomes)} tasks",
        basis="SUM(input_tokens + output_tokens) on agent_actions model-call rows",
    )


def mean_time_to_resolution(outcomes: list[TaskOutcome]) -> Metric:
    """20.1's MTTR, via a declared proxy.

    The spec defines it as `resolved_at - incidents.first_seen`. There
    is no `incidents` table in this codebase, so this measures
    `resolved_at - tasks.created_at` -- task duration, not
    incident-to-resolution. Narrower than the spec's metric and labelled
    as such rather than presented as the same number.
    """
    timed = [o for o in outcomes if o.duration_seconds is not None]
    if not timed:
        return Metric.unavailable(
            "mean scenario duration (proxy for MTTR)",
            "the runner recorded no measured duration for any scenario",
        )
    seconds = [o.duration_seconds for o in timed]
    return Metric(
        name="mean scenario duration (proxy for MTTR)",
        value=sum(seconds) / len(seconds) / 60.0,
        unit=" min",
        detail=f"{len(timed)} scenarios, compute time only",
        basis=(
            "monotonic clock around each scenario -- NOT "
            "tasks.resolved_at - created_at, whose wall-clock span "
            "includes time the machine spent asleep, and NOT "
            "incidents.first_seen, which this codebase has no table for"
        ),
    )


# Metrics from 20.1 that a benchmark run structurally cannot produce.
# Present in the report by name, so the gap is visible rather than
# looking like an oversight.
def structurally_unavailable() -> list[Metric]:
    return [
        Metric.unavailable(
            "PR acceptance rate",
            "needs real GitHub PRs merged without human edits; benchmark runs "
            "produce local branches, not merged PRs",
        ),
        Metric.unavailable(
            "regression rate",
            "defined as resolved bugs recurring within N days — needs elapsed "
            "time and repeat runs, not a single benchmark pass",
        ),
        Metric.unavailable(
            "false positive rate",
            "needs AnomalyAlerts human-cancelled as non-issues; no human "
            "cancellation happens inside an automated benchmark",
        ),
    ]


@dataclass
class Report:
    """One benchmark run's full result."""

    model: str
    outcomes: list[TaskOutcome]
    metrics: list[Metric] = field(default_factory=list)

    def render(self) -> str:
        by_scenario: dict[str, list[TaskOutcome]] = {}
        for o in self.outcomes:
            by_scenario.setdefault(o.scenario, []).append(o)
        repeats = max((len(v) for v in by_scenario.values()), default=1)

        lines = [
            f"Evaluation report — model: {self.model}",
            "=" * 62,
            "",
            f"Per-scenario outcomes (ground truth), {repeats} run(s) each:",
        ]
        if repeats > 1:
            # The spread is the point when a run is repeated: one model
            # passing 3/3 and another passing 1/3 are different results
            # even when a single sampled run would have looked identical.
            for name, runs in by_scenario.items():
                passes = sum(
                    1 for o in runs if o.succeeded and o.tests_pass is not False
                )
                tokens = [o.total_tokens for o in runs]
                durations = [
                    o.duration_seconds for o in runs if o.duration_seconds is not None
                ]
                spread = (
                    f"tokens {min(tokens):,}-{max(tokens):,}" if tokens else "tokens n/a"
                )
                timing = (
                    f"  {min(durations)/60:.1f}-{max(durations)/60:.1f} min"
                    if durations
                    else ""
                )
                flag = "" if passes in (0, len(runs)) else "   <-- INCONSISTENT"
                lines.append(
                    f"  {name:22} {passes}/{len(runs)} passed   {spread}{timing}{flag}"
                )
            lines += ["", "Individual runs:"]
        for o in self.outcomes:
            # tests_pass is None by design for optimization/dependency
            # scenarios -- those are gated by their own orchestrator's
            # enforced checks (measured improvement; blast radius + test
            # verdict), not by re-running a fixture suite. Treating None
            # as falsy here made PASS unreachable for them: one scenario
            # transitioned all the way through PR_CREATION to
            # WAITING_FOR_APPROVAL and still printed FAIL.
            verdict = "PASS" if o.succeeded and o.tests_pass is not False else "FAIL"
            tests = (
                "tests n/a"
                if o.tests_pass is None
                else ("tests pass" if o.tests_pass else "tests FAIL")
            )
            lines.append(
                f"  [{verdict}] {o.scenario:22} {o.state:20} {tests:11} "
                f"tools={o.tool_calls:<3} tokens={o.total_tokens:<7} {o.task_id}"
            )
            if o.error:
                lines.append(f"          error: {o.error}")

        lines += ["", "Metrics:"]
        for metric in self.metrics:
            lines.append(f"  {metric.render()}")
            if metric.available and metric.basis:
                lines.append(f"      from: {metric.basis}")

        lines += [
            "",
            "Every number above is computed from Postgres (tasks, agent_actions)",
            "and from re-running each fixture's own test suite -- never from a",
            "chain's self-report. Task IDs are printed so any figure can be",
            "checked directly against the database.",
        ]
        return "\n".join(lines)


def compute_all(model: str, outcomes: list[TaskOutcome]) -> Report:
    metrics = [
        task_success_rate(outcomes),
        fix_success_rate(outcomes),
        human_intervention_rate(outcomes),
        tool_efficiency(outcomes),
        tokens_per_task(outcomes),
        mean_time_to_resolution(outcomes),
        *structurally_unavailable(),
    ]
    return Report(model=model, outcomes=outcomes, metrics=metrics)
