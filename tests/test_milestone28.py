"""Milestone 28 — Evaluation Framework. Spec Sections 20.1 (metrics),
20.2 (benchmark scenarios), 20.3 (model comparison), 11.3 (token
logging to agent_actions).

The metric functions are pure, so most of this file needs no database
and no agent -- the arithmetic that decides whether AMOP is getting
better or worse is checked against hand-written inputs. The tests that
DO need Postgres are the ones proving the numbers come from ground
truth rather than from anything a chain said about itself.
"""

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import select, text

from amop.audit.actions import MODEL_CALL_TOOL_NAME, record_model_call
from amop.database.models import AgentAction
from amop.evaluation import metrics as m
from amop.evaluation.scenarios import SCENARIOS, SCENARIOS_BY_NAME, select as select_scenarios
from amop.database.session import init_db, make_engine, make_session_factory
from amop.orchestrator.state_machine import TaskState
from tests.test_milestone14 import TEST_DATABASE_URL


@pytest_asyncio.fixture
async def engine():
    eng = make_engine(TEST_DATABASE_URL)
    await init_db(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    session_factory = make_session_factory(engine)
    async with session_factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.execute(
            text('TRUNCATE agent_actions, task_transitions, tasks RESTART IDENTITY CASCADE')
        )


def outcome(
    scenario="s",
    state=TaskState.RESOLVED.value,
    tests_pass=True,
    reached_pr=None,
    tool_calls=0,
    input_tokens=0,
    output_tokens=0,
    created_at=None,
    resolved_at=None,
) -> m.TaskOutcome:
    return m.TaskOutcome(
        task_id=f"id-{scenario}-{state}",
        scenario=scenario,
        task_type="bug_fix",
        state=state,
        tests_pass=tests_pass,
        reached_pr_creation=(
            reached_pr if reached_pr is not None
            else state not in (TaskState.FAILED.value,
                               TaskState.NEEDS_HUMAN_INPUT.value)
        ),
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        created_at=created_at,
        resolved_at=resolved_at,
    )


# =======================================================================
# Scenario suite -- every scenario points at a fixture that exists
# =======================================================================


def test_every_scenario_points_at_a_real_fixture_on_disk():
    """The brief named a `larger_repo` fixture that does not exist. This
    is the test that would have caught that before a two-hour benchmark
    run died halfway through."""
    for scenario in SCENARIOS:
        assert scenario.repo_path.is_dir(), f"{scenario.name} -> {scenario.repo_path}"


def test_the_suite_covers_all_three_scenario_types():
    """Section 20.2 asks for a seeded bug, a slowed endpoint, and a
    vulnerable pinned dependency -- each exercising a different agent."""
    assert {s.task_type for s in SCENARIOS} == {
        "bug_fix",
        "optimization",
        "dependency_update",
    }


def test_selecting_an_unknown_scenario_raises_rather_than_silently_skipping():
    """A typo that quietly shrank the suite would make two runs
    incomparable while still printing a confident-looking report."""
    with pytest.raises(ValueError, match="unknown scenario"):
        select_scenarios(["calculator_average", "no_such_scenario"])


def test_selecting_nothing_returns_the_whole_suite():
    assert len(select_scenarios(None)) == len(SCENARIOS)
    assert len(select_scenarios([])) == len(SCENARIOS)


def test_selection_preserves_the_requested_order():
    picked = select_scenarios(["js_average", "calculator_average"])
    assert [s.name for s in picked] == ["js_average", "calculator_average"]


# =======================================================================
# Metric arithmetic, against hand-verifiable inputs
# =======================================================================


def test_task_success_rate_counts_resolved_over_decided():
    outcomes = [
        outcome(state=TaskState.RESOLVED.value),
        outcome(state=TaskState.RESOLVED.value),
        outcome(state=TaskState.FAILED.value),
    ]
    result = m.task_success_rate(outcomes)
    assert result.value == pytest.approx(200 / 3)  # 2 of 3
    assert "2/3" in result.detail


def test_needs_human_input_is_excluded_from_the_success_denominator():
    """A task parked for a human neither succeeded nor failed. Folding
    it into either would misreport this metric AND the intervention
    metric that exists precisely to count it."""
    decided_only = [
        outcome(state=TaskState.RESOLVED.value),
        outcome(state=TaskState.FAILED.value),
    ]
    with_parked = decided_only + [outcome(state=TaskState.NEEDS_HUMAN_INPUT.value)]

    assert m.task_success_rate(decided_only).value == 50.0
    assert m.task_success_rate(with_parked).value == 50.0  # unchanged
    assert m.human_intervention_rate(with_parked).value == pytest.approx(100 / 3)


def test_fix_success_requires_passing_tests_not_just_a_resolved_state():
    """The heart of the whole harness: RESOLVED is the chain's verdict,
    passing tests are the fact. A chain that declares victory while the
    fixture's suite still fails must not score as a fix."""
    liar = outcome(state=TaskState.RESOLVED.value, tests_pass=False)
    honest = outcome(state=TaskState.RESOLVED.value, tests_pass=True)

    assert m.fix_success_rate([liar]).value == 0.0
    assert m.fix_success_rate([honest]).value == 100.0
    assert m.fix_success_rate([liar, honest]).value == 50.0


def test_tool_efficiency_is_the_failed_over_succeeded_ratio():
    """Spec 20.1: 'a rising ratio signals agent drift worth
    investigating' -- so it must be failed-per-task over
    succeeded-per-task, not the other way round."""
    outcomes = [
        outcome(state=TaskState.RESOLVED.value, tool_calls=4),
        outcome(state=TaskState.FAILED.value, tool_calls=12),
    ]
    result = m.tool_efficiency(outcomes)
    assert result.value == pytest.approx(3.0)


def test_tool_efficiency_is_unavailable_without_both_kinds_of_task():
    """A ratio needs both halves. Reporting 0.0, or silently dividing by
    zero, would be a fabricated measurement."""
    only_wins = [outcome(state=TaskState.RESOLVED.value, tool_calls=4)]
    result = m.tool_efficiency(only_wins)
    assert not result.available
    assert "at least one succeeded AND one failed" in result.unavailable_reason


def test_tokens_per_task_averages_input_plus_output():
    outcomes = [
        outcome(input_tokens=1000, output_tokens=100),
        outcome(input_tokens=800, output_tokens=100),
    ]
    assert m.tokens_per_task(outcomes).value == pytest.approx(1000.0)


def test_zero_tokens_reports_unavailable_rather_than_a_real_looking_zero():
    """Distinguishes 'nothing reported usage' from 'genuinely used zero
    tokens'. The same distinction agent_actions.estimated_cost_usd makes
    when it says it 'is not silently reporting zero-cost'."""
    result = m.tokens_per_task([outcome(input_tokens=0, output_tokens=0)])
    assert not result.available
    assert "no model call reported token usage" in result.unavailable_reason


def test_duration_uses_measured_compute_time_not_wall_clock_timestamps():
    """Found the hard way: one real scenario recorded 47.9 wall-clock
    minutes against 5.8 minutes of compute, because the laptop slept
    mid-run. Postgres timestamps are wall-clock; time.monotonic() stops
    while the machine is asleep. A model-comparison metric that inflates
    with the operator's sleep schedule measures nothing, so duration
    comes from the monotonic clock and the DB timestamps are ignored
    for it."""
    slept_through = m.TaskOutcome(
        task_id="t",
        scenario="s",
        task_type="bug_fix",
        state=TaskState.NEEDS_HUMAN_INPUT.value,
        # Wall clock claims 48 minutes...
        created_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        resolved_at=datetime(2026, 1, 1, 12, 48, tzinfo=UTC),
        # ...but only 6 minutes of it was compute.
        duration_seconds=360.0,
        reached_pr_creation=True,
    )
    result = m.mean_time_to_resolution([slept_through])
    assert result.value == pytest.approx(6.0)
    assert "asleep" in result.basis


def test_duration_is_unavailable_when_the_runner_measured_nothing():
    """Falling back to wall-clock timestamps here would quietly
    reintroduce the sleep-inflation bug, so there is no fallback."""
    result = m.mean_time_to_resolution(
        [
            m.TaskOutcome(
                task_id="t",
                scenario="s",
                task_type="bug_fix",
                state=TaskState.RESOLVED.value,
                created_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                resolved_at=datetime(2026, 1, 1, 12, 48, tzinfo=UTC),
                duration_seconds=None,
            )
        ]
    )
    assert not result.available


# =======================================================================
# Unmeasurable metrics are named, never faked
# =======================================================================


def test_structurally_unmeasurable_metrics_are_reported_with_reasons():
    unavailable = m.structurally_unavailable()
    names = {metric.name for metric in unavailable}
    assert names == {"PR acceptance rate", "regression rate", "false positive rate"}
    for metric in unavailable:
        assert metric.value is None
        assert metric.unavailable_reason


def test_an_unavailable_metric_never_renders_as_a_number():
    """The failure mode this guards against: a 0.0 that looks like a
    measurement, averages into a comparison, and is never questioned."""
    rendered = m.Metric.unavailable("regression rate", "needs elapsed time").render()
    assert "NOT MEASURABLE" in rendered
    assert "0.0" not in rendered
    assert "needs elapsed time" in rendered


def test_the_report_prints_task_ids_so_numbers_can_be_checked_by_hand():
    """A report nobody can verify is just a claim. Every figure must be
    traceable back to rows a human can query themselves."""
    report = m.compute_all("qwen2.5-coder:14b", [outcome(scenario="calculator_average")])
    rendered = report.render()
    assert "calculator_average" in rendered
    assert "id-calculator_average-RESOLVED" in rendered
    assert "never from a" in rendered and "self-report" in rendered


def test_compute_all_reports_every_section_20_1_metric():
    """Present-and-unavailable beats silently absent: a metric missing
    from the report reads as an oversight, not as a known gap."""
    report = m.compute_all("m", [outcome()])
    names = {metric.name for metric in report.metrics}
    for expected in ("task success rate", "PR acceptance rate", "regression rate"):
        assert any(expected in n for n in names), expected


# =======================================================================
# Ground truth, against the real database
# =======================================================================


async def test_token_usage_is_recorded_to_agent_actions(session):
    """Spec 11.3: 'every complete() call logs {input_tokens,
    output_tokens...} to agent_actions'. Before this milestone those
    counts were captured by the provider and discarded by the agent
    loop, which is why the tokens-per-task metric had nothing to read."""

    class Ctx:
        db_session = session
        task_id = None

    await record_model_call(
        Ctx(),
        agent_name="reviewer",
        model="qwen2.5-coder:14b",
        input_tokens=1234,
        output_tokens=56,
        latency_ms=789,
    )

    row = (
        await session.execute(
            select(AgentAction)
            .where(AgentAction.tool_name == MODEL_CALL_TOOL_NAME)
            .order_by(AgentAction.id.desc())
            .limit(1)
        )
    ).scalar_one()

    assert row.input_tokens == 1234
    assert row.output_tokens == 56
    assert row.agent_name == "reviewer"
    assert row.arguments == {"model": "qwen2.5-coder:14b"}
    # No Safety Engine decision was made for a model call, and inventing
    # an "ALLOW" would put a permission check in the audit trail that
    # never actually ran.
    assert row.decision is None


async def test_model_call_rows_are_separable_from_tool_call_rows(session):
    """One table, two row kinds. If they weren't separable, the
    tool-efficiency metric would count model calls as tool calls and
    every efficiency figure would be silently inflated."""

    class Ctx:
        db_session = session
        task_id = None

    before = (
        await session.execute(
            select(AgentAction).where(AgentAction.tool_name != MODEL_CALL_TOOL_NAME)
        )
    ).scalars().all()

    await record_model_call(
        Ctx(),
        agent_name="coder",
        model="qwen2.5-coder:14b",
        input_tokens=10,
        output_tokens=5,
    )

    after = (
        await session.execute(
            select(AgentAction).where(AgentAction.tool_name != MODEL_CALL_TOOL_NAME)
        )
    ).scalars().all()

    # Recording a model call must not change the tool-call population.
    assert len(after) == len(before)


async def test_a_perfect_run_that_cannot_open_a_github_pr_still_scores_as_success(
    session,
):
    """The correction that keeps this benchmark from measuring the wrong
    thing entirely.

    Measured against a real chain run: correct diagnosis, correct fix,
    10/10 tests, Reviewer APPROVED -- and still a terminal state of
    NEEDS_HUMAN_INPUT, because PR creation is unconditional after
    approval and opening a real GitHub PR against a materialized fixture
    fails with "no history in common with main". RESOLVED and MERGED
    need a human merging a real PR, so no scenario can ever reach them.

    Scoring on the terminal state would have marked every scenario a
    failure for a reason no model could affect, and both models in the
    comparison would have scored 0% -- a benchmark measuring GitHub
    configuration rather than model quality.
    """
    perfect_but_pr_failed = m.TaskOutcome(
        task_id="t1",
        scenario="calculator_average",
        task_type="bug_fix",
        state=TaskState.NEEDS_HUMAN_INPUT.value,
        reached_pr_creation=True,   # the Reviewer approved
        tests_pass=True,            # the fix is genuinely correct
    )

    assert perfect_but_pr_failed.succeeded is True
    assert m.task_success_rate([perfect_but_pr_failed]).value == 100.0
    assert m.fix_success_rate([perfect_but_pr_failed]).value == 100.0
    # ...and it must NOT inflate the human-intervention metric: the stop
    # was environmental, not an agent giving up.
    assert perfect_but_pr_failed.needed_human is False
    assert m.human_intervention_rate([perfect_but_pr_failed]).value == 0.0


async def test_a_genuine_agent_stop_still_counts_as_human_intervention(session):
    """The converse, so the exemption above can't swallow real stops: a
    task that never reached the Reviewer and asked for a human is
    exactly what the intervention metric is for."""
    gave_up = m.TaskOutcome(
        task_id="t2",
        scenario="priority_score",
        task_type="bug_fix",
        state=TaskState.NEEDS_HUMAN_INPUT.value,
        reached_pr_creation=False,  # never got past investigation
        tests_pass=False,
    )

    assert gave_up.succeeded is False
    assert gave_up.needed_human is True
    assert m.human_intervention_rate([gave_up]).value == 100.0
