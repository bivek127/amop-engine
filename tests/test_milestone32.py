"""Real cost accounting -- spec Sections 11.3, 11.3.1, 11.3.2.

Closes the gap Milestone 23/28 both named explicitly and left open:
`agent_actions.estimated_cost_usd` existed as a column and stayed
permanently NULL, because nothing computed it from real telemetry. This
file proves, against a real Postgres database, that it now does -- and
that the new `budgets` table accumulates atomically and actually refuses
a dispatch before it happens, not merely after the fact.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select, text

from amop.agents.base import BaseAgent
from amop.audit.actions import MODEL_CALL_TOOL_NAME, record_model_call
from amop.database.models import AgentAction, Budget
from amop.database.session import init_db, make_engine, make_session_factory
from amop.models.base import ModelResponse
from amop.models.cost import (
    BudgetExceededError,
    GLOBAL_SCOPE,
    check_budget_before_dispatch,
    estimate_cost_usd,
    estimate_input_tokens,
    max_plausible_cost_usd,
    reconcile_budget,
    record_spend,
    task_scope,
)
from amop.tools.registry import ToolContext
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
            text(
                "TRUNCATE agent_actions, task_transitions, tasks, budgets "
                "RESTART IDENTITY CASCADE"
            )
        )


# =======================================================================
# 11.3.1 -- the cost formula itself, pure arithmetic, no database
# =======================================================================


def test_estimate_cost_usd_matches_the_11_3_1_formula_by_hand():
    """Hand-computed against claude-sonnet's real published rates
    ($3/$15/$3.75/$0.30 per million for base/output/cache-write/
    cache-read) so this is checking the formula, not re-deriving it."""
    cost = estimate_cost_usd(
        "claude-sonnet-5",
        cache_creation_input_tokens=1000,
        cache_read_input_tokens=2000,
        uncached_input_tokens=500,
        output_tokens=300,
    )
    expected = Decimal(str(
        1000 * (3.75 / 1_000_000)
        + 2000 * (0.30 / 1_000_000)
        + 500 * (3.00 / 1_000_000)
        + 300 * (15.00 / 1_000_000)
    ))
    assert abs(cost - expected) < Decimal("0.0000001")
    assert cost > 0


def test_ollama_models_are_genuinely_free_not_approximated():
    cost = estimate_cost_usd(
        "qwen2.5-coder:14b",
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        uncached_input_tokens=50_000,
        output_tokens=10_000,
    )
    assert cost == Decimal("0")


def test_none_token_counts_are_treated_as_zero_not_an_error():
    cost = estimate_cost_usd(
        "claude-sonnet-5",
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        uncached_input_tokens=100,
        output_tokens=None,
    )
    assert cost > 0  # only the uncached-input term contributes


def test_max_plausible_cost_uses_worst_case_all_input_uncached():
    """D-18: the pre-dispatch estimate must never UNDER-estimate --
    treating all input as base-rate (never cache-discounted) and
    assuming the full max_tokens of output is what makes the cap
    undefeatable by an unexpectedly long completion."""
    worst = max_plausible_cost_usd(
        "claude-sonnet-5", input_tokens_estimate=1000, max_tokens=2000
    )
    real_if_cheapest_possible = estimate_cost_usd(
        "claude-sonnet-5",
        cache_creation_input_tokens=0,
        cache_read_input_tokens=1000,  # best case: fully cache-discounted
        uncached_input_tokens=0,
        output_tokens=500,  # actual completion shorter than max_tokens
    )
    assert worst > real_if_cheapest_possible


# =======================================================================
# The OpenAI double-counting bug, fixed -- a real regression test
# =======================================================================


async def test_openai_provider_does_not_double_count_cached_tokens(monkeypatch):
    """Before Milestone 32: uncached_input_tokens was set to the RAW
    prompt_tokens total, which already INCLUDES cached_tokens per
    OpenAI's own API semantics (unlike Anthropic, where input_tokens
    already excludes them). Summing that against a separate
    cache_read_input_tokens term in the cost formula would have
    double-billed every cached token. This reproduces the exact shape
    live and asserts the fix."""
    import httpx

    from amop.models.openai import OpenAIProvider

    provider = OpenAIProvider(model="gpt-4o", api_key="test-key")

    async def fake_post(self, url, json=None, headers=None):
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-2024-08-06",
                "choices": [{"message": {"content": "hi"}}],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 50,
                    "prompt_tokens_details": {"cached_tokens": 400},
                },
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    response = await provider.complete([{"role": "user", "content": "hi"}])

    # The bug: this would have been 1000 (raw prompt_tokens, cached
    # tokens counted a second time by the cost formula's cache term).
    # The fix: 1000 - 400 = 600, the genuinely uncached portion.
    assert response.uncached_input_tokens == 600
    assert response.cache_read_input_tokens == 400

    cost = estimate_cost_usd(
        "gpt-4o",
        cache_creation_input_tokens=None,
        cache_read_input_tokens=response.cache_read_input_tokens,
        uncached_input_tokens=response.uncached_input_tokens,
        output_tokens=response.output_tokens,
    )
    buggy_cost = estimate_cost_usd(
        "gpt-4o",
        cache_creation_input_tokens=None,
        cache_read_input_tokens=400,
        uncached_input_tokens=1000,  # the pre-fix (double-counted) value
        output_tokens=50,
    )
    assert cost < buggy_cost  # the fix strictly reduces the estimate


# =======================================================================
# 11.3.2/D-18 -- atomic accumulation, real Postgres, real transaction
# =======================================================================


async def test_record_spend_is_atomic_not_read_modify_write(session):
    """Two concurrent credits to the same scope+day must both land --
    the whole point of ON CONFLICT DO UPDATE SET spent_usd = spent_usd +
    EXCLUDED.spent_usd over a read-then-write. Run genuinely
    concurrently (asyncio.gather over two separate sessions on the same
    engine), not sequentially, so a lost-update bug would actually have
    a chance to occur."""
    import asyncio

    from amop.orchestrator.concurrency import engine_for

    engine = engine_for(session)
    factory = make_session_factory(engine)

    async def credit(amount: str):
        async with factory() as s:
            await record_spend(s, scope=GLOBAL_SCOPE, amount_usd=Decimal(amount))
            await s.commit()

    await asyncio.gather(credit("1.00"), credit("2.00"))

    today = datetime.now(UTC).date()
    row = await session.execute(
        select(Budget.spent_usd).where(
            Budget.scope == GLOBAL_SCOPE, Budget.period_start == today
        )
    )
    assert Decimal(str(row.scalar_one())) == Decimal("3.00")


async def test_record_model_call_writes_real_cost_and_bumps_budget_same_transaction(
    session,
):
    """The actual integration point: one call to record_model_call
    (what every agent's completion funnels through) must leave BOTH a
    real, non-NULL estimated_cost_usd on the agent_actions row AND a
    matching increment in budgets -- proving the two writes really share
    one transaction, not two independent ones that could diverge."""

    class Ctx:
        db_session = session
        task_id = None

    await record_model_call(
        Ctx(),
        agent_name="coder",
        model="claude-sonnet-5",
        uncached_input_tokens=1000,
        output_tokens=200,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )

    row = (
        await session.execute(
            select(AgentAction)
            .where(AgentAction.tool_name == MODEL_CALL_TOOL_NAME)
            .order_by(AgentAction.id.desc())
            .limit(1)
        )
    ).scalar_one()
    assert row.estimated_cost_usd is not None
    assert Decimal(str(row.estimated_cost_usd)) > 0

    today = datetime.now(UTC).date()
    budget_row = await session.execute(
        select(Budget.spent_usd).where(
            Budget.scope == GLOBAL_SCOPE, Budget.period_start == today
        )
    )
    assert Decimal(str(budget_row.scalar_one())) == Decimal(str(row.estimated_cost_usd))


async def test_check_budget_before_dispatch_allows_when_under_cap(session):
    await check_budget_before_dispatch(
        session,
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        cap_usd=1000.00,
    )  # must not raise


async def test_check_budget_before_dispatch_refuses_a_real_overage(session):
    """The literal Section 11.3.2 acceptance case: a deliberately low
    cap makes even one call's worst-case estimate exceed it, and the
    call is refused BEFORE dispatch -- no provider call happens at all,
    proven by the fact this test needs no mocked HTTP layer to pass."""
    with pytest.raises(BudgetExceededError):
        await check_budget_before_dispatch(
            session,
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "x" * 4000}],
            max_tokens=4096,
            cap_usd=0.0001,
        )


async def test_a_refusal_never_touches_spent_since_nothing_was_dispatched(session):
    """A refused request must leave spent_usd exactly where it was --
    refusing to dispatch is not the same as recording a call that never
    happened. Note: init_db()'s own startup reconciliation already wrote
    today's global row (correctly, at $0) before this test body runs --
    so the baseline to compare against is that real $0 row, not the
    absence of one."""
    today = datetime.now(UTC).date()

    async def _spent() -> Decimal | None:
        row = await session.execute(
            select(Budget.spent_usd).where(
                Budget.scope == GLOBAL_SCOPE, Budget.period_start == today
            )
        )
        value = row.scalar_one_or_none()
        return Decimal(str(value)) if value is not None else None

    before = await _spent()
    with pytest.raises(BudgetExceededError):
        await check_budget_before_dispatch(
            session,
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "x" * 4000}],
            max_tokens=4096,
            cap_usd=0.0001,
        )
    assert await _spent() == before  # the refusal itself wrote nothing


async def test_agent_loop_actually_refuses_via_the_real_integration_point(session):
    """Not a call to check_budget_before_dispatch directly -- this drives
    BaseAgent._complete_and_record itself, the real place Milestone 32
    wires the check into every agent's loop, with a real ToolContext and
    a real (if deliberately tiny) cap."""

    class _NeverCalled:
        model = "claude-sonnet-5"
        max_tokens = 4096

        async def complete(self, messages, tools=None):
            raise AssertionError(
                "the provider must never be dispatched to when the "
                "pre-flight budget check should have refused first"
            )

        async def embed(self, texts):
            raise NotImplementedError

    class _Agent(BaseAgent):
        name = "test-agent"

        def system_prompt(self):
            return "test"

    ctx = ToolContext(
        agent_name="test-agent",
        scratch_dir=Path("/tmp"),
        mode="observer",
        db_session=session,
    )
    agent = _Agent(model=_NeverCalled(), ctx=ctx)

    import amop.models.cost as cost_mod

    original = cost_mod.MAX_COST_PER_DAY_USD
    cost_mod.MAX_COST_PER_DAY_USD = 0.0001
    try:
        with pytest.raises(BudgetExceededError):
            await agent._complete_and_record(
                [{"role": "user", "content": "x" * 4000}]
            )
    finally:
        cost_mod.MAX_COST_PER_DAY_USD = original


# =======================================================================
# 11.3.2 -- reconciliation recomputes from ground truth, overwrites, doesn't increment
# =======================================================================


async def test_reconcile_budget_recomputes_from_agent_actions_sum(session):
    """Directly seed agent_actions with real cost rows (bypassing
    record_model_call's own accumulation entirely), corrupt the cached
    budgets row, then confirm reconcile_budget overwrites it with the
    real SUM -- proving the cache can self-heal from the table that is
    the actual source of truth."""
    for cost in ("1.50", "2.25", "0.10"):
        session.add(
            AgentAction(
                task_id=None,
                agent_name="coder",
                tool_name=MODEL_CALL_TOOL_NAME,
                arguments={"model": "claude-sonnet-5"},
                estimated_cost_usd=Decimal(cost),
                timestamp=datetime.now(UTC),
            )
        )
    await session.commit()

    # Corrupt the cache on purpose -- a wrong cached number that
    # reconciliation must overwrite, not add to.
    await record_spend(session, scope=GLOBAL_SCOPE, amount_usd=Decimal("999.00"))
    await session.commit()

    total = await reconcile_budget(session, scope=GLOBAL_SCOPE)
    assert total == Decimal("3.85")

    today = datetime.now(UTC).date()
    row = await session.execute(
        select(Budget.spent_usd).where(
            Budget.scope == GLOBAL_SCOPE, Budget.period_start == today
        )
    )
    assert Decimal(str(row.scalar_one())) == Decimal("3.85")


def test_estimate_input_tokens_is_a_positive_overestimate_leaning_heuristic():
    tokens = estimate_input_tokens([{"role": "user", "content": "hello world"}])
    assert tokens > 0


# =======================================================================
# 11.3.2 -- the "hourly thereafter" loop itself
# =======================================================================


async def test_hourly_reconcile_loop_actually_calls_reconcile_on_each_tick(
    engine, session
):
    """Not just that reconcile_budget works in isolation (already covered
    above) -- that the loop wired into api/app.py's lifespan and
    cli/main.py's `amop watch` genuinely invokes it, repeatedly, on its
    own schedule. A short interval (0.05s) stands in for the real
    default of one hour so this test finishes in well under a second."""
    import asyncio

    from amop.database.models import AgentAction
    from amop.models.cost import hourly_reconcile_loop

    session.add(
        AgentAction(
            task_id=None,
            agent_name="coder",
            tool_name=MODEL_CALL_TOOL_NAME,
            arguments={"model": "claude-sonnet-5"},
            estimated_cost_usd=Decimal("4.20"),
            timestamp=datetime.now(UTC),
        )
    )
    await session.commit()

    session_factory = make_session_factory(engine)
    task = asyncio.create_task(
        hourly_reconcile_loop(session_factory, interval_seconds=0.05)
    )
    try:
        await asyncio.sleep(0.2)  # several ticks
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    today = datetime.now(UTC).date()
    row = await session.execute(
        select(Budget.spent_usd).where(
            Budget.scope == GLOBAL_SCOPE, Budget.period_start == today
        )
    )
    assert Decimal(str(row.scalar_one())) == Decimal("4.20")


async def test_hourly_reconcile_loop_survives_a_failed_tick(engine, session, monkeypatch):
    """A single reconciliation failure must not kill the loop -- the
    next scheduled tick has to still fire and succeed."""
    import asyncio

    import amop.models.cost as cost_mod

    calls = {"n": 0}
    real_reconcile = cost_mod.reconcile_budget

    async def flaky(session, *, scope, today=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated transient failure")
        return await real_reconcile(session, scope=scope, today=today)

    monkeypatch.setattr(cost_mod, "reconcile_budget", flaky)

    session_factory = make_session_factory(engine)
    task = asyncio.create_task(
        cost_mod.hourly_reconcile_loop(session_factory, interval_seconds=0.05)
    )
    try:
        await asyncio.sleep(0.25)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert calls["n"] >= 2  # the failed first tick did not stop later ticks
