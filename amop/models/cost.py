"""Real cost accounting — spec Section 11.3, 11.3.1, 11.3.2.

Closes the gap `circuit_breakers.py` and `agents/base.py` both already
named explicitly: `AgentAction.estimated_cost_usd` has existed as a
column since Milestone 23 and stayed permanently NULL, because nothing
ever computed it from real telemetry. Section 29.4 calls this the
single highest-value finding across the spec's own review history.

**This is a second, independent enforcement layer, not a replacement
for the existing one.** `safety/circuit_breakers.py::check_cost_breaker`
(Milestone 9) is a coarser, task-creation-time gate against a flat
per-task guess (`task_context["cost_estimate_usd"]`) — its own docstring
already states plainly that real token-based costing was out of that
milestone's scope. This module builds the real thing alongside it:
per-model-call, telemetry-based, enforced at dispatch time (D-18). Both
read/write the same `AMOP_MAX_COST_PER_DAY_USD` cap so there is one
number an operator configures, but two gates checking it at different
granularities. The old breaker is untouched.

Design Decision D-16's formula (11.3.1):
    estimated_cost_usd = (cache_creation_input_tokens * cache_write_rate)
                        + (cache_read_input_tokens   * cache_read_rate)
                        + (uncached_input_tokens     * base_input_rate)
                        + (output_tokens             * output_rate)

Design Decision D-18 (11.3.2): the budget check happens at DISPATCH time,
before a call is issued, using a deliberate worst-case estimate (actual
input token count, real for a call already returned but only
*approximated* pre-dispatch since real tokenization isn't known until
the provider responds -- reusing this codebase's existing `len // 4`
sizing heuristic, precedented in Milestone 10's context-size work).
Cost is only knowable *after* a call returns, so an already-in-flight
request is NEVER cancelled on a budget trip -- its real cost is recorded
and the overshoot is accepted and logged, per the spec's own "soft
ceiling with a bounded overshoot, not a hard cutoff" framing.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from amop.database.models import AgentAction, Budget
from amop.safety.circuit_breakers import MAX_COST_PER_DAY_USD

logger = logging.getLogger("amop.models.cost")

GLOBAL_SCOPE = "global"

# 11.3.2: "recomputed from SUM(agent_actions.estimated_cost_usd) on
# orchestrator startup and hourly thereafter." Configurable so a demo or
# test can prove the loop actually fires without waiting a real hour.
RECONCILE_INTERVAL_SECONDS = float(
    os.environ.get("AMOP_BUDGET_RECONCILE_INTERVAL_SECONDS", "3600")
)


def task_scope(task_id) -> str:
    return f"task:{task_id}"


@dataclass(frozen=True)
class TokenRates:
    """USD per single token (not per-million) -- per-million published
    rates divided by 1e6 once, here, so every call site multiplies by a
    plain token count with no repeated division to get wrong."""

    base_input: float
    cache_write: float
    cache_read: float
    output: float


# Published per-1M-token USD rates, divided down to per-token. These are
# real numbers as of this module's writing, not placeholders -- but
# provider list pricing changes over time and this table is the one
# place to update when it does; nothing else in this module hardcodes a
# rate. Anthropic bills cache WRITES at a premium over base input;
# OpenAI's prompt caching is automatic with no write premium (only reads
# are discounted), which is why the OpenAI family's cache_write equals
# its base_input rather than a separate higher number -- a real
# difference between the two providers' pricing models, not an
# oversight.
_RATE_TABLE_PER_MILLION: dict[str, dict[str, float]] = {
    "claude-opus": {"base_input": 15.00, "cache_write": 18.75, "cache_read": 1.50, "output": 75.00},
    "claude-sonnet": {"base_input": 3.00, "cache_write": 3.75, "cache_read": 0.30, "output": 15.00},
    "claude-haiku": {"base_input": 0.80, "cache_write": 1.00, "cache_read": 0.08, "output": 4.00},
    "gpt-4o": {"base_input": 2.50, "cache_write": 2.50, "cache_read": 1.25, "output": 10.00},
    "gpt-4": {"base_input": 30.00, "cache_write": 30.00, "cache_read": 15.00, "output": 60.00},
    "gpt-3.5": {"base_input": 0.50, "cache_write": 0.50, "cache_read": 0.25, "output": 1.50},
}

_ZERO_RATES = TokenRates(0.0, 0.0, 0.0, 0.0)
_warned_unknown_models: set[str] = set()


def rates_for_model(model: str) -> TokenRates:
    """Resolve a model string to its rate row. Longest-prefix-style match
    against the table's keys (checked longest-key-first so e.g.
    "claude-opus" is never shadowed by a shorter, coincidentally-matching
    key) -- real provider model strings carry version suffixes
    (`claude-sonnet-4-5-20250514`, `gpt-4o-2024-08-06`) this must still
    match.

    An unmatched model (Ollama's qwen2.5-coder:14b, nomic-embed-text, or
    any future local model) is genuinely free and correctly returns zero
    rates -- but an unmatched PAID-looking model name would silently
    under-report real spend, which is worse than a loud placeholder. So
    anything not in the table AND not recognizable as a known-free local
    provider is logged once, loudly, rather than assumed free.
    """
    lowered = model.lower()
    for key in sorted(_RATE_TABLE_PER_MILLION, key=len, reverse=True):
        if key in lowered:
            row = _RATE_TABLE_PER_MILLION[key]
            return TokenRates(
                base_input=row["base_input"] / 1_000_000,
                cache_write=row["cache_write"] / 1_000_000,
                cache_read=row["cache_read"] / 1_000_000,
                output=row["output"] / 1_000_000,
            )
    # Ollama's real model names all look like "name:tag" (qwen2.5-coder:14b,
    # nomic-embed-text) -- never resembling a claude-*/gpt-* paid family.
    if "claude" in lowered or "gpt" in lowered:
        if model not in _warned_unknown_models:
            _warned_unknown_models.add(model)
            logger.warning(
                "cost: model %r looks like a paid provider but has no entry "
                "in _RATE_TABLE_PER_MILLION -- treating as $0 rather than "
                "guessing, but real spend on this model will under-report "
                "until a rate row is added",
                model,
            )
    return _ZERO_RATES


def estimate_cost_usd(
    model: str,
    *,
    cache_creation_input_tokens: int | None,
    cache_read_input_tokens: int | None,
    uncached_input_tokens: int | None,
    output_tokens: int | None,
) -> Decimal:
    """11.3.1's formula, applied to one real completed call's telemetry.
    None counts (a provider that didn't report a field) are treated as
    zero for that term -- a provider truly reporting nothing looks
    identical to a call that used none of that token class, which is the
    same "missing means zero, not unknown" convention every other
    None-tolerant field in ModelResponse already uses.
    """
    rates = rates_for_model(model)
    total = (
        (cache_creation_input_tokens or 0) * rates.cache_write
        + (cache_read_input_tokens or 0) * rates.cache_read
        + (uncached_input_tokens or 0) * rates.base_input
        + (output_tokens or 0) * rates.output
    )
    return Decimal(str(round(total, 8)))


def estimate_input_tokens(messages: list[dict]) -> int:
    """Pre-dispatch approximation of a request's input token count --
    real tokenization isn't known until the provider responds, so this
    reuses the same `len(text) // 4` sizing heuristic Milestone 10's
    context-size work already established for this codebase, applied to
    the serialized message content. Deliberately an overestimate-leaning
    heuristic (counts JSON structure characters too, not just prose) --
    for a WORST-CASE pre-dispatch bound (D-18), erring toward "counts a
    few more tokens than reality" is the safe direction, not the
    dangerous one.
    """
    import json

    serialized = json.dumps(messages, ensure_ascii=False)
    return max(1, len(serialized) // 4)


def max_plausible_cost_usd(model: str, *, input_tokens_estimate: int, max_tokens: int) -> Decimal:
    """D-18's worst-case pre-dispatch estimate: treats the ENTIRE input
    as uncached (the most expensive real classification it could turn
    out to be) and assumes the call generates the full max_tokens of
    output. Real cost, computed after the call returns from
    `estimate_cost_usd`, is almost always lower than this -- that
    asymmetry is the whole point: the cap can never be defeated by an
    unexpectedly long completion, because the estimate already assumed
    the longest one possible.
    """
    rates = rates_for_model(model)
    total = input_tokens_estimate * rates.base_input + max_tokens * rates.output
    return Decimal(str(round(total, 8)))


class BudgetExceededError(RuntimeError):
    """Raised by check_budget_before_dispatch. A real refusal, not a
    warning -- the caller (agents/base.py) lets this propagate as an
    agent-run failure, the same way any other pre-flight refusal does."""


async def _spent_today(session: AsyncSession, scope: str, today: date) -> Decimal:
    row = await session.execute(
        select(Budget.spent_usd).where(
            Budget.scope == scope, Budget.period_start == today
        )
    )
    value = row.scalar_one_or_none()
    return Decimal(str(value)) if value is not None else Decimal("0")


async def check_budget_before_dispatch(
    session: AsyncSession,
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    cap_usd: float | None = None,
    scope: str = GLOBAL_SCOPE,
) -> None:
    """11.3.2 / D-18: refuse BEFORE dispatch if today's real spend plus
    this request's worst-case cost would exceed the cap. Never cancels
    an in-flight request -- this only ever runs before one starts.

    Deliberately no lock on the read: D-18's own text accepts a bounded
    overshoot from concurrent in-flight requests each passing the check
    against the same not-yet-updated `spent_usd` ("a soft ceiling with a
    bounded overshoot, not a hard cutoff") -- serializing this read would
    turn a documented, accepted tradeoff into unnecessary lock
    contention on every single model call.

    `cap_usd` is resolved from the module-level `MAX_COST_PER_DAY_USD`
    at CALL time (a name lookup in the function body), not bound as a
    default-argument value evaluated once at import time -- the same
    `AMOP_MAX_COST_PER_DAY_USD` env var every other breaker in this
    codebase reads once at process start, but kept re-readable here
    rather than frozen into a signature, since callers (and tests) that
    reassign this module's `MAX_COST_PER_DAY_USD` attribute directly
    expect that to actually take effect.
    """
    if cap_usd is None:
        cap_usd = MAX_COST_PER_DAY_USD
    today = datetime.now(UTC).date()
    spent = await _spent_today(session, scope, today)
    worst_case = max_plausible_cost_usd(
        model,
        input_tokens_estimate=estimate_input_tokens(messages),
        max_tokens=max_tokens,
    )
    projected = spent + worst_case
    if projected > Decimal(str(cap_usd)):
        message = (
            f"budget cap (${cap_usd:.4f}, scope={scope!r}) would be exceeded: "
            f"${spent:.4f} already spent today + ${worst_case:.4f} worst-case "
            f"for this call = ${projected:.4f}"
        )
        logger.warning("cost: dispatch refused -- %s", message)
        raise BudgetExceededError(message)


async def record_spend(
    session: AsyncSession, *, scope: str, amount_usd: Decimal, today: date | None = None
) -> None:
    """Atomic accumulation -- a single `INSERT ... ON CONFLICT DO UPDATE
    SET spent_usd = budgets.spent_usd + EXCLUDED.spent_usd`, never a
    read-then-write. Two concurrent calls both crediting the same
    scope+day cannot lose an update to each other the way a
    read-modify-write in application code could -- Postgres serializes
    the two UPDATEs, and each sees the other's committed delta.

    Caller is responsible for running this inside the SAME transaction
    as the `agent_actions` row it corresponds to (audit/actions.py) --
    this function itself does not commit.
    """
    today = today or datetime.now(UTC).date()
    stmt = pg_insert(Budget).values(
        scope=scope, period_start=today, spent_usd=amount_usd
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[Budget.scope, Budget.period_start],
        set_={"spent_usd": Budget.spent_usd + stmt.excluded.spent_usd},
    )
    await session.execute(stmt)


async def reconcile_budget(
    session: AsyncSession, *, scope: str, today: date | None = None
) -> Decimal:
    """11.3.2's reconciliation: recompute the AUTHORITATIVE total directly
    from `SUM(agent_actions.estimated_cost_usd)`, and overwrite (not
    increment) `budgets.spent_usd` with it. Run at startup and hourly --
    this is what keeps `spent_usd` from drifting from ground truth if a
    crash ever loses an in-flight `record_spend` before it lands (the
    accumulator is a cache; this table's real source of truth is always
    `agent_actions`, per spec 11.3.2's own framing: "the counter is a
    cache, the table is the truth").
    """
    today = today or datetime.now(UTC).date()
    day_start = datetime(today.year, today.month, today.day, tzinfo=UTC)

    query = select(AgentAction.estimated_cost_usd).where(
        AgentAction.timestamp >= day_start
    )
    if scope == GLOBAL_SCOPE:
        pass
    elif scope.startswith("task:"):
        query = query.where(AgentAction.task_id == scope.removeprefix("task:"))
    else:
        raise ValueError(f"unrecognized budget scope: {scope!r}")

    rows = (await session.execute(query)).scalars().all()
    total = sum((Decimal(str(v)) for v in rows if v is not None), Decimal("0"))

    stmt = pg_insert(Budget).values(scope=scope, period_start=today, spent_usd=total)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Budget.scope, Budget.period_start],
        set_={"spent_usd": total},
    )
    await session.execute(stmt)
    await session.commit()
    return total


async def hourly_reconcile_loop(
    session_factory,
    *,
    interval_seconds: float | None = None,
    scope: str = GLOBAL_SCOPE,
) -> None:
    """11.3.2's "hourly thereafter" half. `init_db()` (database/session.py)
    already covers "on startup" for every one of AMOP's real entry
    points; this covers the two processes that actually stay alive long
    enough for "hourly" to mean anything -- `amop serve-api` (wired in
    api/app.py's lifespan) and `amop watch` (spec 9.7.1 already calls
    this "the closest real analog to orchestrator startup" for the same
    reason; wired in cli/main.py's poll loop).

    Runs until cancelled (the caller's asyncio.CancelledError propagates
    normally on shutdown -- this function does not catch it). A single
    reconciliation failure is logged and does not kill the loop; the
    next scheduled tick tries again, same "never take down the run that
    produced it" posture as every other audit-adjacent write in this
    codebase.
    """
    interval = interval_seconds if interval_seconds is not None else RECONCILE_INTERVAL_SECONDS
    while True:
        await asyncio.sleep(interval)
        try:
            async with session_factory() as session:
                total = await reconcile_budget(session, scope=scope)
            logger.info(
                "cost: hourly reconciliation for scope=%r -> $%s", scope, total
            )
        except Exception:  # noqa: BLE001 -- see docstring
            logger.exception(
                "cost: hourly reconciliation failed for scope=%r -- will retry "
                "on the next tick",
                scope,
            )
