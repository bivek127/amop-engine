"""Milestone 22 — Audit Log Hash-Chaining (spec 12.6 / ADR-16).

The success criterion is binary: either tampering is provably detectable
or it isn't. So every test here that claims detection was first run
against code mutated to remove the mechanism it tests, and observed
failing — per CLAUDE.md's standing rule.

The last test in this file deliberately asserts that tampering is NOT
detected. That is not a gap in the suite; it is the milestone's honest
documentation of its own limit (ADR-16: "detection, not prevention —
stated plainly rather than overclaimed"), pinned as an executable fact
so nobody later mistakes this mechanism for immutability.

Requires: Postgres at TEST_DATABASE_URL.
"""

import asyncio
import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import select, text

from amop.audit.chain import (
    GENESIS_HASH,
    backfill_chain,
    canonical_content,
    chain_tip,
    compute_row_hash,
    hash_for_row,
    verify_chain,
)
from amop.database.models import TaskTransition
from amop.database.session import init_db, make_engine, make_session_factory
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import create_task, transition

TEST_DATABASE_URL = "postgresql+asyncpg://localhost/amop_test"


@pytest_asyncio.fixture
async def engine():
    eng = make_engine(TEST_DATABASE_URL)
    await init_db(eng)
    async with eng.begin() as conn:
        await conn.execute(
            # agent_actions included since Milestone 23: the hash chain
            # now SPANS both tables, so a test asserting chain state has
            # to control both. Leaving stray agent_actions rows here made
            # every clean-chain assertion in this file fail.
            text(
                "TRUNCATE agent_actions, task_transitions, tasks "
                "RESTART IDENTITY CASCADE"
            )
        )
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return make_session_factory(engine)


@pytest_asyncio.fixture
async def session(session_factory):
    async with session_factory() as s:
        yield s


async def _task_with_transitions(session, count=3):
    task = await create_task(session, task_type="bug_fix", task_context={})
    path = [
        TaskState.TRIAGING,
        TaskState.INVESTIGATING,
        TaskState.PLANNING_FIX,
        TaskState.CODING,
        TaskState.TESTING,
    ][:count]
    for state in path:
        await transition(session, task, state)
    return task


# ---------------------------------------------------------------------
# Canonicalization -- pure, no DB. The subtle correctness that makes the
# whole chain verifiable in practice rather than only in principle.
# ---------------------------------------------------------------------


def test_canonical_content_is_stable_regardless_of_argument_order():
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    tid = uuid.uuid4()
    a = canonical_content(
        task_id=tid, from_state="A", to_state="B", trigger="t", actor="x", timestamp=ts
    )
    b = canonical_content(
        timestamp=ts, actor="x", trigger="t", to_state="B", from_state="A", task_id=tid
    )
    assert a == b


def test_none_and_empty_string_do_not_collide():
    """The subtlest correctness point in the whole module. If None
    collapsed to "", swapping `trigger=None` for `trigger=""` in a
    historical row would be an undetectable edit."""
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    tid = uuid.uuid4()
    with_none = canonical_content(tid, "A", "B", None, "x", ts)
    with_empty = canonical_content(tid, "A", "B", "", "x", ts)
    assert with_none != with_empty
    assert compute_row_hash(GENESIS_HASH, with_none) != compute_row_hash(
        GENESIS_HASH, with_empty
    )


def test_concatenation_is_unambiguous_across_the_boundary():
    """Plain `prev || content` would let ("ab","c") and ("a","bc") hash
    identically. The separator byte is what prevents a crafted row from
    impersonating a different (prev, content) split."""
    assert compute_row_hash("ab", "c") != compute_row_hash("a", "bc")


# ---------------------------------------------------------------------
# Round-trip: hash-on-write must equal hash-on-verify-after-a-real-read.
# ---------------------------------------------------------------------


async def test_hashes_survive_a_real_postgres_round_trip(session, session_factory):
    """Pinned explicitly because a silent precision mismatch here (PG
    TIMESTAMPTZ vs Python datetime) would make every chain unverifiable
    while each half looked correct in isolation -- the same shape of bug
    as Milestone 20's normalize-on-read-but-not-write."""
    await _task_with_transitions(session, count=3)

    # A FRESH session, so values come from Postgres rather than the
    # identity map that wrote them.
    async with session_factory() as fresh:
        rows = list(
            (
                await fresh.execute(select(TaskTransition).order_by(TaskTransition.id))
            ).scalars().all()
        )
        assert len(rows) == 3
        prev = GENESIS_HASH
        for row in rows:
            assert hash_for_row(row, prev) == row.row_hash
            assert row.timestamp.tzinfo is not None
            prev = row.row_hash


# ---------------------------------------------------------------------
# Clean chain / construction.
# ---------------------------------------------------------------------


async def test_a_clean_chain_verifies_intact(session):
    await _task_with_transitions(session, count=5)
    result = await verify_chain(session)
    assert result.intact
    assert result.checked == 5
    assert result.first_divergence_id is None
    assert result.tip_hash


async def test_the_first_row_chains_from_genesis(session):
    await _task_with_transitions(session, count=1)
    row = (
        await session.execute(select(TaskTransition).order_by(TaskTransition.id))
    ).scalars().first()
    assert row.prev_hash == GENESIS_HASH


async def test_each_row_chains_to_its_predecessor(session):
    await _task_with_transitions(session, count=4)
    rows = list(
        (
            await session.execute(select(TaskTransition).order_by(TaskTransition.id))
        ).scalars().all()
    )
    for earlier, later in zip(rows, rows[1:]):
        assert later.prev_hash == earlier.row_hash


async def test_an_empty_chain_verifies_intact(session):
    result = await verify_chain(session)
    assert result.intact
    assert result.checked == 0


# ---------------------------------------------------------------------
# Tamper detection -- the actual deliverable. Every mutation here is
# direct SQL, bypassing the application entirely, which is exactly what
# a DB-level attacker has.
# ---------------------------------------------------------------------


async def test_a_directly_mutated_row_is_detected_at_the_exact_row(session):
    await _task_with_transitions(session, count=5)
    ids = list(
        (
            await session.execute(select(TaskTransition.id).order_by(TaskTransition.id))
        ).scalars().all()
    )
    target = ids[2]  # a middle row -- not first, not last

    # Straight UPDATE. No application code path touches the row.
    await session.execute(
        text("UPDATE task_transitions SET actor = :a WHERE id = :i"),
        {"a": "human:attacker", "i": target},
    )
    await session.commit()
    session.expire_all()

    result = await verify_chain(session)
    assert not result.intact
    assert result.first_divergence_id == target, (
        f"divergence reported at {result.first_divergence_id}, expected the "
        f"actually-altered row {target}"
    )
    assert result.checked == 2, "should verify the 2 rows before the tampered one"
    assert "altered after it was written" in result.reason


async def test_a_null_to_value_mutation_is_detected(session):
    """NULL handling is the subtlest part of the canonicalization, so it
    gets its own direct-SQL case rather than being assumed covered."""
    task = await create_task(session, task_type="bug_fix", task_context={})
    # trigger=None is only reachable by writing the row directly, since
    # transition() always substitutes the spec trigger.
    row = TaskTransition(
        task_id=task.id,
        from_state="CREATED",
        to_state="TRIAGING",
        trigger=None,
        actor=None,
        timestamp=datetime.now(UTC),
    )
    from amop.audit.chain import append_chained

    await append_chained(session, row)
    await session.commit()
    target = row.id

    await session.execute(
        text("UPDATE task_transitions SET trigger = '' WHERE id = :i"), {"i": target}
    )
    await session.commit()
    session.expire_all()

    result = await verify_chain(session)
    assert not result.intact
    assert result.first_divergence_id == target


async def test_a_deleted_row_breaks_the_chain(session):
    """Deletion is tampering too -- the surviving rows' prev_hash no
    longer matches the row that now precedes them."""
    await _task_with_transitions(session, count=5)
    ids = list(
        (
            await session.execute(select(TaskTransition.id).order_by(TaskTransition.id))
        ).scalars().all()
    )
    await session.execute(
        text("DELETE FROM task_transitions WHERE id = :i"), {"i": ids[2]}
    )
    await session.commit()
    session.expire_all()

    result = await verify_chain(session)
    assert not result.intact
    assert result.first_divergence_id == ids[3], "the row after the deletion diverges"
    # Wording widened in Milestone 23 to name deletion explicitly, since
    # that is the case this branch most often reports.
    assert "broken, reordered, or rows were deleted" in result.reason


# ---------------------------------------------------------------------
# Concurrency -- the part the brief singled out. Milestone 16/20 style:
# real concurrent writers, real connections, assert linearity.
# ---------------------------------------------------------------------


async def test_concurrent_appends_produce_a_linear_verifiable_chain(session_factory):
    """Without the advisory lock the chain FORKS: concurrent writers all
    read the same tip and all claim it as prev_hash. Asserting linearity
    (each prev_hash equals exactly one predecessor's row_hash, no
    duplicates) is what catches that -- merely asserting "N rows exist"
    would pass against a forked chain."""
    task_ids = []
    async with session_factory() as s:
        for _ in range(12):
            t = await create_task(s, task_type="bug_fix", task_context={})
            task_ids.append(t.id)

    async def do_transition(task_id):
        # Each writer gets its own session, so its own connection --
        # genuine concurrency, not interleaved coroutines on one.
        async with session_factory() as s:
            from amop.database.models import Task

            t = await s.get(Task, task_id)
            await transition(s, t, TaskState.TRIAGING)

    await asyncio.gather(*(do_transition(tid) for tid in task_ids))

    async with session_factory() as s:
        rows = list(
            (
                await s.execute(select(TaskTransition).order_by(TaskTransition.id))
            ).scalars().all()
        )
        assert len(rows) == 12

        # No two rows may claim the same predecessor -- that is precisely
        # what a fork looks like.
        prevs = [r.prev_hash for r in rows]
        assert len(set(prevs)) == len(prevs), "chain forked: duplicate prev_hash"

        result = await verify_chain(s)
        assert result.intact, result.reason
        assert result.checked == 12


# ---------------------------------------------------------------------
# Backfill.
# ---------------------------------------------------------------------


async def test_backfill_chains_pre_existing_unchained_rows(session, session_factory):
    task = await create_task(session, task_type="bug_fix", task_context={})
    # Simulate rows written before Milestone 22 existed.
    await session.execute(
        text(
            "INSERT INTO task_transitions (task_id, from_state, to_state, trigger, "
            "actor, timestamp) VALUES (:t, 'CREATED', 'TRIAGING', 'x', 'system', now())"
        ),
        {"t": task.id},
    )
    await session.commit()

    before = await verify_chain(session)
    assert before.unchained == 1

    # Returns (newly_chained, legacy_positioned) since Milestone 23.
    filled, _positioned = await backfill_chain(session)
    assert filled == 1

    session.expire_all()
    after = await verify_chain(session)
    assert after.intact
    assert after.unchained == 0
    assert after.checked == 1


async def test_a_backfilled_chain_still_detects_later_tampering(session):
    """Backfill's own value is limited (rows are attested as of backfill
    time, not write time) -- but it must at least make subsequent
    tampering detectable, which is what this pins."""
    task = await create_task(session, task_type="bug_fix", task_context={})
    await session.execute(
        text(
            "INSERT INTO task_transitions (task_id, from_state, to_state, trigger, "
            "actor, timestamp) VALUES (:t, 'CREATED', 'TRIAGING', 'x', 'system', now())"
        ),
        {"t": task.id},
    )
    await session.commit()
    await backfill_chain(session)

    row_id = (
        await session.execute(select(TaskTransition.id).order_by(TaskTransition.id))
    ).scalars().first()
    await session.execute(
        text("UPDATE task_transitions SET actor = 'human:attacker' WHERE id = :i"),
        {"i": row_id},
    )
    await session.commit()
    session.expire_all()

    result = await verify_chain(session)
    assert not result.intact
    assert result.first_divergence_id == row_id


# ---------------------------------------------------------------------
# The documented LIMIT. This test asserts that tampering is NOT
# detected, on purpose.
# ---------------------------------------------------------------------


async def test_forward_recomputation_defeats_verification_entirely(session):
    """ADR-16's own honesty, made executable.

    An attacker with DB write access alters a row AND recomputes every
    subsequent hash. `verify_chain()` reports INTACT, because it has
    nothing outside the database to compare against. This is tamper
    EVIDENCE, not tamper PREVENTION, and this test exists so that
    distinction can never quietly erode into a claim of immutability.

    The only in-scope mitigation is the tip hash: it necessarily changes
    under this attack, so an operator who recorded it externally can
    still catch it -- asserted below.
    """
    await _task_with_transitions(session, count=5)
    tip_before = await chain_tip(session)

    rows = list(
        (
            await session.execute(select(TaskTransition).order_by(TaskTransition.id))
        ).scalars().all()
    )
    target = rows[2]

    # 1. Alter the row's content.
    await session.execute(
        text("UPDATE task_transitions SET actor = :a WHERE id = :i"),
        {"a": "human:attacker", "i": target.id},
    )
    await session.commit()
    session.expire_all()

    # 2. Recompute every hash from that point forward -- exactly what an
    #    attacker holding this module's own source would do.
    rows = list(
        (
            await session.execute(select(TaskTransition).order_by(TaskTransition.id))
        ).scalars().all()
    )
    prev = GENESIS_HASH
    for row in rows:
        row.prev_hash = prev
        row.row_hash = hash_for_row(row, prev)
        session.add(row)
        prev = row.row_hash
    await session.commit()
    session.expire_all()

    result = await verify_chain(session)
    assert result.intact, (
        "This assertion documents a REAL LIMITATION, not a passing "
        "security property: forward-recomputation is undetectable by "
        "in-database verification alone."
    )

    # ...but the tip changed, which is what an externally-recorded tip
    # would catch. The mitigation is real, and it is entirely dependent
    # on the operator having actually saved that value.
    tip_after = await chain_tip(session)
    assert tip_after != tip_before, (
        "the tip must change under forward-recomputation -- otherwise the "
        "one available mitigation would not work either"
    )


# ---------------------------------------------------------------------
# Interaction with Milestone 16's OCC. Subtle enough to be worth its own
# pin rather than relying on M16's suite to keep catching it.
# ---------------------------------------------------------------------


async def test_a_lost_occ_race_raises_cleanly_and_leaves_the_chain_intact(
    session, session_factory
):
    """Regression pin for a real bug this milestone introduced and
    Milestone 16's exception-TYPE assertion caught.

    `append_chained()` issues session.execute() calls, and SQLAlchemy
    autoflushes pending changes before any query -- so the
    version-guarded UPDATE on `tasks` now fires INSIDE append_chained(),
    and a lost OCC race raises StaleDataError there rather than at
    commit(). With `transition()`'s try/except wrapped around commit()
    alone, that escaped as a raw StaleDataError and every caller that
    keys on ConcurrentUpdateError (the API's 409 mapping,
    transition_with_retry) silently stopped recognizing a conflict.

    Also asserts the audit-trail consequence, which is this milestone's
    own concern: the loser must leave NO audit row behind, or the chain
    would record a transition that never happened.
    """
    from amop.database.models import Task
    from amop.orchestrator.task import ConcurrentUpdateError

    task = await create_task(session, task_type="bug_fix", task_context={})
    for state in (
        TaskState.TRIAGING,
        TaskState.INVESTIGATING,
        TaskState.PLANNING_FIX,
        TaskState.CODING,
        TaskState.TESTING,
        TaskState.REVIEWING,
        TaskState.PR_CREATION,
        TaskState.WAITING_FOR_APPROVAL,
    ):
        await transition(session, task, state)
    task_id = task.id

    async with session_factory() as s:
        before = len(
            (await s.execute(select(TaskTransition))).scalars().all()
        )

    barrier = asyncio.Barrier(2)

    async def writer(to_state):
        async with session_factory() as s:
            row = await s.get(Task, task_id)
            await barrier.wait()  # both hold the same version before either writes
            return await transition(s, row, to_state, actor="human:test")

    outcomes = await asyncio.gather(
        writer(TaskState.MERGED), writer(TaskState.CANCELLED), return_exceptions=True
    )
    failures = [o for o in outcomes if isinstance(o, BaseException)]

    assert len(failures) == 1
    assert isinstance(failures[0], ConcurrentUpdateError), (
        f"loser raised {type(failures[0]).__name__} -- autoflush moved where "
        f"StaleDataError surfaces, and the OCC guard no longer covers it"
    )

    async with session_factory() as s:
        after = len((await s.execute(select(TaskTransition))).scalars().all())
        assert after == before + 1, "the losing writer left an audit row behind"
        result = await verify_chain(s)
        assert result.intact, result.reason
