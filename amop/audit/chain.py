"""Audit-log hash chaining — spec Section 12.6, Design Decision ADR-16.

`task_transitions` is "append-only at the application layer," which
Section 12.6 is careful to call "an accurate description of a
convention, not a guarantee." This module makes retroactive alteration
*detectable*: every row stores `sha256(prev_hash || canonical_content)`,
and `verify_chain()` recomputes the whole chain and reports the first
divergence.

WHAT THIS DOES NOT DO -- stated first, because ADR-16 explicitly rejects
the alternative framing as "a false assurance":

  * It does NOT make rows immutable. Nothing here prevents a write.
  * An attacker with database write access can alter a row AND recompute
    every subsequent `row_hash` forward. The result verifies as intact.
    `verify_chain()` cannot detect this, by construction -- it has
    nothing outside the database to compare against.
  * The only in-scope mitigation is `chain_tip()`: an operator who
    records the tip hash somewhere outside the database can detect
    exactly that attack, because a forward-recomputed chain necessarily
    produces a different tip. That depends entirely on the operator
    actually recording it; it is not automatic, and this module cannot
    make it so.

True immutability needs append-only external storage (WORM), which spec
29.6 puts post-MVP and Milestone 22 explicitly does not build.

SCOPE -- what is chained (Milestone 23):

  * `task_transitions` -- state-machine decisions (who moved a task
    between states, and why).
  * `agent_actions` -- the Safety Engine's ALLOW/DENY tool-call
    decisions. Section 12.6 calls this "the primary artifact... for
    post-incident review of what an agent actually attempted", and it
    was Milestone 22's top-priority unresolved gap.

Both are in ONE shared chain, never two. That is not a stylistic
preference: with separate chains, deleting every row of one table leaves
an empty chain that verifies as INTACT -- strictly worse than no
verification, because it converts "I don't know" into a false
"everything is fine". A shared chain also attests the interleaving (this
tool call happened between those two state changes), which separate
chains cannot express at all. Proven, not assumed:
tests/test_milestone23.py builds the rejected design and shows it
missing a real deletion.

Adding a third table means one more entry in CHAINED_TABLES below --
append/verify/backfill are already table-agnostic.

KNOWN LIMITS, all sharing one mitigation (an externally-recorded
`chain_tip()`), none of them detectable by in-database verification
alone:

  1. Forward recomputation -- alter a row, recompute every hash after it.
  2. Suffix truncation -- delete the most recent N rows across the whole
     chain; what remains is internally consistent and simply shorter.
  3. Backfilled rows are attested as of BACKFILL time, not write time.

Full reasoning in ROADMAP.md's "LOCKED DECISION" section.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from amop.database.models import AgentAction, TaskTransition

# The chain's first row has no predecessor. A fixed, explicit sentinel
# rather than an empty string or NULL: it makes "this is genuinely the
# start of the chain" distinguishable from "this row's prev_hash was
# lost/blanked", which an empty string would not.
GENESIS_HASH = "amop:genesis:v1"

# One global advisory-lock key for the whole chain. Deliberately unlike
# Milestone 16's per-repo/per-file keys: the chain is a single globally
# linear sequence, so every append must serialize against every other
# append, regardless of repo. Derived the same way M16 derives its keys
# (blake2b -> signed int64) so the value is stable across processes --
# Python's hash() is randomized per process and would silently stop
# excluding anything after a restart.
AUDIT_CHAIN_LOCK_KEY = int.from_bytes(
    hashlib.blake2b(b"amop:audit_chain", digest_size=8).digest(), "big", signed=True
)


def _serialize(fields: dict) -> str:
    """The one deterministic serializer every chained table shares.

    Milestone 23 generalized this out of `canonical_content()` so a
    second table could join the chain WITHOUT a second serializer. That
    matters more than it looks: two serializers that drift apart by a
    separator or a NULL convention produce a chain that verifies on the
    table it was written for and silently fails on the other. One
    function, one set of rules, both tables.
    """
    return json.dumps(
        fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def canonical_content(
    task_id: uuid.UUID | str,
    from_state: str | None,
    to_state: str,
    trigger: str | None,
    actor: str | None,
    timestamp: datetime,
) -> str:
    """A `task_transitions` row's content in one deterministic string.

    Determinism is the whole ballgame: if the same logical row can
    serialize two different ways, the chain is unverifiable in practice
    even though the concept is sound. So:

      * `sort_keys=True` -- field order can never drift with dict
        insertion order or a future refactor.
      * `separators=(",", ":")` -- no incidental whitespace.
      * NULLs serialize as JSON `null`, a distinct value from `""`. If
        None collapsed to an empty string, a row with `trigger=None` and
        one with `trigger=""` would hash identically, and swapping one
        for the other would be an undetectable edit.
      * `ensure_ascii=False` with an explicit UTF-8 encode at hash time,
        so non-ASCII content hashes by its actual bytes rather than by
        an escaping convention that could change.

    `id` is deliberately NOT included: it's assigned by the BIGSERIAL
    sequence during INSERT, so it isn't known when the hash is computed.
    Chain position is carried by `prev_hash` instead, which is the
    stronger property anyway -- it pins each row to its predecessor's
    full content, not merely to a number.
    """
    return _serialize(
        {
            "task_id": str(task_id),
            "from_state": from_state,
            "to_state": to_state,
            "trigger": trigger,
            "actor": actor,
            "timestamp": timestamp.isoformat(),
        }
    )


def agent_action_content(
    task_id,
    agent_name: str | None,
    tool_name: str | None,
    arguments: dict | None,
    result: str | None,
    decision: str | None,
    decision_reason: str | None,
    timestamp: datetime,
) -> str:
    """An `agent_actions` row's content (Milestone 23).

    `latency_ms` and `estimated_cost_usd` are deliberately EXCLUDED from
    the hash. They are measurements about the call, not claims about what
    the agent attempted or what the Safety Engine decided -- the things
    Section 12.6 says this record exists to preserve. Including latency
    would also make the hash depend on machine timing, so a legitimate
    re-measurement could look like tampering. What IS covered: who called
    what, with which arguments, and whether it was allowed and why.
    """
    return _serialize(
        {
            "task_id": str(task_id) if task_id is not None else None,
            "agent_name": agent_name,
            "tool_name": tool_name,
            # sort_keys applies recursively, so a nested arguments dict
            # serializes identically regardless of insertion order.
            "arguments": arguments,
            "result": result,
            "decision": decision,
            "decision_reason": decision_reason,
            "timestamp": timestamp.isoformat(),
        }
    )


def compute_row_hash(prev_hash: str, content: str) -> str:
    """Section 12.6's own formula: sha256(prev_hash || canonical_content).

    The `\\x1e` (ASCII record separator) between the two parts is not in
    the spec's literal notation but matters: plain concatenation is
    ambiguous when either side's length can vary, so ("ab", "c") and
    ("a", "bc") would hash identically. A byte that cannot appear in a
    hex digest removes that ambiguity.
    """
    return hashlib.sha256(f"{prev_hash}\x1e{content}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChainedTable:
    """One table participating in the shared chain.

    Milestone 23's generalization of Milestone 22's TaskTransition-only
    plumbing: a model plus the function that turns one of its rows into
    canonical content. Adding a third table later means adding one
    entry to CHAINED_TABLES below -- not touching append/verify/backfill.
    """

    name: str
    model: type
    content_of: object  # Callable[[row], str]


def _transition_content(row) -> str:
    return canonical_content(
        task_id=row.task_id,
        from_state=row.from_state,
        to_state=row.to_state,
        trigger=row.trigger,
        actor=row.actor,
        timestamp=row.timestamp,
    )


def _agent_action_content(row) -> str:
    return agent_action_content(
        task_id=row.task_id,
        agent_name=row.agent_name,
        tool_name=row.tool_name,
        arguments=row.arguments,
        result=row.result,
        decision=row.decision,
        decision_reason=row.decision_reason,
        timestamp=row.timestamp,
    )


TRANSITIONS_TABLE = ChainedTable(
    "task_transitions", TaskTransition, _transition_content
)
AGENT_ACTIONS_TABLE = ChainedTable("agent_actions", AgentAction, _agent_action_content)

# THE shared chain's membership. One chain across all of these -- never
# one chain each. See this module's docstring for why (a per-table chain
# lets wholesale deletion of a table verify as INTACT).
CHAINED_TABLES: tuple[ChainedTable, ...] = (TRANSITIONS_TABLE, AGENT_ACTIONS_TABLE)


def _table_for(row) -> ChainedTable:
    for table in CHAINED_TABLES:
        if isinstance(row, table.model):
            return table
    raise TypeError(f"{type(row).__name__} is not a chained table")


def hash_for_row(row, prev_hash: str) -> str:
    """Recompute a persisted row's hash -- the verification side of the
    same function `append_chained()` uses on the write side. Both go
    through the row's own `content_of` so the two can never drift
    apart, whichever table the row belongs to."""
    return compute_row_hash(prev_hash, _table_for(row).content_of(row))


async def _lock_chain(session: AsyncSession) -> None:
    """Serialize appends on the session's OWN transaction.

    Without this the chain forks: two concurrent writers both read the
    same tip, both insert with the same `prev_hash`, and the sequence is
    no longer linear. `pg_advisory_xact_lock` is held from here until
    this transaction commits or rolls back -- which is exactly the
    window that must be exclusive (read tip -> insert).

    Deliberately on the session's own connection, unlike Milestone 16's
    repo locks, which needed a dedicated one precisely BECAUSE they had
    to survive `transition()`'s commits. Here the opposite is wanted:
    release at commit, automatically.

    Deadlock-free by construction, worth stating explicitly rather than
    hoping: this is a LEAF lock. Nothing is acquired while holding it --
    between this call and the commit, `transition()` does only pure
    validation, attribute assignment, and the INSERT. So M16's repo lock
    -> audit lock is the only ordering that ever occurs, and the reverse
    never does.
    """
    await session.execute(
        text("SELECT pg_advisory_xact_lock(:k)"), {"k": AUDIT_CHAIN_LOCK_KEY}
    )


async def _current_tip_hash(session: AsyncSession) -> str:
    """The newest chained row's hash, or GENESIS_HASH if none exists.

    Ordered by `chain_pos DESC` ACROSS EVERY chained table, not by a
    per-table `id`: since Milestone 23 the chain spans two tables, and
    per-table BIGSERIAL cannot order rows between them. `chain_pos` is
    drawn from one shared sequence inside the append lock, so it is the
    only ordering that reflects true append order across the whole
    chain. Not `timestamp` either -- those are Python-assigned and two
    rows can legitimately share one.
    """
    best_key, best_hash = None, None
    for table in CHAINED_TABLES:
        model = table.model
        # Same composite ordering `_all_rows_in_chain_order` uses,
        # expressed in SQL so this stays two indexed queries rather than
        # loading every row on every append: rows with a NULL chain_pos
        # (pre-Milestone-23 legacy) sort FIRST, ordered by id; positioned
        # rows follow, ordered by chain_pos.
        #
        # Including NULL-chain_pos rows here is not a detail. Found live
        # while migrating the real dev database: when this only looked at
        # positioned rows, a new row written BEFORE `audit backfill` had
        # run found no tip at all, chained itself from GENESIS, and then
        # landed after 426 rows of freshly-positioned history -- breaking
        # a chain that was otherwise perfectly intact. Any operator who
        # upgrades and runs the app before backfilling would hit exactly
        # that.
        row = (
            await session.execute(
                select(model.chain_pos, model.row_hash, model.id)
                .where(model.row_hash.is_not(None))
                .order_by(
                    (model.chain_pos.is_not(None)).desc(),
                    func.coalesce(model.chain_pos, model.id).desc(),
                )
                .limit(1)
            )
        ).first()
        if row is None:
            continue
        key = (row[0] is not None, row[0] if row[0] is not None else row[2])
        if best_key is None or key > best_key:
            best_key, best_hash = key, row[1]
    return best_hash or GENESIS_HASH


async def append_chained(session: AsyncSession, row: TaskTransition) -> TaskTransition:
    """Stamp `row` with its chain hashes and add it to the session.

    Does NOT commit -- the caller's own transaction boundary owns that,
    which is what keeps the audit row atomic with the state change it
    records. A rolled-back transition writes no audit row and releases
    the lock, leaving the chain untouched.
    """
    await _lock_chain(session)
    prev = await _current_tip_hash(session)
    # Allocated INSIDE the lock, so allocation order == chain order ==
    # commit order. Gaps (a rolled-back transaction consuming a value)
    # are fine: verification walks ordered and never assumes contiguity.
    row.chain_pos = await session.scalar(
        text("SELECT nextval('audit_chain_pos_seq')")
    )
    row.prev_hash = prev
    row.row_hash = hash_for_row(row, prev)
    session.add(row)
    return row


@dataclass
class VerificationResult:
    """Deliberately reports the FIRST divergence, not a count.

    A chain is only meaningful up to its first break: once one row's
    hash is wrong, every subsequent row's expected `prev_hash` is
    computed from a value that no longer matches, so "how many rows are
    bad" is not an answerable question -- it would just be "all of them
    after the first," which tells an operator nothing useful.
    """

    intact: bool
    checked: int
    tip_hash: str | None = None
    first_divergence_id: int | None = None
    # Milestone 23: an id alone is ambiguous now that the chain spans two
    # tables -- "row 12" could be either. Reporting the table makes a
    # divergence actionable rather than a lookup puzzle.
    first_divergence_table: str | None = None
    reason: str | None = None
    unchained: int = 0


async def _all_rows_in_chain_order(session: AsyncSession) -> list:
    """Every chained table's rows, merged into one sequence ordered by
    the shared `chain_pos`.

    This merged view is what makes the shared chain real rather than
    nominal: an `agent_actions` row written between two
    `task_transitions` rows sits between them here, so deleting either
    table's rows wholesale breaks the surviving links.

    Never assumes contiguity: the shared sequence has real gaps
    (rolled-back transactions consume a value without leaving a row),
    so "chain_pos N+1 follows N" is simply false.
    """
    rows = []
    for table in CHAINED_TABLES:
        rows.extend(
            (
                await session.execute(select(table.model).order_by(table.model.id))
            ).scalars().all()
        )

    # Sort key, and why every part of it is load-bearing:
    #
    #   * chain_pos IS NULL sorts FIRST (0), because the only rows in
    #     that state are pre-chain_pos legacy rows -- Milestone 22-era
    #     `task_transitions` written before the shared sequence existed,
    #     and genuinely older than anything positioned.
    #   * `r.id` is the tiebreaker, NOT an afterthought. Found live
    #     against the real dev database: 426 Milestone 22 rows all had
    #     chain_pos NULL, so without a tiebreaker they compared equal and
    #     Python's stable sort simply preserved whatever arbitrary order
    #     Postgres returned them in -- which made a perfectly intact
    #     chain report as BROKEN. `ORDER BY id` above plus this
    #     tiebreaker makes the order deterministic in both places.
    return sorted(
        rows,
        key=lambda r: (0, r.id) if r.chain_pos is None else (1, r.chain_pos),
    )


async def verify_chain(session: AsyncSession) -> VerificationResult:
    """Walk the chain from genesis, recompute every hash, report the
    first divergence.

    Since Milestone 23 this spans BOTH `task_transitions` and
    `agent_actions` as one chain, merged by `chain_pos`.
    """
    rows = await _all_rows_in_chain_order(session)

    unchained = sum(1 for r in rows if r.row_hash is None)
    chained = [r for r in rows if r.row_hash is not None]

    prev = GENESIS_HASH
    checked = 0
    for row in chained:
        if row.prev_hash != prev:
            return VerificationResult(
                intact=False,
                checked=checked,
                first_divergence_id=row.id,
                first_divergence_table=_table_for(row).name,
                unchained=unchained,
                reason=(
                    f"{_table_for(row).name} row {row.id} claims "
                    f"prev_hash={row.prev_hash!r} but the preceding row's hash "
                    f"is {prev!r} -- the chain was broken, reordered, or rows "
                    "were deleted at this point"
                ),
            )
        expected = hash_for_row(row, prev)
        if expected != row.row_hash:
            return VerificationResult(
                intact=False,
                checked=checked,
                first_divergence_id=row.id,
                first_divergence_table=_table_for(row).name,
                unchained=unchained,
                reason=(
                    f"{_table_for(row).name} row {row.id}'s content does not "
                    f"match its stored hash (stored {row.row_hash[:16]}..., "
                    f"recomputed {expected[:16]}...) -- this row was altered "
                    "after it was written"
                ),
            )
        prev = row.row_hash
        checked += 1

    return VerificationResult(
        intact=True,
        checked=checked,
        tip_hash=prev if checked else None,
        unchained=unchained,
    )


async def chain_tip(session: AsyncSession) -> str:
    """The chain's current tip hash.

    The one in-scope mitigation for the forward-recomputation attack
    this module cannot otherwise detect: an operator who records this
    value somewhere outside the database can compare it later. A
    forward-recomputed chain necessarily yields a different tip, so the
    comparison catches exactly the attack `verify_chain()` cannot. Only
    works if the operator actually records it -- see the module
    docstring.
    """
    return await _current_tip_hash(session)


async def backfill_chain(session: AsyncSession) -> int:
    """Chain any rows written before this milestone existed.

    Returns (newly_chained, legacy_positioned).

    Honest about what this is worth, since it would be easy to mistake a
    passing verification afterward for evidence about the past: a
    backfilled row is attested **as of backfill time, not as of write
    time**. If a historical row was already altered before today, this
    happily certifies the altered version. No in-database mechanism can
    do better retroactively -- the guarantee necessarily starts when
    chaining starts.
    """
    await _lock_chain(session)
    rows = await _all_rows_in_chain_order(session)

    # Count legacy rows FIRST, then number them upward from -count, so
    # the earliest is most negative and their original order is
    # preserved. Counting up front is deliberately boring: an earlier
    # version numbered downward and flipped signs afterwards, and the
    # flip arithmetic was simply wrong (it produced positive, reversed
    # values). A pre-count needs no arithmetic to get wrong.
    legacy_rows = [r for r in rows if r.row_hash is not None and r.chain_pos is None]
    legacy_pos = -len(legacy_rows) - 1

    prev = GENESIS_HASH
    filled = 0
    positioned = 0
    for row in rows:
        if row.row_hash is None:
            if row.chain_pos is None:
                row.chain_pos = await session.scalar(
                    text("SELECT nextval('audit_chain_pos_seq')")
                )
            row.prev_hash = prev
            row.row_hash = hash_for_row(row, prev)
            session.add(row)
            filled += 1
        elif row.chain_pos is None:
            # Milestone 22-era row: already chained, but written before
            # the shared sequence existed. Give it a position WITHOUT
            # touching its hashes -- recomputing them would silently
            # re-attest content this backfill has no business vouching
            # for, destroying the very evidence the chain exists to
            # preserve. Position only; the existing chain is untouched.
            #
            # NEGATIVE positions, counting up toward 0, NOT nextval().
            # Found live: the sequence may already have handed positive
            # positions to newer rows, so assigning legacy rows from it
            # places recorded HISTORY after data written later, and the
            # chain reports as broken. `nextval` only ever returns
            # positive values, so negatives are permanently reserved for
            # "predates the shared sequence" and can never collide.
            legacy_pos += 1
            row.chain_pos = legacy_pos
            positioned += 1
            session.add(row)
        prev = row.row_hash

    await session.commit()
    return filled, positioned
