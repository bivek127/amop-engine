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

SCOPE -- what is and is not chained today:

  * CHAINED: `task_transitions` -- state-machine decisions (who moved a
    task between states, and why).
  * NOT CHAINED: the Safety Engine's ALLOW/DENY tool-call decisions.
    Those live in `tasks.task_context["tool_calls"]`, a JSONB list that
    is wholly overwritten on every write, so it is neither append-only
    nor chainable as-is. Section 12.6 calls that record "the primary
    artifact... for post-incident review of what an agent actually
    attempted" -- i.e. the MORE security-relevant of the two. It is
    explicitly unresolved, not overlooked. See ROADMAP.md's Milestone 22
    "UNRESOLVED GAP" section.

LOCKED DECISION for whoever builds a real `agent_actions` table: it
joins THIS chain -- one shared chain across both tables, never a second
independent one. With separate chains, deleting every row of one table
leaves an empty chain that verifies as INTACT, which is worse than no
verification because it actively misleads. A shared chain also attests
the interleaving (this tool call happened between those two state
changes), which separate chains cannot express.

`AUDIT_CHAIN_LOCK_KEY` below is already a single GLOBAL key rather than
a per-table one specifically to keep that path open. The part still to
design is a shared total ordering across both tables -- per-table
BIGSERIAL does not provide one -- and it must be settled BEFORE rows
exist, because retrofitting an ordering onto a live chain is far harder.
Full reasoning in ROADMAP.md's "LOCKED DECISION" section.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from amop.database.models import TaskTransition

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


def canonical_content(
    task_id: uuid.UUID | str,
    from_state: str | None,
    to_state: str,
    trigger: str | None,
    actor: str | None,
    timestamp: datetime,
) -> str:
    """A row's content in one deterministic, unambiguous string.

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
    return json.dumps(
        {
            "task_id": str(task_id),
            "from_state": from_state,
            "to_state": to_state,
            "trigger": trigger,
            "actor": actor,
            "timestamp": timestamp.isoformat(),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
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


def hash_for_row(row: TaskTransition, prev_hash: str) -> str:
    """Recompute a persisted row's hash -- the verification side of the
    same function `append_chained()` uses on the write side. Both go
    through `canonical_content()` so the two can never drift apart."""
    return compute_row_hash(
        prev_hash,
        canonical_content(
            task_id=row.task_id,
            from_state=row.from_state,
            to_state=row.to_state,
            trigger=row.trigger,
            actor=row.actor,
            timestamp=row.timestamp,
        ),
    )


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

    Ordered by `id DESC`, not by `timestamp`: timestamps are
    Python-assigned and two rows can share one, while `id` comes from a
    sequence allocated inside the append lock, making it the only
    ordering that reflects true append order.
    """
    tip = (
        await session.execute(
            select(TaskTransition.row_hash)
            .where(TaskTransition.row_hash.is_not(None))
            .order_by(TaskTransition.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return tip or GENESIS_HASH


async def append_chained(session: AsyncSession, row: TaskTransition) -> TaskTransition:
    """Stamp `row` with its chain hashes and add it to the session.

    Does NOT commit -- the caller's own transaction boundary owns that,
    which is what keeps the audit row atomic with the state change it
    records. A rolled-back transition writes no audit row and releases
    the lock, leaving the chain untouched.
    """
    await _lock_chain(session)
    prev = await _current_tip_hash(session)
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
    reason: str | None = None
    unchained: int = 0


async def verify_chain(session: AsyncSession) -> VerificationResult:
    """Walk the chain from genesis, recompute every hash, report the
    first divergence.

    Walks `ORDER BY id` and never assumes contiguity: the sequence
    already has real gaps (rolled-back transactions consume a value
    without leaving a row -- dev had ids 1..433 for 426 rows before this
    milestone), so "id N+1 follows id N" is simply false here.
    """
    rows = list(
        (
            await session.execute(select(TaskTransition).order_by(TaskTransition.id))
        ).scalars().all()
    )

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
                unchained=unchained,
                reason=(
                    f"row {row.id} claims prev_hash={row.prev_hash!r} but the "
                    f"preceding row's hash is {prev!r} -- the chain was broken "
                    "or reordered at this point"
                ),
            )
        expected = hash_for_row(row, prev)
        if expected != row.row_hash:
            return VerificationResult(
                intact=False,
                checked=checked,
                first_divergence_id=row.id,
                unchained=unchained,
                reason=(
                    f"row {row.id}'s content does not match its stored hash "
                    f"(stored {row.row_hash[:16]}..., recomputed "
                    f"{expected[:16]}...) -- this row was altered after it "
                    "was written"
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

    Returns how many rows were newly chained.

    Honest about what this is worth, since it would be easy to mistake a
    passing verification afterward for evidence about the past: a
    backfilled row is attested **as of backfill time, not as of write
    time**. If a historical row was already altered before today, this
    happily certifies the altered version. No in-database mechanism can
    do better retroactively -- the guarantee necessarily starts when
    chaining starts.
    """
    await _lock_chain(session)
    rows = list(
        (
            await session.execute(select(TaskTransition).order_by(TaskTransition.id))
        ).scalars().all()
    )

    prev = GENESIS_HASH
    filled = 0
    for row in rows:
        if row.row_hash is None:
            row.prev_hash = prev
            row.row_hash = hash_for_row(row, prev)
            session.add(row)
            filled += 1
        prev = row.row_hash
    await session.commit()
    return filled
