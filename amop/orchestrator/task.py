import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm.exc import StaleDataError
from sqlalchemy.ext.asyncio import AsyncSession

from amop.audit.chain import append_chained
from amop.database.models import Task, TaskTransition
from amop.orchestrator.state_machine import (
    TERMINAL_STATES,
    IllegalTransitionError,
    TaskState,
    validate_transition,
)

# Section 4.3.1: how many times a losing writer re-reads and re-evaluates
# before giving up. Small on purpose -- a conflict means a real second
# writer acted, and re-evaluation either finds the move still legal (one
# retry is enough) or finds the task somewhere it must not be dragged out
# of, which no amount of retrying fixes.
MAX_OCC_RETRIES = 3


class ConcurrentUpdateError(Exception):
    """A write lost the optimistic-concurrency race (Section 4.3.1).

    Raised instead of leaking SQLAlchemy's StaleDataError so callers can
    handle "someone else changed this row" without importing ORM
    internals, and so the API layer can map it onto a real 409.
    """

    def __init__(self, task_id, attempted: TaskState) -> None:
        self.task_id = task_id
        self.attempted = attempted
        super().__init__(
            f"concurrent update on task {task_id}: another writer modified it "
            f"while this caller was attempting {attempted.value}"
        )


async def create_task(
    session: AsyncSession,
    task_type: str,
    task_context: dict | None = None,
    severity: str | None = None,
) -> Task:
    """Create a new Task row in state CREATED. No task_transitions row is
    written here — Section 4.2's table only ever lists CREATED as a
    from_state, never a to_state, so entering it isn't itself a logged
    transition."""
    task = Task(
        id=uuid.uuid4(),
        task_type=task_type,
        state=TaskState.CREATED.value,
        severity=severity,
        task_context=task_context,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task


async def transition(
    session: AsyncSession,
    task: Task,
    to_state: TaskState,
    trigger: str | None = None,
    actor: str = "system",
) -> Task:
    """Drive `task` from its current state to `to_state`. Validates the
    move against the state machine (Section 4.2/4.6) and writes the audit
    row to task_transitions before returning. Raises
    IllegalTransitionError for an invalid move — nothing is written in
    that case.
    """
    from_state = TaskState(task.state)
    spec_trigger = validate_transition(from_state, to_state)

    now = datetime.now(UTC)
    task.state = to_state.value
    task.updated_at = now
    # version is NOT touched here: the mapper's version_id_col owns the
    # increment (models.py). Incrementing by hand as well would fight it.
    if to_state in TERMINAL_STATES:
        task.resolved_at = now

    # Captured BEFORE the rollback below. session.rollback() expires every
    # attribute on the instance, so reading task.id afterwards triggers a
    # lazy reload -- synchronous IO on an async session, which raises
    # MissingGreenlet and buries the real conflict under a confusing error.
    #
    # The tests did not catch this: they asserted that the losing writer
    # raised *something*, and MissingGreenlet satisfied that perfectly.
    # Found by running the live demo instead. The assertion now pins the
    # exception TYPE, which is what would have caught it.
    task_id = task.id

    # The OCC guard wraps BOTH the audit append and the commit, not the
    # commit alone. Milestone 22 made that distinction load-bearing:
    # append_chained() issues its own session.execute() calls (the
    # advisory lock, the tip read), and SQLAlchemy AUTOFLUSHES pending
    # changes before any query -- so the version-guarded UPDATE on
    # `tasks` now fires inside append_chained(), and a lost OCC race
    # raises StaleDataError there rather than at commit().
    #
    # With the try around commit() alone, that escaped as a raw
    # StaleDataError and callers (api/routes/tasks.py's 409 mapping,
    # transition_with_retry) silently stopped recognizing it as a
    # conflict. Caught by Milestone 16's own exception-TYPE assertion --
    # the one tightened in that milestone after a count-only version let
    # a real bug through. Wrapping the whole region is also simply more
    # honest: a flush can legitimately happen anywhere in here, and every
    # such failure means the same thing.
    try:
        # Milestone 22 (Section 12.6): stamped with its chain hashes and
        # added to THIS transaction, so the audit row is atomic with the
        # state change it records -- a rolled-back transition writes no
        # audit row and leaves the chain untouched. The global advisory
        # lock append_chained() takes releases at the commit below; see
        # audit/chain.py for why read-tip -> insert must be exclusive.
        await append_chained(
            session,
            TaskTransition(
                task_id=task.id,
                from_state=from_state.value,
                to_state=to_state.value,
                trigger=trigger or spec_trigger,
                actor=actor,
                timestamp=now,
            ),
        )
        await session.commit()
    except StaleDataError as exc:
        # Zero rows matched `WHERE id = ? AND version = ?` -- a competing
        # write landed between this caller's read and its commit. Nothing
        # from this transaction persists, including the task_transitions
        # audit row, which is what keeps the trail honest.
        await session.rollback()
        raise ConcurrentUpdateError(task_id, to_state) from exc
    await session.refresh(task)
    return task


async def transition_with_retry(
    session: AsyncSession,
    task: Task,
    to_state: TaskState,
    trigger: str | None = None,
    actor: str = "system",
    max_retries: int = MAX_OCC_RETRIES,
) -> Task:
    """transition(), but on conflict re-read the row and re-evaluate.

    Section 4.3.1's rule is "re-read and re-evaluate", NOT "retry until
    it sticks", and the difference is the whole point. After a conflict
    the task may be somewhere that makes this move illegal -- a human
    rejecting to CANCELLED while a chain was mid-run is the concrete
    case -- and blindly re-applying would resurrect a task the human
    explicitly killed. Re-validation happens against the FRESH state, so
    an illegal move surfaces as IllegalTransitionError rather than
    silently landing.
    """
    task_id = task.id  # same expiry hazard as above
    for attempt in range(max_retries):
        try:
            return await transition(
                session, task, to_state, trigger=trigger, actor=actor
            )
        except ConcurrentUpdateError:
            if attempt == max_retries - 1:
                raise
            # populate_existing: overwrite this session's cached copy
            # rather than handing back the same stale object. Without it
            # expire_on_commit=False means the retry re-reads its own
            # stale snapshot and conflicts forever.
            refreshed = await session.get(Task, task_id, populate_existing=True)
            if refreshed is None:
                raise
            task = refreshed
            # Re-validate against where the task ACTUALLY is now. Raises
            # IllegalTransitionError if the competing writer moved it
            # somewhere this transition may not follow from.
            validate_transition(TaskState(task.state), to_state)
    raise ConcurrentUpdateError(task_id, to_state)


async def get_task(session: AsyncSession, task_id: uuid.UUID) -> Task | None:
    return await session.get(Task, task_id)


async def get_transitions(
    session: AsyncSession, task_id: uuid.UUID
) -> list[TaskTransition]:
    result = await session.execute(
        select(TaskTransition)
        .where(TaskTransition.task_id == task_id)
        .order_by(TaskTransition.id)
    )
    return list(result.scalars().all())
