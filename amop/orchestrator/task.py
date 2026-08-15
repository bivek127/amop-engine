import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from amop.database.models import Task, TaskTransition
from amop.orchestrator.state_machine import TERMINAL_STATES, TaskState, validate_transition


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
    task.version += 1
    if to_state in TERMINAL_STATES:
        task.resolved_at = now

    session.add(
        TaskTransition(
            task_id=task.id,
            from_state=from_state.value,
            to_state=to_state.value,
            trigger=trigger or spec_trigger,
            actor=actor,
            timestamp=now,
        )
    )
    await session.commit()
    await session.refresh(task)
    return task


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
