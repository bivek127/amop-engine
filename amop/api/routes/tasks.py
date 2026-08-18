"""Section 15.2's task endpoints -- the largest single group, since
every other interface's core loop is "look at tasks, act on tasks."
"""

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from amop.api.auth import require_operator_token
from amop.api.deps import get_session
from amop.api.errors import api_error
from amop.api.schemas import DiffOut, TaskCreate, TaskOut, TaskTransitionOut
from amop.database.models import Repository, Task
from amop.orchestrator.state_machine import IllegalTransitionError, TaskState
from amop.orchestrator.task import create_task, get_task, get_transitions, transition

router = APIRouter(prefix="/tasks", tags=["tasks"])


async def _get_or_404(session: AsyncSession, task_id: uuid.UUID) -> Task:
    task = await get_task(session, task_id)
    if task is None:
        raise api_error(404, "TASK_NOT_FOUND", f"no task with id {task_id}")
    return task


async def _transition_or_409(
    session: AsyncSession, task: Task, to_state: TaskState, *, actor: str, trigger: str
) -> Task:
    """Section 4.2's own validation is the check here -- approve/reject/
    cancel don't re-implement "is this task in the right state", they
    just attempt the real transition() and translate its own
    IllegalTransitionError into the standard envelope. One source of
    truth for what's legal, same one the CLI and the chain both already
    use."""
    try:
        return await transition(session, task, to_state, actor=actor, trigger=trigger)
    except IllegalTransitionError as exc:
        raise api_error(409, "ILLEGAL_TRANSITION", str(exc))


@router.post(
    "",
    response_model=TaskOut,
    status_code=201,
    dependencies=[Depends(require_operator_token)],
)
async def create_task_endpoint(
    body: TaskCreate, session: AsyncSession = Depends(get_session)
) -> Task:
    """Creates the row only -- does not execute the chain. See the
    Milestone 15 plan's own decision on this: there is no async
    worker/queue in this codebase (spec's architecture diagram assumes a
    Redis-backed one that was never built), and this endpoint's own spec
    description is "create a task manually", not "create and run".
    Running one still goes through `amop fix`/`update-deps`/`optimize`.
    """
    repo_path = body.repo
    if body.repo_id is not None:
        repo_row = await session.get(Repository, body.repo_id)
        if repo_row is None:
            raise api_error(
                404, "REPOSITORY_NOT_FOUND", f"no repository with id {body.repo_id}"
            )
        repo_path = repo_row.repo_path
    task_context = {"repo": repo_path, "prompt": body.description}
    return await create_task(session, task_type=body.task_type, task_context=task_context)


@router.get("/{task_id}", response_model=TaskOut)
async def get_task_endpoint(
    task_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> Task:
    return await _get_or_404(session, task_id)


@router.get("", response_model=list[TaskOut])
async def list_tasks_endpoint(
    state: str | None = None,
    repo_id: uuid.UUID | None = None,
    limit: int = Query(default=50, le=200),
    session: AsyncSession = Depends(get_session),
) -> list[Task]:
    stmt = select(Task).order_by(Task.created_at.desc()).limit(limit)
    if state is not None:
        stmt = stmt.where(Task.state == state)
    if repo_id is not None:
        repo_row = await session.get(Repository, repo_id)
        if repo_row is None:
            return []
        # Task never gained a repo_id column (Milestone 1 dropped it,
        # same reasoning as every other table) -- filtered on the same
        # free-form repo path every other endpoint already keys on.
        stmt = stmt.where(Task.task_context["repo"].astext == repo_row.repo_path)
    return list((await session.execute(stmt)).scalars().all())


@router.post(
    "/{task_id}/cancel",
    response_model=TaskOut,
    dependencies=[Depends(require_operator_token)],
)
async def cancel_task_endpoint(
    task_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> Task:
    task = await _get_or_404(session, task_id)
    return await _transition_or_409(
        session, task, TaskState.CANCELLED, actor="human:api", trigger="cancelled via API"
    )


@router.post(
    "/{task_id}/approve",
    response_model=TaskOut,
    dependencies=[Depends(require_operator_token)],
)
async def approve_task_endpoint(
    task_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> Task:
    """WAITING_FOR_APPROVAL -> MERGED. This transition has existed in
    TRANSITIONS since Milestone 1 but has never had a real caller until
    this endpoint -- every task that reached WAITING_FOR_APPROVAL before
    now just sat there, approved by a human clicking "merge" on GitHub
    directly, outside AMOP's own state machine entirely."""
    task = await _get_or_404(session, task_id)
    return await _transition_or_409(
        session, task, TaskState.MERGED, actor="human:api", trigger="approved via API"
    )


@router.post(
    "/{task_id}/reject",
    response_model=TaskOut,
    dependencies=[Depends(require_operator_token)],
)
async def reject_task_endpoint(
    task_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> Task:
    task = await _get_or_404(session, task_id)
    return await _transition_or_409(
        session, task, TaskState.CANCELLED, actor="human:api", trigger="rejected via API"
    )


@router.get("/{task_id}/actions", response_model=list[TaskTransitionOut])
async def task_actions_endpoint(task_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    """Section 15.2 names this `agent_actions` -- no such table exists
    (Milestone 14 already made this exact substitution for Reporter).
    `task_transitions` is what this codebase actually has: a real,
    append-only audit trail of every state change, which is what "the
    full log for this task" means when there's no separate per-tool-call
    ledger."""
    await _get_or_404(session, task_id)
    return await get_transitions(session, task_id)


@router.get("/{task_id}/diff", response_model=DiffOut)
async def task_diff_endpoint(
    task_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> DiffOut:
    task = await _get_or_404(session, task_id)
    diff = (task.task_context or {}).get("diff") or ""
    return DiffOut(task_id=task.id, diff=diff, available=bool(diff))
