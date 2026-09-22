"""Section 15.2's task endpoints -- the largest single group, since
every other interface's core loop is "look at tasks, act on tasks."
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from amop.api.auth import require_operator_token
from amop.api.deps import get_session, get_session_factory
from amop.api.errors import api_error
from amop.api.schemas import DiffOut, TaskCreate, TaskOut, TaskTransitionOut
from amop.database.models import Repository, Task
from amop.models.ollama import DEFAULT_MODEL, OllamaProvider
from amop.orchestrator.chain import run_fix_and_persist
from amop.orchestrator.state_machine import IllegalTransitionError, TaskState
from amop.orchestrator.task import (
    ConcurrentUpdateError,
    create_task,
    get_task,
    get_transitions,
    transition,
)

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
    except ConcurrentUpdateError as exc:
        # Milestone 16 / Section 4.3.1. Deliberately NOT retried here:
        # a 409 tells the caller their view of the task was stale, and
        # for a human clicking Approve the honest answer is "someone
        # else just changed this, look again" -- not a silent retry that
        # applies their click to a task they never actually saw.
        raise api_error(409, "CONCURRENT_UPDATE", str(exc))


@router.post(
    "",
    response_model=TaskOut,
    status_code=201,
    dependencies=[Depends(require_operator_token)],
)
async def create_task_endpoint(
    body: TaskCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> Task:
    """Creates the row -- and, for a `bug_fix` with a real repo and
    description, also kicks off `run_fix()` in the background (Milestone
    18 Resumed / Step 0). Milestone 15's original "create the row only"
    contract still holds for every other case: no repo/description means
    there is nothing to run yet (Section 15.2's "create a task manually"
    shape, e.g. for a task a human will populate/dispatch some other
    way), and `optimization`/`dependency_update` tasks are unaffected --
    those still only ever run via `amop optimize`/`amop update-deps`, per
    this milestone's own explicit scope (the dashboard's submit form is a
    bug-report form, not a generic task-type dispatcher).
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
    task = await create_task(session, task_type=body.task_type, task_context=task_context)

    if body.task_type == "bug_fix" and repo_path and body.description:
        background_tasks.add_task(
            run_fix_and_persist,
            get_session_factory(),
            task.id,
            description=body.description,
            repo_path=Path(repo_path),
            model=OllamaProvider(model=DEFAULT_MODEL),
        )
    return task


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
