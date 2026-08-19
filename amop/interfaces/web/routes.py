"""Server-rendered web dashboard (Section 16.2), Stage 3 -- CLAUDE.md's
scoped-down page list: a task list grouped by state, and a task detail
page with a plain diff viewer. Mounted as a second router on the same
FastAPI app as Stage 1 (`amop/api/app.py`) -- same DB session
dependency (`get_session`), same underlying rows every API caller
already sees; HTML instead of JSON is a rendering choice, not a second
path around Section 3.1's Command Layer.

Explicitly not built (CLAUDE.md's own Phase-3 list): a live WebSocket
feed, cost/trend charts.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from amop.api.deps import get_session
from amop.database.models import Task
from amop.interfaces.web.auth import (
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    make_session_cookie_value,
    require_web_session,
    token_is_valid,
)
from amop.orchestrator.state_machine import TaskState

router = APIRouter(prefix="/web", tags=["web"])

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Section 4.1's own state list, in the state machine's declared
# lifecycle order (not alphabetical) -- the dashboard reads top-to-
# bottom as a pipeline (just-created first, done last), a plainer
# grouping than Telegram's phone-first "urgent things first" one, which
# a browser's extra screen room doesn't need.
_GROUP_ORDER = list(TaskState)
_TASKS_PER_GROUP_CAP = 25


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, error: str | None = None) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html", {"error": error})


@router.post("/login")
async def login_submit(token: str = Form(...)) -> RedirectResponse:
    if not token_is_valid(token):
        return RedirectResponse(url="/web/login?error=1", status_code=303)
    response = RedirectResponse(url="/web/tasks", status_code=303)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        make_session_cookie_value(),
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
    )
    return response


@router.get("/logout")
async def logout() -> RedirectResponse:
    response = RedirectResponse(url="/web/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


@router.get(
    "/tasks", response_class=HTMLResponse, dependencies=[Depends(require_web_session)]
)
async def task_list(
    request: Request, session: AsyncSession = Depends(get_session)
) -> HTMLResponse:
    stmt = select(Task).order_by(Task.created_at.desc()).limit(500)
    tasks = list((await session.execute(stmt)).scalars().all())

    groups = []
    for state in _GROUP_ORDER:
        state_tasks = [t for t in tasks if t.state == state.value]
        if not state_tasks:
            continue
        groups.append(
            {
                "state": state.value,
                "count": len(state_tasks),
                "tasks": state_tasks[:_TASKS_PER_GROUP_CAP],
                "overflow": max(0, len(state_tasks) - _TASKS_PER_GROUP_CAP),
            }
        )

    return templates.TemplateResponse(
        request, "tasks.html", {"groups": groups, "show_logout": True}
    )


@router.get(
    "/tasks/{task_id}",
    response_class=HTMLResponse,
    dependencies=[Depends(require_web_session)],
)
async def task_detail(
    task_id: uuid.UUID, request: Request, session: AsyncSession = Depends(get_session)
) -> HTMLResponse:
    task = await session.get(Task, task_id)
    if task is None:
        return templates.TemplateResponse(
            request,
            "task_not_found.html",
            {"task_id": task_id, "show_logout": True},
            status_code=404,
        )
    ctx = task.task_context or {}
    # Same field this task's own /tasks/{id}/diff API endpoint reads --
    # one source of "what counts as this task's diff", not a second one
    # invented for the dashboard.
    diff = ctx.get("diff") or ""
    return templates.TemplateResponse(
        request,
        "task_detail.html",
        {
            "task": task,
            "prompt": ctx.get("prompt"),
            "repo": ctx.get("repo"),
            "diff": diff,
            "show_logout": True,
        },
    )
