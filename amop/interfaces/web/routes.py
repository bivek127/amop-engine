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
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from amop.api.deps import get_session
from amop.api.routes.tasks import create_task_endpoint
from amop.api.schemas import TaskCreate
from amop.database.models import PullRequest, Repository, Task
from amop.interfaces.web.auth import (
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    make_session_cookie_value,
    require_web_session,
    token_is_valid,
)
from amop.orchestrator.state_machine import TERMINAL_STATES, TaskState
from amop.orchestrator.task import get_transitions

router = APIRouter(prefix="/web", tags=["web"])

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Section 4.1's own state list, in the state machine's declared
# lifecycle order (not alphabetical) -- the dashboard reads top-to-
# bottom as a pipeline (just-created first, done last), a plainer
# grouping than Telegram's phone-first "urgent things first" one, which
# a browser's extra screen room doesn't need.
_GROUP_ORDER = list(TaskState)
_TASKS_PER_GROUP_CAP = 25

# Milestone 18: a badge color per state, matching base.html's token
# names exactly (amber/red/green/blue/gray) -- one source of "what color
# is this state" shared by every template rather than each re-deciding.
# Same semantic-color-separate-from-accent split Telegram's /status
# reformat (Milestone 15) used, now applied visually instead of by emoji.
STATE_COLOR: dict[str, str] = {
    "WAITING_FOR_APPROVAL": "amber",
    "FAILED": "red",
    "CANCELLED": "gray",
    "MERGED_INTO_EXISTING": "gray",
    "NEEDS_HUMAN_INPUT": "amber",
    "MERGED": "green",
    "RESOLVED": "green",
}


def _state_color(state: str) -> str:
    return STATE_COLOR.get(state, "blue")  # everything else is "in progress"


def _relative_time(value: datetime | None) -> str:
    if value is None:
        return "—"
    now = datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    delta = now - value
    seconds = int(delta.total_seconds())
    if seconds < 5:
        return "just now"
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    return f"{days // 365}y ago"


templates.env.filters["relative_time"] = _relative_time
templates.env.filters["state_color"] = _state_color


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

    # Milestone 18 (Resumed) #3/#4: the real journey, from real ground
    # truth, not a fabricated pipeline -- task_transitions is Milestone
    # 22's hash-chained audit trail (the actual state-machine skeleton),
    # `stages` is chain.py's own narration of what happened inside each
    # state (root cause, test results, review verdict, PR link -- see
    # chain.py's `stage(...)` calls), and `pull_requests` is the real
    # table a PR is ever actually written to (never a task_context field
    # that could drift from what GitHub says).
    transitions = await get_transitions(session, task_id)
    pr_row = (
        await session.execute(
            select(PullRequest).where(PullRequest.task_id == task_id)
        )
    ).scalar_one_or_none()

    return templates.TemplateResponse(
        request,
        "task_detail.html",
        {
            "task": task,
            "prompt": ctx.get("prompt"),
            "repo": ctx.get("repo"),
            "diff": diff,
            "diff_lines": _diff_lines(diff),
            "transitions": transitions,
            "stages": ctx.get("stages") or [],
            "root_cause_report": ctx.get("root_cause_report"),
            "error": ctx.get("error") or ctx.get("background_execution_error"),
            "pull_request": pr_row,
            "is_live": TaskState(task.state) not in TERMINAL_STATES,
            "show_logout": True,
        },
    )


def _diff_lines(diff: str) -> list[dict]:
    """A unified diff, split for real +/- line coloring in the template --
    no client-side library, this project has no JS build step to hang one
    off (same "no external deps for something a 20-line loop covers"
    posture as tasks.html's own vanilla-JS search filter)."""
    lines = []
    for raw in diff.splitlines():
        if raw.startswith("+++") or raw.startswith("---"):
            kind = "meta"
        elif raw.startswith("@@"):
            kind = "hunk"
        elif raw.startswith("+"):
            kind = "add"
        elif raw.startswith("-"):
            kind = "del"
        else:
            kind = "ctx"
        lines.append({"kind": kind, "text": raw})
    return lines


@router.get(
    "/submit", response_class=HTMLResponse, dependencies=[Depends(require_web_session)]
)
async def submit_form(
    request: Request, session: AsyncSession = Depends(get_session)
) -> HTMLResponse:
    repos = list(
        (await session.execute(select(Repository).order_by(Repository.repo_path))).scalars()
    )
    return templates.TemplateResponse(
        request, "submit.html", {"repos": repos, "show_logout": True}
    )


@router.post("/submit", dependencies=[Depends(require_web_session)])
async def submit_bug(
    repo: str = Form(...),
    description: str = Form(...),
    session: AsyncSession = Depends(get_session),
) -> RedirectResponse:
    """Reuses `create_task_endpoint` directly rather than re-implementing
    its repo_id resolution / background-execution scheduling -- Section
    3.2's "no interface has a privileged bypass path" principle, applied
    the same way `_fix`'s own consistency test already proves it for the
    CLI (test_a_task_created_via_the_cli_path_is_visible_through_the_api).
    Calling the route function directly (not over HTTP) is safe here:
    `Depends(...)` defaults are only resolved by FastAPI when a request
    is routed through it, not when called as a plain function with real
    arguments -- and `background_tasks` is handed to the redirect
    response the same way FastAPI would have, so the background job runs
    exactly as it would via the JSON API.
    """
    background_tasks = BackgroundTasks()
    task = await create_task_endpoint(
        TaskCreate(task_type="bug_fix", repo=repo, description=description),
        background_tasks,
        session,
    )
    response = RedirectResponse(url=f"/web/tasks/{task.id}", status_code=303)
    response.background = background_tasks
    return response
