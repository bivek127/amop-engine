"""`POST /reports` -- not in Section 15.2's literal endpoint table.

Added here because Section 16.1's required `/report` Telegram command
has nothing else to call: Reporter (Milestone 14) computes a
`ReportSummary` fresh per invocation and has never persisted a "latest"
one anywhere, and generating one needs a real LLM call the bot process
can't make itself (it's a pure API client per Section 3.1, no Ollama of
its own). So the API generates the report on request, wrapping
Milestone 14's existing `orchestrator.reporting.run_report()` unchanged.

Auth-gated like a mutating endpoint even though it changes no persisted
state -- it has a real compute cost (one Ollama call), and CLAUDE.md's
"single shared-secret token... sufficient for v1" is as much about
guarding cost as it is about guarding writes.
"""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from amop.api.auth import require_operator_token
from amop.api.deps import get_session
from amop.models.ollama import DEFAULT_MODEL, OllamaProvider
from amop.orchestrator.reporting import run_report

router = APIRouter(prefix="/reports", tags=["reports"])

DEFAULT_WINDOW_DAYS = 7


class ReportRequest(BaseModel):
    since: datetime | None = None
    until: datetime | None = None


class ReportOut(BaseModel):
    period_start: str
    period_end: str
    tasks_resolved: int
    prs_opened: int
    prs_merged: int
    dependencies_updated: int
    top_issues: list[str]


@router.post("", response_model=ReportOut, dependencies=[Depends(require_operator_token)])
async def generate_report(
    body: ReportRequest, session: AsyncSession = Depends(get_session)
) -> ReportOut:
    end = body.until or datetime.now(UTC)
    start = body.since or (end - timedelta(days=DEFAULT_WINDOW_DAYS))
    summary, _window = await run_report(
        session, start=start, end=end, model=OllamaProvider(model=DEFAULT_MODEL)
    )
    return ReportOut(**summary.model_dump())
