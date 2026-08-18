from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from amop.api.deps import get_session
from amop.api.schemas import PullRequestOut
from amop.database.models import PullRequest

router = APIRouter(prefix="/pull-requests", tags=["pull-requests"])


@router.get("", response_model=list[PullRequestOut])
async def list_pull_requests(
    status: str | None = None, session: AsyncSession = Depends(get_session)
):
    stmt = select(PullRequest).order_by(PullRequest.created_at.desc())
    if status is not None:
        stmt = stmt.where(PullRequest.status == status)
    return list((await session.execute(stmt)).scalars().all())
