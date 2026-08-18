import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from amop.api.auth import require_operator_token
from amop.api.deps import get_session
from amop.api.errors import api_error
from amop.api.schemas import MemoryDisputeUpdate, MemoryOut
from amop.database.models import MemoryItem, Repository
from amop.memory.store import mark_disputed

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("", response_model=list[MemoryOut])
async def list_memory(
    repo_id: uuid.UUID | None = None,
    type: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(MemoryItem).order_by(MemoryItem.created_at.desc())
    if repo_id is not None:
        repo_row = await session.get(Repository, repo_id)
        if repo_row is None:
            return []
        stmt = stmt.where(MemoryItem.repo_path == repo_row.repo_path)
    if type is not None:
        stmt = stmt.where(MemoryItem.memory_type == type)
    return list((await session.execute(stmt)).scalars().all())


@router.patch(
    "/{memory_id}",
    response_model=MemoryOut,
    dependencies=[Depends(require_operator_token)],
)
async def dispute_memory(
    memory_id: uuid.UUID,
    body: MemoryDisputeUpdate,
    session: AsyncSession = Depends(get_session),
) -> MemoryItem:
    """Section 10.4's dispute toggle, over HTTP -- the logic is
    Milestone 14's `memory.store.mark_disputed()` unchanged, this is
    just the wrapper CLAUDE.md's own item 1 describes it as needing."""
    found = await mark_disputed(session, memory_id, disputed=body.disputed)
    if not found:
        raise api_error(404, "MEMORY_NOT_FOUND", f"no memory item with id {memory_id}")
    row = await session.get(MemoryItem, memory_id)
    return row
