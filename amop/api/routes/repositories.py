from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from amop.api.auth import require_operator_token
from amop.api.deps import get_session
from amop.api.errors import api_error
from amop.api.schemas import RepositoryCreate, RepositoryOut
from amop.database.models import Repository

router = APIRouter(prefix="/repositories", tags=["repositories"])


@router.get("", response_model=list[RepositoryOut])
async def list_repositories(session: AsyncSession = Depends(get_session)):
    stmt = select(Repository).order_by(Repository.created_at.desc())
    return list((await session.execute(stmt)).scalars().all())


@router.post(
    "",
    response_model=RepositoryOut,
    status_code=201,
    dependencies=[Depends(require_operator_token)],
)
async def register_repository(
    body: RepositoryCreate, session: AsyncSession = Depends(get_session)
) -> Repository:
    row = Repository(repo_path=body.repo_path, display_name=body.display_name)
    session.add(row)
    try:
        await session.commit()
    except IntegrityError:
        # repo_path is unique -- registering the same path twice is a
        # conflict, not a server error.
        await session.rollback()
        raise api_error(
            409, "REPOSITORY_EXISTS", f"{body.repo_path!r} is already registered"
        )
    await session.refresh(row)
    return row
