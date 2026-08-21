import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from amop.api.auth import require_operator_token
from amop.api.deps import get_session
from amop.api.errors import api_error
from amop.api.schemas import (
    RepositoryCreate,
    RepositoryOut,
    RepositoryPermissionsUpdate,
)
from amop.database.models import Repository
from amop.safety.permissions import normalize_repo_identity

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
    # Milestone 20: stored in canonical form, because permission lookup
    # and Milestone 16's lock keys both resolve identity this way. A row
    # stored raw would silently never match either (on macOS "/tmp/x"
    # and "/private/tmp/x" are the same directory but not the same
    # string) -- and a permission override that never matches fails
    # quietly toward the global mode, which is exactly the kind of silent
    # safety downgrade this project's rules exist to prevent.
    row = Repository(
        repo_path=normalize_repo_identity(body.repo_path),
        display_name=body.display_name,
        url=body.url,
        default_branch=body.default_branch or "main",
        permission_overrides=body.permission_overrides,
    )
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


@router.patch(
    "/{repo_id}/permissions",
    response_model=RepositoryOut,
    dependencies=[Depends(require_operator_token)],
)
async def update_permissions(
    repo_id: uuid.UUID,
    body: RepositoryPermissionsUpdate,
    session: AsyncSession = Depends(get_session),
) -> Repository:
    """Spec 15.2's `PATCH /repositories/{id}/permissions` (Section 12.1).

    Operator-gated like every other mutating endpoint. Note what this
    can and cannot do: it sets a repo's mode policy, but it can never
    weaken the hardcoded protections -- Milestone 19's protected-path
    check runs unconditionally and is not mode-derived, so no override
    written here can unlock `.github/workflows/`.
    """
    row = await session.get(Repository, repo_id)
    if row is None:
        raise api_error(404, "REPOSITORY_NOT_FOUND", f"no repository with id {repo_id}")
    row.permission_overrides = body.permission_overrides
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row
