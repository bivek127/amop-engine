"""Shared FastAPI dependencies.

One engine/session-factory built once at app startup (app.py's
lifespan), not per request. A fresh `make_engine()` call inside every
route handler would open a new asyncpg connection pool on every single
API call -- wasteful, and at any real request rate would exhaust
Postgres's max_connections. Every other part of this codebase
(`cli/main.py`'s commands) already builds one engine per *process
run*, not per operation; this is the same discipline applied to a
long-lived server process instead of a short-lived CLI invocation.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_session_factory: async_sessionmaker[AsyncSession] | None = None


def configure_session_factory(factory: async_sessionmaker[AsyncSession]) -> None:
    global _session_factory
    _session_factory = factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if _session_factory is None:
        raise RuntimeError(
            "session factory not configured -- app startup (lifespan) did not run"
        )
    async with _session_factory() as session:
        yield session


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """For code that isn't part of FastAPI's request-scoped dependency
    cycle and so can't use `Depends(get_session)` -- Milestone 21's
    webhook `BackgroundTasks` job is the first caller: it runs after the
    HTTP response has already been returned, opening its own session
    from the same factory every request handler shares.
    """
    if _session_factory is None:
        raise RuntimeError(
            "session factory not configured -- app startup (lifespan) did not run"
        )
    return _session_factory
