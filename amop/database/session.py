import os

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from amop.database.base import Base

load_dotenv()


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to a .env file or export it, "
            "e.g. DATABASE_URL=postgresql+asyncpg://localhost/amop_dev"
        )
    return url


def make_engine(database_url: str | None = None) -> AsyncEngine:
    """Build an engine for the given URL, or DATABASE_URL from the
    environment/.env if none is given. Kept as an explicit factory (no
    module-level singleton) so tests can point it at a separate database
    without fighting import-time state.
    """
    return create_async_engine(database_url or get_database_url())


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    """Create all tables. Fine for this milestone — real Alembic
    migrations can wait until the schema needs to evolve."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
