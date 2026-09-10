import os

from dotenv import load_dotenv
from sqlalchemy import text
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
    """Create the vector extension (Milestone 5's code_chunks.embedding
    column needs it) and all tables. Fine for this milestone — real
    Alembic migrations can wait until the schema needs to evolve.

    CREATE EXTENSION IF NOT EXISTS is idempotent and safe to run on every
    startup -- this makes pgvector self-serve for dev and test databases
    alike instead of a manual step the human has to remember to repeat
    per database. It has to run before create_all(): the Vector column
    type can't be created against a database that doesn't have the
    extension yet.
    """
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        for statement in _ADDITIVE_COLUMNS:
            await conn.execute(text(statement))


# create_all() creates missing TABLES; it never alters an existing one.
# So a column added to a model after its table already exists is simply
# absent from any database created earlier -- and the failure mode is
# nasty: the model claims the column, every INSERT naming it errors, and
# only on the older database. These statements close that gap.
#
# ADD COLUMN IF NOT EXISTS is idempotent, so this is a no-op on a fresh
# database (create_all already made the column) and a one-time fix on an
# existing one. That keeps both converged with no manual step and no
# fresh-clone-vs-dev-database divergence.
#
# This is the "until the schema needs to evolve" moment init_db's own
# docstring anticipated. It is deliberately limited to ADDITIVE,
# nullable columns: anything that rewrites or drops data belongs in a
# real migration tool, not here.
_ADDITIVE_COLUMNS = (
    "ALTER TABLE agent_actions ADD COLUMN IF NOT EXISTS input_tokens INTEGER",
    "ALTER TABLE agent_actions ADD COLUMN IF NOT EXISTS output_tokens INTEGER",
)
