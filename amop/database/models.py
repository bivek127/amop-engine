import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, Numeric, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from amop.database.base import Base

# Section 7.5: code and prose share one embedding model (D-6) --
# nomic-embed-text, confirmed 768-dimensional (verified live against a
# running Ollama, not assumed from documentation). The column's fixed
# dimension ties this schema to that specific model; switching embedding
# models later means a migration, not a config change.
EMBEDDING_DIM = 768


class Task(Base):
    """Spec Section 14.2's `tasks` table, trimmed to what the bug_fix state
    machine needs (Milestone 1). Dropped: repo_id, incident_id,
    cost_usd_total — repos/incidents/cost tracking are later milestones.
    """

    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_type: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False, default="CREATED")
    severity: Mapped[str | None] = mapped_column(Text)
    priority_score: Mapped[float | None] = mapped_column(Numeric)
    task_context: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class TaskTransition(Base):
    """Spec Section 14.2's `task_transitions` table — the append-only
    decision audit trail (Section 4.2)."""

    __tablename__ = "task_transitions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tasks.id")
    )
    from_state: Mapped[str | None] = mapped_column(Text)
    to_state: Mapped[str] = mapped_column(Text, nullable=False)
    trigger: Mapped[str | None] = mapped_column(Text)
    actor: Mapped[str | None] = mapped_column(Text)  # agent:<name> | human:<id> | system
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class CodeChunk(Base):
    """Spec Section 14.2's `code_chunks` table, trimmed to Milestone 5's
    declared column list. `repo_path` is the resolved SOURCE repo path
    (e.g. tests/fixtures/task_tracker), not a per-task scratch directory
    -- a scratch dir is unique per task and would make the index
    unreusable across runs, defeating the point of indexing at all.
    "Index fresh each time" (this milestone's authorized simplification
    over Section 7.1's incremental re-index) is implemented as: delete
    every row for a repo_path, then re-populate, each time index_repo()
    runs for that path.

    No ANN index (ivfflat/hnsw) -- at fixture-repo scale (dozens to low
    hundreds of rows) a plain ORDER BY embedding <=> query sequential
    scan is fast enough; building one is later tuning, not something
    this repo size needs. The plain btree index below is just for the
    repo_path filter every query applies before ranking.
    """

    __tablename__ = "code_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    repo_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    symbol_name: Mapped[str] = mapped_column(Text, nullable=False)
    symbol_type: Mapped[str] = mapped_column(Text, nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (Index("ix_code_chunks_repo_path", "repo_path"),)
