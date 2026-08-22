import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
)
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

    # Section 4.3.1's OCC primitive, Milestone 16. SQLAlchemy's own
    # version_id_col emits exactly the statement the spec specifies --
    # UPDATE tasks SET ..., version = version + 1 WHERE id = ? AND
    # version = ? -- and raises StaleDataError when zero rows match.
    #
    # Declared on the mapper rather than hand-written into transition()
    # deliberately: this way EVERY write to a Task row is guarded, not
    # just the one function someone remembered to protect. chain.py and
    # cli/main.py both assign task.task_context after a run and commit;
    # those are lost-update candidates too, and they are covered here
    # without either file changing.
    __mapper_args__ = {"version_id_col": version}


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


class MemoryItem(Base):
    """Spec Section 14.2's `memory_items` table -- long-term incident
    memory (Section 10.1's third layer, 10.2's write-on-resolution).

    Two deliberate deviations from 14.2's literal DDL, both following
    precedent this codebase already set rather than inventing a third
    convention:

      * `repo_path: TEXT` instead of `repo_id UUID REFERENCES
        repositories(id)` -- there is no `repositories` table (Task
        dropped repo_id for the same reason at Milestone 1). CodeChunk
        above already uses the resolved source repo path as its stable
        repo identity; memory uses the same key so the two can be
        filtered consistently.
      * `Vector(EMBEDDING_DIM)` = 768, not the spec's VECTOR(1536).
        Section 7.5/D-6 says code and prose share ONE embedding model,
        and that model (nomic-embed-text) is 768-dimensional -- verified
        live, see EMBEDDING_DIM's comment. 1536 would be a column no
        vector this project produces could ever be stored in.

    `disputed` is Section 10.4's correctness escape hatch: a human can
    mark a memory wrong (a root cause that turned out not to be the
    cause) and retrieval excludes it -- but the row is NEVER deleted,
    it's preserved for audit. Indexed alongside repo_path because every
    retrieval query filters on both.

    No ANN index (ivfflat/hnsw), same reasoning as CodeChunk's: at this
    scale a sequential scan over the cosine distance is fast enough, and
    ivfflat on a near-empty table would actively hurt recall.
    """

    __tablename__ = "memory_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    repo_path: Mapped[str] = mapped_column(Text, nullable=False)
    # Deliberately a plain UUID, NOT a ForeignKey("tasks.id"), and that
    # is a real decision rather than an oversight: Section 14.2's
    # retention policy keeps tasks/agent_messages for 90 days while
    # Section 10.4 keeps memory_items indefinitely. A hard FK inverts
    # that -- pruning a 90-day-old task would either be blocked by the
    # constraint or cascade-delete the very memory that was supposed to
    # outlive it. The id is still recorded for provenance; it just isn't
    # a referential guarantee, because the referent is the shorter-lived
    # of the two by design. (Caught by a test writing memory for a task
    # that no longer existed -- exactly the pruning case, arriving early.)
    task_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    # Section 14.2: incident_resolution | codebase_note | preference.
    # Only incident_resolution is written this milestone; the column is
    # the spec's, kept general so the other two don't need a migration.
    memory_type: Mapped[str] = mapped_column(Text, nullable=False)
    # Section 10.2's structured summary:
    # {anomaly_signature, root_cause, fix_summary, outcome}
    content: Mapped[dict] = mapped_column(JSONB, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    disputed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_memory_items_repo_path", "repo_path"),
        Index("ix_memory_items_disputed", "disputed"),
    )


class Repository(Base):
    """Milestone 15's `GET/POST /repositories` -- an optional registry,
    NOT the relational hub spec 14.2 imagines. Every existing table
    (Task.task_context["repo"], CodeChunk.repo_path,
    MemoryItem.repo_path) already keys on a free-form repo path string;
    building the spec's normalized `repo_id UUID` this milestone would
    mean migrating three tables' worth of existing data and every reader
    of them, for an API surface that only needs a way to list/register
    repos by name. So this is additive only: a place to register a repo
    path with a display name, queryable on its own, leaving every
    existing repo_path column exactly as it is.
    """

    __tablename__ = "repositories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    repo_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    display_name: Mapped[str | None] = mapped_column(Text)

    # --- Milestone 20: spec 14.2's remaining columns -------------------
    # All nullable/defaulted deliberately: every column here is additive,
    # so rows written by Milestone 15 stay valid and every existing query
    # keeps working untouched. `tasks.repo_id` (14.2's foreign key) is
    # still NOT built -- see this class's docstring above for why, and
    # Milestone 20's own reasoning: per-repo permission lookup resolves
    # fine through `repo_path`, so the three-table migration buys nothing
    # this milestone needs.
    #
    # `url` vs `repo_path` -- both, because they are genuinely different
    # identities and this codebase uses both. Spec 14.2 has only `url`
    # because it assumes AMOP clones the repo itself; AMOP as built works
    # from a local checkout. Concretely, `task_context["repo"]` already
    # holds a LOCAL PATH when a task comes from `amop fix`, but a GITHUB
    # SLUG ("owner/name") when it comes from Watcher -- and the lock keys
    # (orchestrator/concurrency.py) plus the RAG index
    # (codebase_intel/indexer.py) both key on the resolved local path.
    # So `repo_path` stays the working identity and the unique key;
    # `url` is the canonical one you watch and open PRs against.
    url: Mapped[str | None] = mapped_column(Text)
    default_branch: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="main"
    )
    detected_stack: Mapped[dict | None] = mapped_column(JSONB)  # Section 6.3.4
    index_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="unindexed"
    )  # unindexed | indexing | ready | stale
    # Section 12.1 / D-8. Shape (spec 12.1's own worked example):
    #   {"default": "observer", "agents": {"dependency_updater": "autonomous"}}
    permission_overrides: Mapped[dict | None] = mapped_column(JSONB)

    # Milestone 21 / spec 21's `github.webhook_secret_env`. Per-repo
    # override of the global GITHUB_WEBHOOK_SECRET env var -- same
    # global-default-with-per-entity-override shape D-8 already
    # established for permission_overrides, reused rather than inventing
    # a second precedence pattern. Never a default: a repo with neither
    # this nor the env var set rejects webhook traffic outright (see
    # api/routes/webhooks.py) rather than silently accepting unsigned
    # requests.
    webhook_secret: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ProcessedEvent(Base):
    """Spec Section 4.3.2 / Design Decision D-17's `processed_events`
    table -- consumer-side idempotency for at-least-once delivery.

    Milestone 21's scope decision (CLAUDE.md, reaffirmed here, not
    relitigated): the spec's original design puts this table downstream
    of a Redis Streams consumer group. This project has deliberately not
    built Redis (ADR-04/ADR-06). For a single-process receiver handling
    webhook POSTs directly, this table alone provides the same guarantee
    spec 4.3.2 describes -- a unique-constraint violation on retry means
    "already handled," full stop -- without a queue in front of it.

    `event_key` here is GitHub's own `X-GitHub-Delivery` header value
    directly, not spec's `sha256(source + external_id + payload_digest)`
    formula. Simplification, not a shortcut: GitHub already guarantees
    that header is unique per delivery attempt, so hashing it (or
    combining it with a payload digest) adds no additional collision
    resistance -- it would just be hashing an already-unique value.
    """

    __tablename__ = "processed_events"

    event_key: Mapped[str] = mapped_column(Text, primary_key=True)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PullRequest(Base):
    """Milestone 15's `GET /pull-requests`. No PR-tracking table existed
    before this -- a PR's URL has only ever been a transient field on a
    ChainResult, printed to CLI stdout and discarded. Written once, the
    one place a PR is ever actually opened
    (orchestrator/chain.py's PR_CREATION stage), so this list needs no
    live call back to GitHub to answer "what PRs has AMOP opened."

    `task_id` has no FK constraint, same reasoning as MemoryItem.task_id
    above -- a PR record should outlive the task's own retention window.
    """

    __tablename__ = "pull_requests"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    repo_path: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    number: Mapped[int | None] = mapped_column(Integer)
    # open | merged | closed -- set to "open" when written (the only
    # state create_pull_request's own return value can attest to); this
    # milestone has no poller updating it afterward (that needs
    # get_ci_status/webhook wiring, explicitly queued separately).
    status: Mapped[str] = mapped_column(Text, nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (Index("ix_pull_requests_repo_path", "repo_path"),)
