"""Section 4.3 / 4.3.1 concurrency primitives — Milestone 16.

Two mechanisms with deliberately different scopes:

* **Advisory locks** (Postgres, durable, cross-process) serialize work
  that would touch the same files in the same repo.
* **Slot limits** (in-process counters) bound how much runs at once.

Why the limits are counted in-process rather than from `tasks` rows:
Section 4.5 defines the bound as "asyncio tasks per active Task row,
bounded by max_concurrent_tasks_global" — an in-process bound, correct
under ADR-04/ADR-06's single-process model. Counting non-terminal rows
instead would be a real bug, not a stylistic choice: this project's own
dev database currently holds ~16 tasks stranded in CODING/TESTING by
interrupted runs, and a row-counting limiter would read those as live
work and refuse everything forever. Reaping them is Section 4.6.2's
crash-recovery reconciliation, which is separately queued and NOT part
of this milestone.

Why Redis is not here, re-derived rather than inherited (ADR-12): Redlock
solves multi-host contention, which is a Non-Goal (NG2) under ADR-04/06,
while adding a lease lifecycle — acquire, renew, expire, detect loss
mid-execution. A 20-minute CODING stage would have to answer "my lease
expired mid-run_tests, does another worker own this repo now?". Postgres
advisory locks have no lease: they are held by a connection and die with
it, which was verified against a real server (SIGKILL the holder, the
lock is immediately available) rather than taken on faith.
"""

import asyncio
import functools
import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

# Section 4.3's own defaults, via the established AMOP_* convention
# (safety/circuit_breakers.py:53-80 sets the pattern).
DEFAULT_MAX_CONCURRENT_TASKS_GLOBAL = 4
DEFAULT_MAX_CONCURRENT_TASKS_PER_REPO = 1

MAX_CONCURRENT_TASKS_GLOBAL = int(
    os.environ.get(
        "AMOP_MAX_CONCURRENT_TASKS_GLOBAL", str(DEFAULT_MAX_CONCURRENT_TASKS_GLOBAL)
    )
)
MAX_CONCURRENT_TASKS_PER_REPO = int(
    os.environ.get(
        "AMOP_MAX_CONCURRENT_TASKS_PER_REPO", str(DEFAULT_MAX_CONCURRENT_TASKS_PER_REPO)
    )
)


class ConcurrencyLimitExceeded(Exception):
    """A task start was refused because a limit was already saturated.

    Raised, never swallowed: the brief's own requirement is that a task
    refused for concurrency reasons gets an honest status rather than
    silently disappearing.
    """

    def __init__(self, scope: str, limit: int, repo_path: str | None = None) -> None:
        self.scope = scope
        self.limit = limit
        self.repo_path = repo_path
        where = f" for repo {repo_path}" if repo_path else ""
        super().__init__(
            f"{scope} concurrency limit reached ({limit} already running{where}) "
            f"-- refusing to start another task now"
        )


# ---------------------------------------------------------------------
# Lock keys
# ---------------------------------------------------------------------


def _key(*parts: str) -> int:
    """A stable signed 64-bit key for pg_advisory_xact_lock(bigint).

    blake2b rather than Python's hash(): hash() is randomized per process
    by PYTHONHASHSEED, so two orchestrator processes -- or the same one
    restarted mid-run -- would derive different keys for the same repo
    and fail to exclude each other. A lock that silently stops locking
    after a restart is exactly the failure this milestone exists to
    prevent.
    """
    digest = hashlib.blake2b("\x00".join(parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def normalize_repo(repo_path: str) -> str:
    return str(Path(repo_path).resolve())


def repo_lock_key(repo_path: str) -> int:
    return _key("amop:repo", normalize_repo(repo_path))


def file_lock_keys(repo_path: str, files) -> list[int]:
    """One key per declared file, sorted.

    Sorted is not cosmetic -- it is what makes this deadlock-free. Two
    tasks declaring {a, b} and {b, a} that grabbed locks in their own
    order could each hold one and wait forever for the other; a total
    order on acquisition makes that impossible.

    File granularity (rather than the single hash(repo_id) key Section
    4.3 names) is what lets same-repo tasks on disjoint files run in
    parallel, which Section 4.3 also explicitly requires. A repo-grained
    key cannot express that: deciding "do our file sets overlap?" and
    then locking the repo has a check-then-act hole where two tasks both
    observe no conflict and both proceed. Locks ARE the exclusion here,
    so there is no window between the check and the act. Section 29.1
    files file-level locking as the v2.1 direction; it is brought forward
    because the repo-grained alternative cannot satisfy this milestone's
    own parallelism requirement without that hole.
    """
    from amop.orchestrator.chain import normalize_repo_path  # circular at module scope

    repo = normalize_repo(repo_path)
    return sorted(
        {_key("amop:file", repo, normalize_repo_path(f)) for f in files if f and f.strip()}
    )


# ---------------------------------------------------------------------
# Advisory locking
# ---------------------------------------------------------------------


def engine_for(session) -> AsyncEngine:
    """Recover the AsyncEngine behind an AsyncSession.

    `session.get_bind()` hands back the *sync* Engine that the async one
    wraps; this maps it back to the identical AsyncEngine object (checked:
    `recovered is original`). Used so locking needs no new parameter
    threaded through run_fix/run_chain and every call site -- and so the
    lock connection comes from the SAME pool the caller is already using,
    rather than a second pool quietly opening more connections to
    Postgres than anyone configured.
    """
    return AsyncEngine._retrieve_proxy_for_target(session.get_bind())


@asynccontextmanager
async def repo_file_lock(engine: AsyncEngine, repo_path: str, files=()):
    """Hold advisory locks across the caller's whole critical section.

    On a DEDICATED connection, not the caller's session. That is the
    load-bearing detail: transition() commits on every state change, and
    a transaction-scoped lock taken on the chain's own session would be
    released by the very next hop -- CODING -> TESTING would silently
    drop the lock mid-stage. A separate connection holding one open
    transaction is unaffected by the session's commits (verified against
    a real server), so the lock survives exactly as long as intended.

    pg_advisory_xact_lock, so release is structural rather than
    something a finally-block has to get right: commit, rollback, or the
    process dying all free it. A crashed orchestrator cannot leave a repo
    permanently locked.
    """
    keys = file_lock_keys(repo_path, files) or [repo_lock_key(repo_path)]
    conn = await engine.connect()
    try:
        trans = await conn.begin()
        try:
            for key in keys:
                await conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": key})
            yield keys
        finally:
            # Nothing was written; rollback releases the locks just as
            # commit would, without pretending this was a real write.
            await trans.rollback()
    finally:
        await conn.close()


async def try_repo_file_lock(engine: AsyncEngine, repo_path: str, files=()) -> bool:
    """Non-blocking probe, for tests and diagnostics. True == was free."""
    keys = file_lock_keys(repo_path, files) or [repo_lock_key(repo_path)]
    async with engine.connect() as conn:
        trans = await conn.begin()
        try:
            for key in keys:
                got = await conn.scalar(
                    text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": key}
                )
                if not got:
                    return False
            return True
        finally:
            await trans.rollback()


# ---------------------------------------------------------------------
# Slot limits
# ---------------------------------------------------------------------

_active_global = 0
_active_per_repo: dict[str, int] = {}
_slot_mutex = asyncio.Lock()


def active_counts() -> tuple[int, dict[str, int]]:
    return _active_global, dict(_active_per_repo)


@asynccontextmanager
async def task_slot(repo_path: str):
    """Reserve a slot for one running task, or refuse the start.

    The release path is in `finally` deliberately: a task that raises
    must give its slot back, or a handful of failures would permanently
    wedge the orchestrator at zero capacity -- a slow-motion outage that
    looks like "AMOP stopped picking things up" rather than a crash.
    """
    global _active_global
    repo = normalize_repo(repo_path)
    async with _slot_mutex:
        # Read the module attributes at call time, not import time, so a
        # test (or a future config reload) can adjust the ceiling.
        if _active_global >= MAX_CONCURRENT_TASKS_GLOBAL:
            raise ConcurrencyLimitExceeded("global", MAX_CONCURRENT_TASKS_GLOBAL)
        if _active_per_repo.get(repo, 0) >= MAX_CONCURRENT_TASKS_PER_REPO:
            raise ConcurrencyLimitExceeded(
                "per-repo", MAX_CONCURRENT_TASKS_PER_REPO, repo_path=repo
            )
        _active_global += 1
        _active_per_repo[repo] = _active_per_repo.get(repo, 0) + 1
    try:
        yield
    finally:
        async with _slot_mutex:
            _active_global -= 1
            remaining = _active_per_repo.get(repo, 1) - 1
            if remaining > 0:
                _active_per_repo[repo] = remaining
            else:
                _active_per_repo.pop(repo, None)


def gated_by_task_slot(fn):
    """Wrap a `run_*` entry point so it holds a concurrency slot.

    Applied at the entry points rather than inside run_chain() so the
    slot also covers container creation and indexing -- the expensive,
    resource-hungry parts -- and so every caller is gated by construction
    (CLI, Watcher, and any future one) instead of each remembering to.

    A refusal propagates as ConcurrencyLimitExceeded rather than being
    swallowed and retried: the brief's own requirement is that a task
    turned away for concurrency reasons says so, instead of silently
    vanishing.
    """

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        repo_path = kwargs.get("repo_path")
        async with task_slot(str(repo_path)):
            return await fn(*args, **kwargs)

    return wrapper
