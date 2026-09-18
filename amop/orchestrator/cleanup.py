"""Resource cleanup -- spec Section 9.7.1 (orphan container/process
reaping) plus the host-side scratch-directory leak Milestone 25 found
and named as bigger and older than it. Milestone 31.

This is NOT an extension of reconcile.py, deliberately -- checked
directly before writing anything here. reconcile() is a per-task git/DB
reconciliation function, only ever called by resume_fix(), only ever
triggered by a human running `amop resume <task_id>` against ONE
already-known stranded task. There is no persistent "orchestrator"
process anywhere in this codebase to hook an automatic startup sweep
into -- every command (`amop fix`, `amop resume`, `amop evaluate`, ...)
is its own short-lived process. `amop watch`'s poll loop is the one
genuinely long-running process this system has, so it's the closest
real analog to "orchestrator startup"; `amop watch` calls
reap_orphan_containers() once before entering its loop (see
cli/main.py). Everything below is also independently callable via its
own CLI command for manual, dry-run-first use -- the live demo doesn't
require `amop watch` to be running at all.

Two independent sweeps, not a shared "cleanup service" (this project's
own standing rule against premature generalization): containers and
scratch directories have different correct triggers and, per an
explicit human decision, different safety postures for a non-terminal
task -- a stale container is disposable (resume_fix() always creates a
fresh one regardless), but a non-terminal task's scratch directory is
exactly the state a future `amop resume` depends on and is never
reaped by age, only by its task having genuinely ended. They share one
small pure lookup (task state, keyed by directory/label name) because
literally duplicating that query would be worse than the small, real
overlap -- not because they're secretly one framework.
"""

import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import docker
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from amop.database.models import Task
from amop.orchestrator.state_machine import TERMINAL_STATES, TaskState
from amop.sandbox import tools as sandbox_tools
from amop.sandbox.manager import MAX_LIFETIME_SECONDS, TASK_LABEL


async def _task_states(session: AsyncSession, task_ids: list[str]) -> dict[str, str]:
    """task_id (as a string) -> tasks.state, for every id in `task_ids`
    that's a real UUID and a real row. A non-UUID id (a handful of this
    project's own test/demo containers use hand-picked labels like
    "t-stale-real", never a real Task row anyway) is silently excluded,
    same as a UUID with no matching row -- both correctly resolve to
    "unknown" for the caller."""
    valid: list[uuid.UUID] = []
    for tid in task_ids:
        try:
            valid.append(uuid.UUID(tid))
        except (ValueError, AttributeError):
            continue
    if not valid:
        return {}
    rows = (
        await session.execute(select(Task.id, Task.state).where(Task.id.in_(valid)))
    ).all()
    return {str(tid): state for tid, state in rows}


def _parse_docker_timestamp(created: str) -> datetime:
    # Docker's own "Created" field: real wall-clock ISO 8601, often with
    # nanosecond precision and a trailing "Z" -- both handled directly
    # by datetime.fromisoformat on Python 3.11+, confirmed rather than
    # assumed (3.9/3.10 would need normalizing this by hand; this
    # project's venv is 3.12).
    return datetime.fromisoformat(created)


# =======================================================================
# 1. Orphan sandbox container reaping (spec 9.7.1)
# =======================================================================


@dataclass
class ContainerReapCandidate:
    task_id: str
    container_id: str
    container_name: str
    reason: str  # "terminal" | "unknown_task" | "stale_non_terminal"
    age_seconds: float


async def find_orphan_containers(session: AsyncSession) -> list[ContainerReapCandidate]:
    """Real Docker containers carrying amop.task_id, cross-referenced
    against real Postgres task state. Never destroys anything -- pure
    read, safe to call any time, the dry-run half of this sweep.

    A container is a candidate when:
      - its task has reached a terminal state (TERMINAL_STATES) --
        definitely done; a task that finished normally already went
        through run_fix()/resume_fix()'s own `finally:
        manager.destroy(...)`, so anything still here for a terminal
        task never reached that finally block at all (e.g. SIGKILL).
      - its task_id matches no real tasks row ("unknown", spec's own
        words for this case).
      - its task is non-terminal but the container is older than
        MAX_LIFETIME_SECONDS (sandbox/manager.py's own already-
        calibrated number, reused rather than a second invented
        threshold) -- resume_fix() always creates a brand-new container
        regardless of what the old one looked like, so an old one is
        disposable once no legitimate run could plausibly still be
        using it. A FRESH non-terminal task's container -- the one a
        real, currently-running `amop fix`/`amop resume` process is
        actively using -- is never a candidate.
    """
    client = docker.from_env()
    containers = client.containers.list(all=True, filters={"label": TASK_LABEL})

    task_ids = [c.labels.get(TASK_LABEL) for c in containers if c.labels.get(TASK_LABEL)]
    states = await _task_states(session, task_ids)

    now = time.time()
    candidates = []
    for c in containers:
        task_id = c.labels.get(TASK_LABEL)
        if not task_id:
            continue
        created = c.attrs.get("Created")
        age = (
            (now - _parse_docker_timestamp(created).timestamp())
            if created
            else float("inf")  # can't determine age -- treat as arbitrarily old, never as fresh
        )
        state = states.get(task_id)
        if state is None:
            reason = "unknown_task"
        elif TaskState(state) in TERMINAL_STATES:
            reason = "terminal"
        elif age > MAX_LIFETIME_SECONDS:
            reason = "stale_non_terminal"
        else:
            continue  # non-terminal and fresh -- never a candidate
        candidates.append(
            ContainerReapCandidate(task_id, c.id, c.name, reason, age)
        )
    return candidates


def reap_containers(candidates: list[ContainerReapCandidate]) -> int:
    """The real-deletion half. Takes candidates already produced (and,
    in real use, already shown to a human) by find_orphan_containers --
    does not re-derive them, so it can never destroy something it
    didn't just tell someone it was about to. Returns how many were
    actually removed; a container already gone by the time this runs
    (a race with something else cleaning up) is not an error."""
    client = docker.from_env()
    removed = 0
    for candidate in candidates:
        try:
            container = client.containers.get(candidate.container_id)
            container.remove(force=True)
            removed += 1
        except docker.errors.NotFound:
            pass
    return removed


# =======================================================================
# 2. Scratch-directory cleanup (going-forward half is
#    SandboxManager.destroy() itself, see sandbox/manager.py -- this is
#    the one-time sweep for the ~100+ directories that already leaked
#    before that fix existed)
# =======================================================================


@dataclass
class ScratchDirReapCandidate:
    task_id: str
    path: Path
    reason: str  # "terminal" | "unknown_task"
    size_bytes: int


def _dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue  # a file removed mid-walk (rare) shouldn't abort sizing the rest
    return total


async def find_stale_scratch_dirs(session: AsyncSession) -> list[ScratchDirReapCandidate]:
    """Every directory directly under SCRATCH_DIR whose name is a real
    task_id, cross-referenced the same way as find_orphan_containers --
    but deliberately NOT age-gated for a non-terminal task (an explicit,
    separate decision from the container sweep): a non-terminal task's
    scratch directory is exactly the state a future `amop resume`
    reads, and unlike a running container it costs only disk to leave
    alone, not CPU/memory/a live Docker resource. A directory whose name
    isn't a real UUID at all (this project's own hand-named
    investigation checkpoints from Milestones 10-12, `m10-baseline-...`
    etc. -- never task-driven, never resumable) is reported separately,
    not silently swept in as "unknown_task", since a human should
    recognize those by name rather than have them look identical to a
    genuinely orphaned real task.
    """
    root = Path(sandbox_tools.SCRATCH_DIR)
    if not root.is_dir():
        return []

    all_dirs = [d for d in root.iterdir() if d.is_dir()]
    uuid_named = []
    for d in all_dirs:
        try:
            uuid.UUID(d.name)
            uuid_named.append(d)
        except ValueError:
            continue  # not a real task_id-named directory -- see non_task_dirs below

    states = await _task_states(session, [d.name for d in uuid_named])

    candidates = []
    for d in uuid_named:
        state = states.get(d.name)
        if state is None:
            reason = "unknown_task"
        elif TaskState(state) in TERMINAL_STATES:
            reason = "terminal"
        else:
            continue  # non-terminal -- never a candidate, no age exception
        candidates.append(ScratchDirReapCandidate(d.name, d, reason, _dir_size(d)))
    return candidates


def non_task_scratch_dirs() -> list[Path]:
    """Directories under SCRATCH_DIR whose name is NOT a real task_id at
    all -- reported for a human to look at, never auto-swept by this
    milestone's own sweep (find_stale_scratch_dirs above deliberately
    excludes them). Read-only."""
    root = Path(sandbox_tools.SCRATCH_DIR)
    if not root.is_dir():
        return []
    result = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        try:
            uuid.UUID(d.name)
        except ValueError:
            result.append(d)
    return result


def reap_scratch_dirs(candidates: list[ScratchDirReapCandidate]) -> int:
    """The real-deletion half, same contract as reap_containers: acts
    only on candidates already produced (and shown to a human) by
    find_stale_scratch_dirs, never re-derives them."""
    removed = 0
    for candidate in candidates:
        if candidate.path.is_dir():
            shutil.rmtree(candidate.path, ignore_errors=True)
            removed += 1
    return removed
