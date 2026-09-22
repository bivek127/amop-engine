"""Resource Cleanup Debt -- spec 9.7.1 (orphan container/process
reaping) plus Milestone 20's stale code_chunks and Milestone 25's
scratch-directory leak. Milestone 31.

Real Docker daemon and real Postgres (TEST_DATABASE_URL) required --
same discipline as every other sandbox-level test in this project.
"""

import asyncio
import os
import time
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select, text

from amop.codebase_intel.indexer import find_stale_code_chunk_paths, prune_stale_code_chunks
from amop.database.models import CodeChunk, Repository
from amop.database.session import init_db, make_engine, make_session_factory
from amop.orchestrator import cleanup as cleanup_mod
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import create_task, transition_with_retry
from amop.sandbox import tools as sandbox_tools
from amop.sandbox.manager import TASK_LABEL, SandboxManager

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://localhost/amop_test"
)


# =======================================================================
# Postgres fixtures
# =======================================================================


@pytest_asyncio.fixture
async def engine():
    eng = make_engine(TEST_DATABASE_URL)
    await init_db(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    session_factory = make_session_factory(engine)
    async with session_factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE agent_actions, task_transitions, tasks, "
                "code_chunks, repositories RESTART IDENTITY CASCADE"
            )
        )


@pytest.fixture
def sandbox_manager():
    manager = SandboxManager()
    yield manager
    manager.destroy_all()


async def _task_in(session, state) -> object:
    task = await create_task(session, task_type="bug_fix")
    if state is TaskState.CREATED:
        return task
    path = {
        TaskState.FAILED: [
            TaskState.TRIAGING, TaskState.INVESTIGATING, TaskState.PLANNING_FIX,
            TaskState.CODING, TaskState.TESTING, TaskState.FAILED,
        ],
        TaskState.NEEDS_HUMAN_INPUT: [
            TaskState.TRIAGING, TaskState.NEEDS_HUMAN_INPUT,
        ],
        TaskState.CODING: [
            TaskState.TRIAGING, TaskState.INVESTIGATING, TaskState.PLANNING_FIX, TaskState.CODING,
        ],
    }[state]
    for to_state in path:
        task = await transition_with_retry(session, task, to_state, actor="test")
    return task


# =======================================================================
# 1. Stale code_chunks pruning
# =======================================================================


async def test_stale_path_not_registered_and_not_on_disk_is_flagged(session, tmp_path):
    dead_path = str(tmp_path / "long-gone-repo")  # never created -- doesn't exist
    session.add(
        CodeChunk(
            repo_path=dead_path, file_path="x.py", symbol_name="f", symbol_type="function",
            start_line=1, end_line=2, content="def f(): pass", embedding=[0.0] * 768,
        )
    )
    await session.commit()

    stale = await find_stale_code_chunk_paths(session)

    assert any(p == dead_path for p, _ in stale)
    
async def test_real_unregistered_path_that_still_exists_is_not_flagged(session, tmp_path):
    """The real nuance found against the actual dev database: a
    legitimate ad-hoc fixture path (never registered via `amop repos
    add`, but still real on disk) must never be flagged -- registry
    membership alone is not the right criterion."""
    real_path = tmp_path / "real-adhoc-fixture"
    real_path.mkdir()
    session.add(
        CodeChunk(
            repo_path=str(real_path), file_path="x.py", symbol_name="f", symbol_type="function",
            start_line=1, end_line=2, content="def f(): pass", embedding=[0.0] * 768,
        )
    )
    await session.commit()

    stale = await find_stale_code_chunk_paths(session)

    assert not any(p == str(real_path) for p, _ in stale)


async def test_registered_path_is_never_flagged_even_if_temporarily_missing(session, tmp_path):
    """Milestone 20's own registry says this path is real; a directory
    that's merely temporarily unavailable (network drive unmounted,
    etc.) must not be pruned out from under it."""
    missing_but_registered = str(tmp_path / "temporarily-unavailable")
    session.add(Repository(repo_path=missing_but_registered, index_status="ready"))
    session.add(
        CodeChunk(
            repo_path=missing_but_registered, file_path="x.py", symbol_name="f",
            symbol_type="function", start_line=1, end_line=2, content="def f(): pass",
            embedding=[0.0] * 768,
        )
    )
    await session.commit()

    stale = await find_stale_code_chunk_paths(session)

    assert not any(p == missing_but_registered for p, _ in stale)


async def test_prune_stale_code_chunks_deletes_only_the_given_paths(session, tmp_path):
    dead = str(tmp_path / "dead")
    kept = str(tmp_path / "kept")
    Path(kept).mkdir()
    for path in (dead, kept):
        session.add(
            CodeChunk(
                repo_path=path, file_path="x.py", symbol_name="f", symbol_type="function",
                start_line=1, end_line=2, content="def f(): pass", embedding=[0.0] * 768,
            )
        )
    await session.commit()

    deleted = await prune_stale_code_chunks(session, [dead])

    assert deleted == 1
    remaining = (
        await session.execute(select(CodeChunk.repo_path))
    ).scalars().all()
    assert dead not in remaining
    assert kept in remaining


# =======================================================================
# 2. Orphan sandbox container reaping -- the rigor-matters section
# =======================================================================


async def test_terminal_tasks_container_is_a_reap_candidate(session, sandbox_manager, tmp_path):
    task = await _task_in(session, TaskState.FAILED)
    sandbox = sandbox_manager.create(str(task.id), tmp_path)

    candidates = await cleanup_mod.find_orphan_containers(session)

    matches = [c for c in candidates if c.task_id == str(task.id)]
    assert len(matches) == 1
    assert matches[0].reason == "terminal"


async def test_unknown_tasks_container_is_a_reap_candidate(session, sandbox_manager, tmp_path):
    fake_task_id = str(uuid.uuid4())  # a real UUID, but no tasks row
    sandbox_manager.create(fake_task_id, tmp_path)

    candidates = await cleanup_mod.find_orphan_containers(session)

    matches = [c for c in candidates if c.task_id == fake_task_id]
    assert len(matches) == 1
    assert matches[0].reason == "unknown_task"


async def test_stale_non_terminal_tasks_container_is_a_reap_candidate(
    session, sandbox_manager, tmp_path, monkeypatch
):
    """A non-terminal task's container CAN be reaped once old enough
    that no legitimate run could still be using it -- resume_fix()
    always builds a brand-new container regardless."""
    monkeypatch.setattr(cleanup_mod, "MAX_LIFETIME_SECONDS", 0)
    task = await _task_in(session, TaskState.CODING)
    sandbox_manager.create(str(task.id), tmp_path)
    time.sleep(1.1)  # real wall-clock age, past the 0s threshold just set

    candidates = await cleanup_mod.find_orphan_containers(session)

    matches = [c for c in candidates if c.task_id == str(task.id)]
    assert len(matches) == 1
    assert matches[0].reason == "stale_non_terminal"


async def test_a_genuinely_active_fresh_non_terminal_tasks_container_is_never_a_candidate(
    session, sandbox_manager, tmp_path
):
    """THE test that actually matters, per its own explicit standing:
    a false-positive reap of live work is a real safety regression, not
    a cosmetic bug. A real, non-terminal task with a real, freshly-
    created container -- exactly what a currently-running `amop fix`
    looks like from the outside -- must NEVER be a reap candidate,
    under the real, unmodified MAX_LIFETIME_SECONDS default. Mutation-
    tested (see the session's own verification, not just asserted
    here): removing the non-terminal/age guard entirely and rerunning
    this exact test is how that claim was checked, not assumed.
    """
    task = await _task_in(session, TaskState.CODING)
    sandbox_manager.create(str(task.id), tmp_path)

    candidates = await cleanup_mod.find_orphan_containers(session)

    assert not any(c.task_id == str(task.id) for c in candidates)


async def test_reap_containers_removes_only_the_given_candidates(
    session, sandbox_manager, tmp_path
):
    keep_task = await _task_in(session, TaskState.CODING)
    keep_sandbox = sandbox_manager.create(str(keep_task.id), tmp_path)
    dead_task = await _task_in(session, TaskState.FAILED)
    dead_sandbox = sandbox_manager.create(str(dead_task.id), tmp_path)

    candidates = await cleanup_mod.find_orphan_containers(session)
    dead_candidates = [c for c in candidates if c.task_id == str(dead_task.id)]
    assert dead_candidates  # sanity: the fixture we're about to reap is really a candidate

    removed = cleanup_mod.reap_containers(dead_candidates)

    assert removed == 1
    # The dead one is really gone from Docker, not just from our list.
    import docker
    client = docker.from_env()
    with pytest.raises(docker.errors.NotFound):
        client.containers.get(dead_sandbox.container.id)
    # The active one was never touched.
    client.containers.get(keep_sandbox.container.id)  # raises if it's gone


# =======================================================================
# 3. Scratch-directory cleanup
# =======================================================================


async def test_sandbox_manager_destroy_leaves_host_dir_by_default(sandbox_manager, tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "marker.txt").write_text("x")
    sandbox = sandbox_manager.create("t-default", scratch)

    sandbox_manager.destroy("t-default")

    assert scratch.is_dir()  # unchanged: the pre-existing, safe default


async def test_sandbox_manager_destroy_removes_host_dir_when_opted_in(sandbox_manager, tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "marker.txt").write_text("x")
    sandbox_manager.create("t-optin", scratch)

    sandbox_manager.destroy("t-optin", remove_scratch_dir=True)

    assert not scratch.exists()


async def test_find_stale_scratch_dirs_flags_a_terminal_tasks_directory(session, monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox_tools, "SCRATCH_DIR", tmp_path)
    task = await _task_in(session, TaskState.FAILED)
    (tmp_path / str(task.id)).mkdir()

    candidates = await cleanup_mod.find_stale_scratch_dirs(session)

    matches = [c for c in candidates if c.task_id == str(task.id)]
    assert len(matches) == 1
    assert matches[0].reason == "terminal"


async def test_find_stale_scratch_dirs_flags_an_unknown_tasks_directory(session, monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox_tools, "SCRATCH_DIR", tmp_path)
    fake_id = str(uuid.uuid4())
    (tmp_path / fake_id).mkdir()

    candidates = await cleanup_mod.find_stale_scratch_dirs(session)

    matches = [c for c in candidates if c.task_id == fake_id]
    assert len(matches) == 1
    assert matches[0].reason == "unknown_task"


async def test_find_stale_scratch_dirs_never_flags_a_non_terminal_task_even_when_old(
    session, monkeypatch, tmp_path
):
    """The scratch-dir equivalent of the container test above, with a
    deliberately DIFFERENT answer to the age question (an explicit,
    separate human decision, not an oversight): a non-terminal task's
    directory is NEVER reaped by this sweep, no matter how old --
    that's exactly the state a future `amop resume` depends on, and
    unlike a container it costs only disk to leave alone."""
    monkeypatch.setattr(sandbox_tools, "SCRATCH_DIR", tmp_path)
    task = await _task_in(session, TaskState.CODING)
    scratch = tmp_path / str(task.id)
    scratch.mkdir()
    old_time = time.time() - 999_999  # far older than any container threshold
    os.utime(scratch, (old_time, old_time))

    candidates = await cleanup_mod.find_stale_scratch_dirs(session)

    assert not any(c.task_id == str(task.id) for c in candidates)


async def test_non_task_scratch_dirs_are_listed_separately_never_auto_swept(
    session, monkeypatch, tmp_path
):
    monkeypatch.setattr(sandbox_tools, "SCRATCH_DIR", tmp_path)
    hand_named = tmp_path / "m10-baseline-abc123"
    hand_named.mkdir()

    others = cleanup_mod.non_task_scratch_dirs()
    candidates = await cleanup_mod.find_stale_scratch_dirs(session)

    assert hand_named in others
    assert not any(c.path == hand_named for c in candidates)


async def test_reap_scratch_dirs_removes_only_the_given_candidates(session, monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox_tools, "SCRATCH_DIR", tmp_path)
    keep_task = await _task_in(session, TaskState.CODING)
    (tmp_path / str(keep_task.id)).mkdir()
    dead_task = await _task_in(session, TaskState.FAILED)
    dead_dir = tmp_path / str(dead_task.id)
    dead_dir.mkdir()

    candidates = await cleanup_mod.find_stale_scratch_dirs(session)
    dead_candidates = [c for c in candidates if c.task_id == str(dead_task.id)]
    assert dead_candidates

    removed = cleanup_mod.reap_scratch_dirs(dead_candidates)

    assert removed == 1
    assert not dead_dir.exists()
    assert (tmp_path / str(keep_task.id)).is_dir()
