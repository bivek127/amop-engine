"""Milestone 16 — Concurrency Safety.

Adversarial, not illustrative. Every test here was run against the
UNFIXED code first and observed to fail; a concurrency test that has
never been seen failing is not evidence that it can detect anything.
(Milestone 14's lesson, applied deliberately: a profiler test asserted
the hot function was *present* and passed for weeks while its *rank* —
the property that actually mattered — was broken.)

Genuine concurrency, not simulated: every racing coroutine gets its own
AsyncSession, which means its own pooled connection, and an
asyncio.Barrier ensures both writers have LOADED the row before either
is allowed to write. Without that barrier the two coroutines serialize
naturally and the test passes against broken code, proving nothing.

Requires: Postgres at TEST_DATABASE_URL.
"""

import asyncio

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text

from amop.database.models import Task, TaskTransition
from amop.database.session import init_db, make_engine, make_session_factory
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import ConcurrentUpdateError, create_task, transition

TEST_DATABASE_URL = "postgresql+asyncpg://localhost/amop_test"

# CREATED -> ... -> WAITING_FOR_APPROVAL, the real chain order.
_TO_APPROVAL = (
    TaskState.TRIAGING,
    TaskState.INVESTIGATING,
    TaskState.PLANNING_FIX,
    TaskState.CODING,
    TaskState.TESTING,
    TaskState.REVIEWING,
    TaskState.PR_CREATION,
    TaskState.WAITING_FOR_APPROVAL,
)


@pytest_asyncio.fixture
async def engine():
    eng = make_engine(TEST_DATABASE_URL)
    await init_db(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    yield make_session_factory(engine)
    async with engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE task_transitions, tasks RESTART IDENTITY CASCADE")
        )


@pytest_asyncio.fixture
async def session(session_factory):
    async with session_factory() as s:
        yield s


async def _task_at_approval(session):
    task = await create_task(session, task_type="bug_fix", task_context={})
    for state in _TO_APPROVAL:
        await transition(session, task, state)
    return task


async def _count_transitions_from(session, task_id, from_state):
    return await session.scalar(
        select(func.count())
        .select_from(TaskTransition)
        .where(
            TaskTransition.task_id == task_id,
            TaskTransition.from_state == from_state.value,
        )
    )


# ---------------------------------------------------------------------
# A. OCC — two concurrent writers on the SAME task row.
#
# This is not a hypothetical race. Milestone 15 built the second writer:
# `POST /tasks/{id}/approve` and `/reject` (API, Telegram bot, and web
# dashboard all reach them) both call transition() on a task that
# `run_fix()` may still be holding in memory. Two humans -- or one human
# on two surfaces -- approving and rejecting at once is the literal
# scenario reproduced here.
# ---------------------------------------------------------------------


async def test_occ_two_concurrent_writers_leave_exactly_one_transition(
    session, session_factory
):
    """The assertion with teeth is the audit-row count, not the state.

    A lost update writes TWO task_transitions rows (both writers believe
    they moved the task) while `tasks.state` reflects only whichever
    committed last. The audit trail and the row then disagree
    permanently, and nothing in the system ever notices -- which is
    precisely the silent corruption this milestone exists to stop.
    """
    task = await _task_at_approval(session)
    task_id, before = task.id, task.version

    barrier = asyncio.Barrier(2)

    async def writer(to_state: TaskState):
        async with session_factory() as s:
            row = await s.get(Task, task_id)
            # Both writers now hold the same version. Nothing has been
            # written yet -- this is the exact interleaving that makes
            # the race real rather than accidental.
            await barrier.wait()
            return await transition(s, row, to_state, actor="human:test")

    outcomes = await asyncio.gather(
        writer(TaskState.MERGED),
        writer(TaskState.CANCELLED),
        return_exceptions=True,
    )

    failures = [o for o in outcomes if isinstance(o, BaseException)]
    successes = [o for o in outcomes if not isinstance(o, BaseException)]

    assert len(successes) == 1, (
        f"expected exactly one writer to win, got {len(successes)} "
        f"-- both writes landed, this is a lost update"
    )
    assert len(failures) == 1, "expected the losing writer to detect the conflict"
    # The TYPE matters, not merely that something was raised. An earlier
    # version of this assertion only counted exceptions, and a real bug
    # slipped through it: reading task.id after session.rollback() (which
    # expires every attribute) triggered a lazy reload -- sync IO on an
    # async session -- so the loser raised MissingGreenlet instead of
    # ConcurrentUpdateError. The count-only assertion was perfectly happy.
    # Caught by running the live demo, not by this suite.
    assert isinstance(failures[0], ConcurrentUpdateError), (
        f"loser raised {type(failures[0]).__name__}: {failures[0]} -- expected "
        f"a clean ConcurrentUpdateError"
    )
    assert failures[0].task_id == task_id

    await session.commit()  # drop this session's snapshot, re-read fresh
    rows = await _count_transitions_from(
        session, task_id, TaskState.WAITING_FOR_APPROVAL
    )
    assert rows == 1, (
        f"{rows} transitions recorded out of WAITING_FOR_APPROVAL -- the audit "
        f"trail claims the task left that state more than once"
    )

    fresh = await session.get(Task, task_id, populate_existing=True)
    assert fresh.state in {TaskState.MERGED.value, TaskState.CANCELLED.value}
    assert fresh.version == before + 1, (
        f"version went {before} -> {fresh.version}; exactly one write landed, "
        f"so exactly one increment should have"
    )


async def test_a_stale_object_cannot_bypass_the_state_machine(session, session_factory):
    """The sharpest consequence of the missing guard, and the reason this
    is a safety bug rather than a tidiness one.

    `transition()` validates against `TaskState(task.state)` -- the value
    on the in-memory object. When another writer has already moved the row,
    that value is stale, so validate_transition() passes judgement on a
    state the task is no longer in. The result is an ILLEGAL transition
    landing in the database: a task a human explicitly CANCELLED comes
    back as MERGED. Milestone 1's state machine is the thing every later
    milestone leans on for "this cannot happen", and a stale object walks
    straight around it.

    No barrier and no race here -- the writes are strictly ordered. That
    is deliberate: this failure needs no unlucky timing, only a second
    writer somewhere in the system, which Milestone 15 shipped.
    """
    task = await _task_at_approval(session)
    task_id = task.id

    # Writer A loads the task and holds it (a chain mid-run does exactly
    # this for minutes at a time).
    stale_session = session_factory()
    stale_task = await stale_session.get(Task, task_id)

    # Writer B rejects it. Committed, done, final.
    async with session_factory() as s:
        row = await s.get(Task, task_id)
        await transition(s, row, TaskState.CANCELLED, actor="human:telegram")

    # Writer A now acts on what it still believes the state to be.
    outcome = None
    try:
        await transition(
            stale_session, stale_task, TaskState.MERGED, actor="human:web"
        )
    except Exception as exc:  # noqa: BLE001 -- any refusal is a pass here
        outcome = exc
    finally:
        await stale_session.close()

    await session.commit()
    fresh = await session.get(Task, task_id, populate_existing=True)

    assert outcome is not None, (
        "a stale writer was allowed to transition a CANCELLED task -- "
        "validate_transition() judged the move against a state the task "
        "had already left"
    )
    assert fresh.state == TaskState.CANCELLED.value, (
        f"human-issued CANCELLED was overwritten to {fresh.state} by a "
        f"writer holding a stale snapshot"
    )


# ---------------------------------------------------------------------
# B / C. Advisory locking — does it serialize the right things and,
# just as importantly, NOT serialize the wrong ones.
#
# Test C is not a nice-to-have. A lock that simply always serializes
# would satisfy B perfectly while destroying the parallelism Section 4.3
# exists to provide, and nothing in B would notice. C is what makes B's
# pass meaningful.
# ---------------------------------------------------------------------

import os  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

from amop.orchestrator import concurrency  # noqa: E402
from amop.orchestrator.concurrency import (  # noqa: E402
    ConcurrencyLimitExceeded,
    repo_file_lock,
    repo_lock_key,
    task_slot,
    try_repo_file_lock,
)

_HOLD = 0.4  # long enough that accidental overlap is unmistakable


async def _critical_section(engine, repo, files, name, log):
    async with repo_file_lock(engine, repo, files):
        log.append((name, "enter", time.monotonic()))
        await asyncio.sleep(_HOLD)
        log.append((name, "exit", time.monotonic()))


def _spans(log):
    return {
        name: (
            next(t for n, k, t in log if n == name and k == "enter"),
            next(t for n, k, t in log if n == name and k == "exit"),
        )
        for name in {n for n, _, _ in log}
    }


async def test_overlapping_files_on_the_same_repo_serialize(engine, tmp_path):
    repo = str(tmp_path)
    log = []
    await asyncio.gather(
        _critical_section(engine, repo, ["a.py", "shared.py"], "A", log),
        _critical_section(engine, repo, ["shared.py", "b.py"], "B", log),
    )
    (a_in, a_out), (b_in, b_out) = _spans(log)["A"], _spans(log)["B"]

    serialized = a_out <= b_in or b_out <= a_in
    assert serialized, (
        f"two tasks both declaring shared.py ran concurrently: "
        f"A[{a_in:.3f}..{a_out:.3f}] B[{b_in:.3f}..{b_out:.3f}] -- both were "
        f"free to edit the same file at the same time"
    )


async def test_disjoint_files_on_the_same_repo_run_in_parallel(engine, tmp_path):
    """The precision test: same repo, nothing in common, must NOT wait."""
    repo = str(tmp_path)
    log = []
    started = time.monotonic()
    await asyncio.gather(
        _critical_section(engine, repo, ["a.py"], "A", log),
        _critical_section(engine, repo, ["b.py"], "B", log),
    )
    elapsed = time.monotonic() - started
    (a_in, a_out), (b_in, b_out) = _spans(log)["A"], _spans(log)["B"]

    assert a_in < b_out and b_in < a_out, (
        f"disjoint file sets were serialized: A[{a_in:.3f}..{a_out:.3f}] "
        f"B[{b_in:.3f}..{b_out:.3f}] -- the lock is acting as a blunt "
        f"repo-wide mutex"
    )
    assert elapsed < _HOLD * 1.8, (
        f"took {elapsed:.2f}s for two {_HOLD}s sections that should overlap"
    )


async def test_different_repos_run_in_parallel(engine, tmp_path):
    """Section 4.3's own example: unrelated repos are fully parallel."""
    repo_a = str(tmp_path / "one")
    repo_b = str(tmp_path / "two")
    os.makedirs(repo_a, exist_ok=True)
    os.makedirs(repo_b, exist_ok=True)
    log = []
    await asyncio.gather(
        _critical_section(engine, repo_a, ["same_name.py"], "A", log),
        _critical_section(engine, repo_b, ["same_name.py"], "B", log),
    )
    (a_in, a_out), (b_in, b_out) = _spans(log)["A"], _spans(log)["B"]
    assert a_in < b_out and b_in < a_out, (
        "identical filenames in DIFFERENT repos were serialized -- the key "
        "is not repo-scoped"
    )


# ---------------------------------------------------------------------
# D. Crash mid-lock. A real OS process, killed with SIGKILL.
#
# SIGKILL rather than SIGTERM on purpose: SIGTERM runs cleanup handlers,
# so a polite shutdown would release the lock through ordinary code and
# prove nothing about crashes. SIGKILL gives the process no chance to
# tidy up, which is the actual scenario -- and the exact failure mode
# ADR-12 says a Redis lease has to work hard to survive.
# ---------------------------------------------------------------------

_CHILD = """
import asyncio, sys, asyncpg
key = int(sys.argv[1])
async def main():
    conn = await asyncpg.connect("postgresql://localhost/amop_test")
    tx = conn.transaction(); await tx.start()
    await conn.execute("SELECT pg_advisory_xact_lock($1)", key)
    print("LOCKED", flush=True)
    await asyncio.sleep(300)
asyncio.run(main())
"""


async def test_a_killed_process_does_not_leave_the_repo_locked(engine, tmp_path):
    repo = str(tmp_path)
    key = repo_lock_key(repo)

    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(key)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "LOCKED", "child never took the lock"

        held = await try_repo_file_lock(engine, repo)
        assert held is False, "child claims the lock but we acquired it anyway"

        child.kill()  # SIGKILL -- no cleanup, no unlock, a real crash
        child.wait(timeout=10)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if await try_repo_file_lock(engine, repo):
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError(
                "the repo stayed locked after the holding process was killed -- "
                "a crash has permanently wedged this repo"
            )
    finally:
        if child.poll() is None:
            child.kill()


# ---------------------------------------------------------------------
# E. Slot limits.
# ---------------------------------------------------------------------


async def test_global_limit_refuses_the_extra_task(monkeypatch, tmp_path):
    monkeypatch.setattr(concurrency, "MAX_CONCURRENT_TASKS_GLOBAL", 2)
    monkeypatch.setattr(concurrency, "MAX_CONCURRENT_TASKS_PER_REPO", 5)
    repo = str(tmp_path)
    release = asyncio.Event()

    async def holder():
        async with task_slot(repo):
            await release.wait()

    held = [asyncio.create_task(holder()) for _ in range(2)]
    await asyncio.sleep(0.05)

    with pytest.raises(ConcurrencyLimitExceeded) as excinfo:
        async with task_slot(repo):
            pass
    assert "global" in str(excinfo.value)

    release.set()
    await asyncio.gather(*held)

    # Capacity must come back once they finish, not stay consumed.
    async with task_slot(repo):
        pass


async def test_per_repo_limit_is_independent_across_repos(monkeypatch, tmp_path):
    monkeypatch.setattr(concurrency, "MAX_CONCURRENT_TASKS_GLOBAL", 10)
    monkeypatch.setattr(concurrency, "MAX_CONCURRENT_TASKS_PER_REPO", 1)
    repo_a, repo_b = str(tmp_path / "a"), str(tmp_path / "b")
    release = asyncio.Event()

    async def holder(repo):
        async with task_slot(repo):
            await release.wait()

    first = asyncio.create_task(holder(repo_a))
    await asyncio.sleep(0.05)

    with pytest.raises(ConcurrencyLimitExceeded):
        async with task_slot(repo_a):
            pass

    # A different repo must be unaffected -- otherwise the per-repo limit
    # is really a global one wearing a different name.
    async with task_slot(repo_b):
        pass

    release.set()
    await first


async def test_a_failing_task_gives_its_slot_back(monkeypatch, tmp_path):
    """A leaked slot is a slow-motion outage: after N failures the
    orchestrator silently stops accepting work and looks merely idle."""
    monkeypatch.setattr(concurrency, "MAX_CONCURRENT_TASKS_GLOBAL", 1)
    monkeypatch.setattr(concurrency, "MAX_CONCURRENT_TASKS_PER_REPO", 1)
    repo = str(tmp_path)

    with pytest.raises(RuntimeError):
        async with task_slot(repo):
            raise RuntimeError("the task blew up")

    async with task_slot(repo):
        pass  # must be free again


async def test_every_run_entry_point_is_slot_gated():
    """A guard against the most likely future regression: someone adds a
    fourth `run_*` entry point and it quietly bypasses the limits.

    Checked structurally rather than by running the chains, which would
    need Docker and a live model for what is really a wiring question.
    """
    from amop.orchestrator.chain import run_fix
    from amop.orchestrator.deps import run_dependency_update
    from amop.orchestrator.optimize import run_optimization

    for fn in (run_fix, run_optimization, run_dependency_update):
        assert getattr(fn, "__wrapped__", None) is not None, (
            f"{fn.__name__} is not wrapped by gated_by_task_slot -- it can "
            f"start work without claiming a concurrency slot"
        )


async def test_a_saturated_limit_refuses_run_fix_with_a_real_error(
    monkeypatch, tmp_path, session
):
    """End-to-end shape of a refusal: the caller gets an exception it can
    report, not a silent no-op or a task stuck at CREATED forever."""
    from amop.orchestrator.chain import run_fix

    monkeypatch.setattr(concurrency, "MAX_CONCURRENT_TASKS_GLOBAL", 1)
    repo = str(tmp_path)
    release = asyncio.Event()

    async def occupy():
        async with task_slot(repo):
            await release.wait()

    holder = asyncio.create_task(occupy())
    await asyncio.sleep(0.05)

    task = await create_task(session, task_type="bug_fix", task_context={})
    with pytest.raises(ConcurrencyLimitExceeded) as excinfo:
        # Never reaches Docker/Ollama: the gate refuses before any of the
        # expensive setup, which is the point of gating at the entry point.
        await run_fix(
            session, task, description="x", repo_path=tmp_path, model=object()
        )
    assert "refusing to start" in str(excinfo.value)

    release.set()
    await holder
