"""Milestone 20 — Multi-Repo Support.

Same posture as Milestone 16: don't trust the machinery until you've
tried to break it. Every timing/isolation test here was run against
deliberately-mutated code first and observed failing -- per CLAUDE.md's
standing rule, a concurrency test never seen failing is not evidence.

What's genuinely being proven, as opposed to merely exercised:

  * Two repos run in PARALLEL -- asserted as real overlap in time
    (`enter2 < exit1`), not "both eventually finished", which a fully
    serialized implementation would also satisfy.
  * The same repo with overlapping files still SERIALIZES -- the
    regression check that multi-repo didn't quietly undo Milestone 16.
    Without this, a lock keyed on nothing at all would pass the
    parallelism test perfectly.
  * An `observer` repo genuinely cannot mutate while an `autonomous`
    repo concurrently can -- through the real Safety Engine and a real
    Docker sandbox, asserting the SPECIFIC denial reason and the absence
    of the file on disk, not just a falsy result.

Requires: Postgres at TEST_DATABASE_URL, a real Docker daemon.
"""

import asyncio
import time

import pytest
import pytest_asyncio
from sqlalchemy import text

from amop.database.models import Repository
from amop.database.session import init_db, make_engine, make_session_factory
from amop.orchestrator.concurrency import repo_file_lock, task_slot
from amop.safety.permissions import (
    load_permission_overrides,
    normalize_repo_identity,
    resolve_mode_from,
)
from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext, invoke_tool

TEST_DATABASE_URL = "postgresql+asyncpg://localhost/amop_test"

# Long enough that accidental overlap (or accidental serialization) is
# unmistakable rather than a timing coin-flip.
_HOLD = 0.4


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
        await conn.execute(text("TRUNCATE repositories RESTART IDENTITY CASCADE"))


@pytest_asyncio.fixture
async def session(session_factory):
    async with session_factory() as s:
        yield s


@pytest.fixture
def sandbox_manager():
    manager = SandboxManager()
    yield manager
    manager.destroy_all()


async def _register(session, path, **kwargs):
    row = Repository(repo_path=normalize_repo_identity(str(path)), **kwargs)
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


# ---------------------------------------------------------------------
# D-8 precedence — pure, no I/O. Section 12.1's three levels.
# ---------------------------------------------------------------------


def test_global_mode_applies_when_a_repo_has_no_overrides():
    assert resolve_mode_from("coder", "suggestor", None) == "suggestor"
    assert resolve_mode_from("coder", "suggestor", {}) == "suggestor"


def test_per_repo_override_beats_the_global_default():
    assert resolve_mode_from("coder", "autonomous", {"default": "observer"}) == "observer"


def test_per_agent_override_beats_the_per_repo_default():
    """Section 12.1's own worked example, verbatim: 'DependencyUpdater is
    autonomous, everything else on this repo is suggestor'."""
    overrides = {"default": "suggestor", "agents": {"dependency_updater": "autonomous"}}
    assert resolve_mode_from("coder", "observer", overrides) == "suggestor"
    assert resolve_mode_from("dependency_updater", "observer", overrides) == "autonomous"


def test_an_unrecognized_mode_falls_back_instead_of_being_honored():
    """A typo in a hand-edited JSONB column must not silently become a
    permission level, and must not raise inside the safety gate either."""
    assert resolve_mode_from("coder", "observer", {"default": "autonomus"}) == "observer"
    assert (
        resolve_mode_from(
            "coder", "observer", {"default": "suggestor", "agents": {"coder": "root"}}
        )
        == "suggestor"
    )


# ---------------------------------------------------------------------
# Repo identity: the write-side normalization bug found during build.
# ---------------------------------------------------------------------


async def test_overrides_resolve_regardless_of_how_the_path_is_spelled(session, tmp_path):
    """Regression pin for a real bug found live, not hypothesized.

    Lookup normalized but registration didn't, so on macOS (where /tmp is
    a symlink to /private/tmp) a row stored raw never matched -- and a
    per-repo override that never matches fails SILENTLY toward the global
    mode. Nothing raised; permissions just quietly didn't apply.
    """
    await _register(session, tmp_path, permission_overrides={"default": "observer"})

    for spelling in (str(tmp_path), str(tmp_path) + "/", f"{tmp_path}/./"):
        got = await load_permission_overrides(session, spelling)
        assert got == {"default": "observer"}, f"{spelling!r} failed to resolve"


async def test_an_unregistered_repo_resolves_to_no_overrides(session, tmp_path):
    """Not being registered is not an error -- `amop fix` against an
    unregistered directory must still work, at the global mode."""
    assert await load_permission_overrides(session, str(tmp_path)) is None


# ---------------------------------------------------------------------
# Parallel across repos / serialize within one — the timing proof.
# ---------------------------------------------------------------------


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


async def test_two_repos_run_genuinely_in_parallel(engine, session, tmp_path):
    """Done-When #3. Two registered repos, identical filenames inside
    them -- must NOT serialize each other."""
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    await _register(session, repo_a)
    await _register(session, repo_b)

    log = []
    await asyncio.gather(
        _critical_section(engine, str(repo_a), ["app.py"], "A", log),
        _critical_section(engine, str(repo_b), ["app.py"], "B", log),
    )
    (a_in, a_out), (b_in, b_out) = _spans(log)["A"], _spans(log)["B"]

    # Overlap is asserted by comparing the two sections' OWN timestamps
    # to each other, never against a wall-clock budget. That distinction
    # is deliberate: an earlier version of this test also asserted
    # `elapsed < _HOLD * 1.8`, which flaked once under load right after a
    # cache clear. A wall-clock bound measures how busy the machine is as
    # much as whether the lock serialized -- so it can fail when the code
    # is correct, and (worse) pass on a fast machine even if overlap were
    # marginal. The relative check below is load-independent and is the
    # actual property; the wall-clock one only ever duplicated it, less
    # reliably, so it's gone rather than merely loosened.
    assert a_in < b_out and b_in < a_out, (
        f"two different repos serialized each other: "
        f"A[{a_in:.3f}..{a_out:.3f}] B[{b_in:.3f}..{b_out:.3f}]"
    )


async def test_same_repo_overlapping_files_still_serializes(engine, session, tmp_path):
    """Milestone 16 regression check -- the assertion that makes the
    parallelism test above meaningful. A lock keyed on nothing would pass
    that one and fail this one."""
    repo = tmp_path / "repo_a"
    repo.mkdir()
    await _register(session, repo)

    log = []
    await asyncio.gather(
        _critical_section(engine, str(repo), ["shared.py", "a.py"], "A", log),
        _critical_section(engine, str(repo), ["shared.py", "b.py"], "B", log),
    )
    (a_in, a_out), (b_in, b_out) = _spans(log)["A"], _spans(log)["B"]

    assert a_out <= b_in or b_out <= a_in, (
        f"two tasks both declaring shared.py in the SAME repo ran "
        f"concurrently: A[{a_in:.3f}..{a_out:.3f}] B[{b_in:.3f}..{b_out:.3f}]"
    )


async def test_per_repo_slot_limits_do_not_block_a_different_repo(
    monkeypatch, session, tmp_path
):
    """The other half of parallelism: slots are per-repo, so saturating
    one repo's limit must not starve another's."""
    from amop.orchestrator import concurrency

    monkeypatch.setattr(concurrency, "MAX_CONCURRENT_TASKS_GLOBAL", 10)
    monkeypatch.setattr(concurrency, "MAX_CONCURRENT_TASKS_PER_REPO", 1)
    repo_a, repo_b = tmp_path / "a", tmp_path / "b"
    repo_a.mkdir()
    repo_b.mkdir()
    release = asyncio.Event()

    async def holder(repo):
        async with task_slot(str(repo)):
            await release.wait()

    first = asyncio.create_task(holder(repo_a))
    await asyncio.sleep(0.05)

    with pytest.raises(concurrency.ConcurrencyLimitExceeded):
        async with task_slot(str(repo_a)):
            pass

    async with task_slot(str(repo_b)):  # different repo -- unaffected
        pass

    release.set()
    await first


# ---------------------------------------------------------------------
# Per-repo permission isolation, through the REAL Safety Engine and a
# REAL Docker sandbox, with both repos active at the same time.
# ---------------------------------------------------------------------


async def test_observer_repo_cannot_mutate_while_autonomous_repo_can(
    session, sandbox_manager, tmp_path
):
    """Done-When #2, and the point of per-repo overrides existing at all.

    Both tasks run concurrently. The `observer` repo's write must be
    denied for the RIGHT reason and leave nothing on disk; the
    `autonomous` repo's identical write must succeed. Asserting the
    specific reason matters: a test that only checked "denied" would pass
    even if the denial came from something unrelated to the override.
    """
    repo_obs = tmp_path / "locked_down"
    repo_auto = tmp_path / "wide_open"
    repo_obs.mkdir()
    repo_auto.mkdir()
    await _register(session, repo_obs, permission_overrides={"default": "observer"})
    await _register(session, repo_auto, permission_overrides={"default": "autonomous"})

    sb_obs = sandbox_manager.create("m20-obs", repo_obs)
    sb_auto = sandbox_manager.create("m20-auto", repo_auto)
    try:

        async def attempt_write(repo_dir, sandbox, agent):
            overrides = await load_permission_overrides(session, str(repo_dir))
            ctx = ToolContext(
                agent_name=agent,
                scratch_dir=repo_dir,
                # Global mode is the SAME for both -- so any difference in
                # outcome comes from the per-repo override, nothing else.
                mode="suggestor",
                sandbox=sandbox,
                repo_path=str(repo_dir),
                permission_overrides=overrides,
            )
            return await invoke_tool(
                "write_file",
                {"path": "touched.py", "content": "x = 1\n"},
                ctx,
                agent_name=agent,
            )

        denied, allowed = await asyncio.gather(
            attempt_write(repo_obs, sb_obs, "coder"),
            attempt_write(repo_auto, sb_auto, "coder"),
        )

        assert not denied.success
        assert denied.error_code == "DENIED"
        assert denied.message == "mode_forbids_mutation"
        assert not (repo_obs / "touched.py").exists(), "observer repo was written to"

        assert allowed.success, allowed.message
        assert (repo_auto / "touched.py").read_text() == "x = 1\n"
    finally:
        sandbox_manager.destroy("m20-obs")
        sandbox_manager.destroy("m20-auto")


async def test_per_agent_override_lets_one_agent_through_on_a_locked_repo(
    session, sandbox_manager, tmp_path
):
    """Section 12.1's exact example, end to end: the whole repo is
    observer, except DependencyUpdater."""
    repo = tmp_path / "mostly_locked"
    repo.mkdir()
    await _register(
        session,
        repo,
        permission_overrides={
            "default": "observer",
            "agents": {"dependency_updater": "autonomous"},
        },
    )
    sandbox = sandbox_manager.create("m20-peragent", repo)
    try:
        overrides = await load_permission_overrides(session, str(repo))

        async def attempt(agent, filename):
            ctx = ToolContext(
                agent_name=agent,
                scratch_dir=repo,
                mode="suggestor",
                sandbox=sandbox,
                repo_path=str(repo),
                permission_overrides=overrides,
            )
            return await invoke_tool(
                "write_file", {"path": filename, "content": "y = 2\n"}, ctx, agent_name=agent
            )

        blocked = await attempt("coder", "by_coder.py")
        assert not blocked.success
        assert blocked.message == "mode_forbids_mutation"
        assert not (repo / "by_coder.py").exists()

        permitted = await attempt("dependency_updater", "by_deps.py")
        assert permitted.success, permitted.message
        assert (repo / "by_deps.py").read_text() == "y = 2\n"
    finally:
        sandbox_manager.destroy("m20-peragent")


async def test_an_override_cannot_unlock_a_protected_path(
    session, sandbox_manager, tmp_path
):
    """Milestone 19 must still hold: `autonomous` via a per-repo override
    is still not permission to rewrite CI config. The protected-path
    check is not mode-derived, so no override can reach it."""
    repo = tmp_path / "auto_repo"
    repo.mkdir()
    await _register(session, repo, permission_overrides={"default": "autonomous"})
    sandbox = sandbox_manager.create("m20-protected", repo)
    try:
        overrides = await load_permission_overrides(session, str(repo))
        ctx = ToolContext(
            agent_name="coder",
            scratch_dir=repo,
            mode="observer",  # even the global mode is the most restrictive
            sandbox=sandbox,
            repo_path=str(repo),
            permission_overrides=overrides,
        )
        result = await invoke_tool(
            "write_file",
            {"path": ".github/workflows/ci.yml", "content": "on: push\n"},
            ctx,
            agent_name="coder",
        )
        assert not result.success
        assert result.message == "protected_infrastructure_path"
        assert not (repo / ".github").exists()
    finally:
        sandbox_manager.destroy("m20-protected")


# ---------------------------------------------------------------------
# Watcher across repos.
# ---------------------------------------------------------------------


async def test_watch_targets_only_returns_ready_repos_with_a_url(session, tmp_path):
    from amop.cli.main import _watch_targets

    ready = tmp_path / "ready"
    unindexed = tmp_path / "unindexed"
    no_url = tmp_path / "no_url"
    for d in (ready, unindexed, no_url):
        d.mkdir()
    await _register(session, ready, url="owner/ready", index_status="ready")
    await _register(session, unindexed, url="owner/unindexed", index_status="unindexed")
    # Registered and indexed, but no slug -- can't be watched, and must be
    # skipped rather than have a slug guessed from its local path.
    await _register(session, no_url, index_status="ready")

    factory = _SessionFactoryShim(session)
    targets = await _watch_targets(factory)

    assert [slug for slug, _ in targets] == ["owner/ready"]


class _SessionFactoryShim:
    """_watch_targets opens its own session; hand it the test's one so
    everything stays in a single transaction the fixture can roll back."""

    def __init__(self, session):
        self._session = session

    def __call__(self):
        return self

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False


async def test_anomaly_rate_breaker_is_independent_per_repo(session):
    """A noisy repo must not exhaust another repo's budget. Verified
    against the real breaker, which counts watcher-created tasks per
    repo."""
    from amop.orchestrator.task import create_task
    from amop.safety import circuit_breakers

    noisy, quiet = "owner/noisy", "owner/quiet"
    cap = 3
    for _ in range(cap):
        await create_task(
            session,
            task_type="bug_fix",
            task_context={"source": "github_watcher", "repo": noisy, "prompt": "x"},
        )

    noisy_check = await circuit_breakers.check_anomaly_rate(session, noisy, cap=cap)
    quiet_check = await circuit_breakers.check_anomaly_rate(session, quiet, cap=cap)

    assert not noisy_check.allow
    assert quiet_check.allow, "a noisy repo consumed a different repo's budget"
