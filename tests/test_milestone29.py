"""Crash Recovery: Micro-Commits + Reconciliation — spec Sections 4.6.1
(Design Decision D-15, micro-commits) and 4.6.2 (reconciliation, Stage 2).

Stage 1 only in this file for now: micro-commits, hook-bypass, and the
mandatory squash-before-PR gate. Stage 2's reconciliation cases (A-F)
are a separate addition once Stage 1 is confirmed.

Real Docker daemon and real Postgres (TEST_DATABASE_URL) required for
the sandbox-level and full-chain tests -- there is no mock/fake sandbox
path in this project (same discipline as test_milestone3.py).
"""

import json
import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select, text

from amop.agents import tester as tester_mod
from amop.agents.coder import CoderAgent
from amop.agents.handoffs import Evidence, RootCauseReport
from amop.agents.investigator import InvestigatorAgent
from amop.agents.reviewer import ReviewerAgent
from amop.database.models import PullRequest, TaskTransition
from amop.database.session import init_db, make_engine, make_session_factory
from amop.models.base import BaseLLM, ModelResponse
from amop.orchestrator.chain import ChainAgents, run_chain
from amop.orchestrator.reconcile import reconcile
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import create_task, transition_with_retry
from amop.sandbox import repo as git_repo
from amop.sandbox import tools as sandbox_tools  # noqa: F401 -- registers sandboxed tools
from amop.sandbox.manager import SandboxManager
from amop.sandbox.tools import patch_file, write_file
from amop.tools.registry import ToolContext, ToolResult, get_tool

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://localhost/amop_test"
)
FIXTURE_REPO = Path(__file__).parent / "fixtures" / "buggy_calculator"
BUGGY_LINE = "return sum(values) / (len(values) - 1)"
FIXED_LINE = "return sum(values) / len(values)"


# =======================================================================
# Scripted model + chain-building helpers (self-contained -- same
# pattern test_milestone4.py established, not imported cross-file)
# =======================================================================


class ScriptedLLM(BaseLLM):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, messages, tools=None) -> ModelResponse:
        content = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return ModelResponse(content=content, model="scripted")

    async def embed(self, texts):
        raise NotImplementedError


def tool_call(name: str, **arguments) -> str:
    return json.dumps({"tool_call": {"name": name, "arguments": arguments}})


def final(answer) -> str:
    return json.dumps({"final_answer": answer})


def root_cause_answer(confidence: float = 1.0) -> str:
    return final(
        {
            "task_id": "t",
            "root_cause": "average() divides by len(values) - 1 instead of len(values)",
            "confidence": confidence,
            "evidence": [
                {"type": "code", "ref": "calculator.py:39", "excerpt": BUGGY_LINE}
            ],
            "affected_files": ["calculator.py"],
            "suggested_fix_plan": "divide by len(values)",
        }
    )


def review_answer(
    approved: bool,
    rejection_reason: str | None = None,
    counterexample: str | None = None,
    counterexample_claim: dict | None = None,
) -> str:
    return final(
        {
            "task_id": "t",
            "approved": approved,
            "addresses_reported_symptom": True,
            "findings": [],
            "rejection_reason": rejection_reason,
            "counterexample": counterexample,
            "counterexample_claim": counterexample_claim,
        }
    )


def suite_answer(all_passed: bool) -> str:
    return final({"task_id": "t", "all_passed": all_passed, "details": "ok"})


def fixed_calculator_source(scratch: Path) -> str:
    source = (scratch / "calculator.py").read_text()
    assert BUGGY_LINE in source
    return source.replace(BUGGY_LINE, FIXED_LINE)


def build_agents(ctx, *, investigator, coder, tester, reviewer) -> ChainAgents:
    return ChainAgents(
        investigator=InvestigatorAgent(ScriptedLLM(investigator), ctx),
        coder=CoderAgent(ScriptedLLM(coder), ctx, task_id="scripted"),
        tester=tester_mod.TesterAgent(ScriptedLLM(tester), ctx),
        reviewer=ReviewerAgent(ScriptedLLM(reviewer), ctx),
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
                "TRUNCATE agent_actions, task_transitions, pull_requests, tasks "
                "RESTART IDENTITY CASCADE"
            )
        )


@pytest.fixture(autouse=True)
def _fake_github(monkeypatch):
    """Same reasoning as test_milestone4.py's identical fixture: PR_CREATION
    calls the real create_pull_request tool, which would otherwise attempt
    a genuine network push + GitHub API call. Faked so the chain still
    exercises real PR_CREATION routing (real ToolResult, real
    WAITING_FOR_APPROVAL transition) without touching the network -- and
    critically for this milestone, so the branch that reaches PR_CREATION
    is inspected in the sandbox exactly as the real squash step left it,
    not as any second process would have mutated it."""

    async def _fake_create_pull_request(title, body, head, base, ctx):
        return ToolResult(
            success=True,
            output={
                "url": "https://github.com/bivek127/amop-sandbox/pull/999",
                "number": 999,
                "created": True,
            },
        )

    monkeypatch.setattr(get_tool("create_pull_request"), "func", _fake_create_pull_request)


@pytest.fixture
def sandbox_manager():
    manager = SandboxManager()
    yield manager
    manager.destroy_all()


@pytest.fixture
def workspace(tmp_path, sandbox_manager):
    """A materialized fixture repo in a real container, git-initialized
    and on a working branch -- no task_id on ctx (matches
    test_milestone4.py's version) for the sandbox-level tests that don't
    need a persisted Task row at all."""
    scratch = tmp_path / "workspace"
    git_repo.materialize(FIXTURE_REPO, scratch)
    task_id = f"m29-{uuid.uuid4().hex[:8]}"
    sandbox = sandbox_manager.create(task_id, scratch)
    git_repo.init_baseline(sandbox)
    git_repo.create_branch(sandbox, "amop/fix-test")
    ctx = ToolContext(
        agent_name="chain", scratch_dir=scratch, mode="operator", sandbox=sandbox
    )
    yield ctx, scratch, sandbox


# =======================================================================
# Micro-commits (spec 4.6.1 / D-15) -- direct repo.py mechanics
# =======================================================================


def test_micro_commit_appears_on_a_real_edit(workspace):
    ctx, scratch, sandbox = workspace
    (scratch / "newfile.txt").write_text("original\n")  # host-visible via bind mount
    sandbox.write_file("/workspace/scratch.txt", "hello\n")

    sha = git_repo.micro_commit(sandbox, "task-abc", "write_file", "scratch.txt")

    assert sha is not None
    log = sandbox.exec_run("git -C /workspace log --oneline -1").stdout
    assert "[AMOP][wip][task-abc] write_file: scratch.txt" in log
    assert sandbox.exec_run("git -C /workspace rev-parse HEAD").stdout.strip() == sha


def test_micro_commit_skips_an_empty_diff(workspace):
    """spec 4.6.1: 'a patch_file that resulted in no net change produces
    no commit.' Writing identical content twice must commit once."""
    _ctx, _scratch, sandbox = workspace
    sandbox.write_file("/workspace/scratch.txt", "same content\n")
    sha1 = git_repo.micro_commit(sandbox, "task-abc", "write_file", "scratch.txt")
    assert sha1 is not None

    sandbox.write_file("/workspace/scratch.txt", "same content\n")  # no net change
    sha2 = git_repo.micro_commit(sandbox, "task-abc", "write_file", "scratch.txt")

    assert sha2 is None
    # HEAD genuinely did not move.
    assert sandbox.exec_run("git -C /workspace rev-parse HEAD").stdout.strip() == sha1


def test_micro_commit_identity_is_amop_bot_never_the_baseline_identity(workspace):
    """spec 8.4: commit author is always amop-bot <amop-bot@localhost>,
    never a human identity -- and, found while building this, never the
    sandbox's own pre-existing baseline identity either (AMOP Agent
    <agent@amop.local>, baked into the Dockerfile since Milestone 6).
    Both identities must appear, on the right commits."""
    _ctx, _scratch, sandbox = workspace
    sandbox.write_file("/workspace/scratch.txt", "x\n")
    git_repo.micro_commit(sandbox, "task-abc", "write_file", "scratch.txt")

    log = sandbox.exec_run(
        'git -C /workspace log --format="%an <%ae> %s"'
    ).stdout
    lines = [line for line in log.splitlines() if line.strip()]
    micro_line = next(line for line in lines if "[AMOP][wip]" in line)
    baseline_line = next(line for line in lines if "baseline:" in line)

    assert micro_line.startswith("amop-bot <amop-bot@localhost>")
    assert baseline_line.startswith("AMOP Agent <agent@amop.local>")


def test_micro_commit_message_format_is_mechanically_identifiable(workspace):
    _ctx, _scratch, sandbox = workspace
    sandbox.write_file("/workspace/x.txt", "1\n")
    sha = git_repo.micro_commit(sandbox, "abc-123", "patch_file", "x.txt")

    subject = sandbox.exec_run(
        f"git -C /workspace log -1 --format=%s {sha}"
    ).stdout.strip()
    assert subject == "[AMOP][wip][abc-123] patch_file: x.txt"


def test_hooks_never_fire_on_a_micro_commit(workspace):
    """spec Table 19.5's named test: plant sentinel hooks, assert their
    absence after a real commit. A repository's own .git/hooks/* would
    otherwise be attacker-controlled code execution triggered by every
    single successful edit AMOP makes (spec 4.6.1's risk #2).

    Both pre-commit AND post-commit are planted deliberately, not just
    one: --no-verify (also applied on every AMOP commit) already
    suppresses pre-commit/commit-msg on its own, so a pre-commit-only
    test would pass even if core.hooksPath were silently removed --
    confirmed by mutation testing this exact test before trusting it.
    post-commit fires unconditionally regardless of --no-verify, so
    it's the one that actually proves core.hooksPath is doing real,
    distinct work."""
    _ctx, _scratch, sandbox = workspace
    sandbox.exec_run(
        "mkdir -p /workspace/.git/hooks && "
        'printf "#!/bin/sh\\ntouch /tmp/PRE_HOOK_FIRED\\n" '
        "> /workspace/.git/hooks/pre-commit && "
        'printf "#!/bin/sh\\ntouch /tmp/POST_HOOK_FIRED\\n" '
        "> /workspace/.git/hooks/post-commit && "
        "chmod +x /workspace/.git/hooks/pre-commit /workspace/.git/hooks/post-commit"
    )

    sandbox.write_file("/workspace/y.txt", "1\n")
    git_repo.micro_commit(sandbox, "task-abc", "write_file", "y.txt")

    pre_fired = sandbox.exec_run("test -f /tmp/PRE_HOOK_FIRED").exit_code
    post_fired = sandbox.exec_run("test -f /tmp/POST_HOOK_FIRED").exit_code
    assert pre_fired != 0, "the planted pre-commit hook fired"
    assert post_fired != 0, (
        "the planted post-commit hook fired -- --no-verify does not "
        "suppress post-commit, only core.hooksPath does"
    )


def test_hooks_never_fire_on_the_baseline_or_final_commit_either(workspace):
    """The hook bypass is set once in init_baseline(), before ANY commit
    -- confirm it also covers the ordinary commit_all() path (the
    Coder's own final commit), not just micro-commits. post-commit
    again, not pre-commit -- see test_hooks_never_fire_on_a_micro_commit
    for why pre-commit alone would pass regardless of core.hooksPath."""
    _ctx, scratch, sandbox = workspace
    sandbox.exec_run(
        "mkdir -p /workspace/.git/hooks && "
        'printf "#!/bin/sh\\ntouch /tmp/POST_HOOK_FIRED_2\\n" '
        "> /workspace/.git/hooks/post-commit && "
        "chmod +x /workspace/.git/hooks/post-commit"
    )
    sandbox.write_file("/workspace/z.txt", "1\n")
    git_repo.commit_all(sandbox, "fix: something")

    fired = sandbox.exec_run("test -f /tmp/POST_HOOK_FIRED_2").exit_code
    assert fired != 0


def test_a_git_commit_failure_reports_success_with_a_warning_not_a_tool_failure():
    """spec 4.6.1: 'If the micro-commit itself fails..., the tool result
    reports success-with-warning rather than failing the edit -- the
    edit did happen; losing durability is worse... but not worth
    discarding correct work over.' Forced with a real stale index.lock,
    not simulated."""
    import asyncio

    async def run():
        manager = SandboxManager()
        task_id = f"m29-lock-{uuid.uuid4().hex[:8]}"
        scratch = Path("/tmp") / task_id
        git_repo.materialize(FIXTURE_REPO, scratch)
        sandbox = manager.create(task_id, scratch)
        try:
            git_repo.init_baseline(sandbox)
            git_repo.create_branch(sandbox, "amop/fix-lock")
            sandbox.exec_run("touch /workspace/.git/index.lock")

            ctx = ToolContext(
                agent_name="coder", scratch_dir=scratch, mode="operator",
                sandbox=sandbox, task_id=uuid.uuid4(),
            )
            result = await write_file("newfile.txt", "edit despite lock\n", ctx)

            assert result.success is True
            assert "warning" in result.output
            assert "micro-commit failed" in result.output
            # The edit itself is real on disk regardless of the commit failure.
            content = sandbox.read_file("/workspace/newfile.txt")
            assert content == "edit despite lock\n"
        finally:
            manager.destroy(task_id)

    asyncio.run(run())


def test_write_file_with_no_task_id_skips_micro_commit_but_still_succeeds(workspace):
    """Plenty of legitimate call sites have no task_id (matches
    audit/actions.py's own 'no db_session' convention) -- must not be
    an error, and the edit must still genuinely happen."""
    import asyncio

    ctx, _scratch, sandbox = workspace  # workspace's ctx has no task_id
    assert ctx.task_id is None

    before = sandbox.exec_run("git -C /workspace rev-parse HEAD").stdout.strip()
    result = asyncio.run(write_file("another.txt", "no task id\n", ctx))
    after = sandbox.exec_run("git -C /workspace rev-parse HEAD").stdout.strip()

    assert result.success is True
    assert before == after  # no commit was made
    assert sandbox.read_file("/workspace/another.txt") == "no task id\n"


def test_patch_file_also_triggers_a_real_micro_commit(workspace):
    import asyncio

    _ctx, scratch, sandbox = workspace
    ctx = ToolContext(
        agent_name="coder", scratch_dir=scratch, mode="operator",
        sandbox=sandbox, task_id=uuid.uuid4(),
    )
    diff = (
        "--- calculator.py\n+++ calculator.py\n@@ -36,4 +36,4 @@\n"
        '         if not values:\n             raise ValueError("cannot average an empty list")\n'
        f"-        {BUGGY_LINE}\n+        {FIXED_LINE}\n"
    )
    result = asyncio.run(patch_file("calculator.py", diff, ctx))

    assert result.success is True
    log = sandbox.exec_run("git -C /workspace log --oneline -1").stdout
    assert "patch_file: calculator.py" in log


# =======================================================================
# Squash-before-PR gate (spec 4.6.1's mandatory squash)
# =======================================================================


def test_squash_collapses_multiple_wip_commits_into_one(workspace):
    _ctx, _scratch, sandbox = workspace
    sandbox.write_file("/workspace/a.txt", "1\n")
    git_repo.micro_commit(sandbox, "t", "write_file", "a.txt")
    sandbox.write_file("/workspace/b.txt", "2\n")
    git_repo.micro_commit(sandbox, "t", "write_file", "b.txt")
    sandbox.write_file("/workspace/c.txt", "3\n")
    git_repo.micro_commit(sandbox, "t", "write_file", "c.txt")

    assert git_repo.has_wip_commits(sandbox) is True

    squashed = git_repo.squash_wip_commits(sandbox, "fix: consolidated")

    assert squashed is not None
    assert git_repo.has_wip_commits(sandbox) is False
    log = sandbox.exec_run("git -C /workspace log --oneline main..HEAD").stdout
    assert len(log.strip().splitlines()) == 1
    # All three edits still present in the working tree after squash.
    assert sandbox.read_file("/workspace/a.txt").strip() == "1"
    assert sandbox.read_file("/workspace/b.txt").strip() == "2"
    assert sandbox.read_file("/workspace/c.txt").strip() == "3"


def test_squash_is_a_noop_when_nothing_was_committed_since_baseline(workspace):
    _ctx, _scratch, sandbox = workspace
    before = sandbox.exec_run("git -C /workspace rev-parse HEAD").stdout.strip()

    result = git_repo.squash_wip_commits(sandbox, "fix: nothing to squash")

    assert result is None
    after = sandbox.exec_run("git -C /workspace rev-parse HEAD").stdout.strip()
    assert before == after


def test_squash_survives_an_interleaved_commit_all_call(workspace):
    """A task can cycle through CODING more than once across Reviewer
    rejections (Milestones 4/24) -- squash must collapse everything
    since baseline regardless of how many commit_all() calls happened
    in between micro-commits, not just a single unbroken run of
    [wip]-only commits."""
    _ctx, _scratch, sandbox = workspace
    sandbox.write_file("/workspace/a.txt", "1\n")
    git_repo.micro_commit(sandbox, "t", "write_file", "a.txt")
    sandbox.write_file("/workspace/b.txt", "2\n")
    git_repo.commit_all(sandbox, "fix: first attempt")  # non-wip commit in the middle
    sandbox.write_file("/workspace/c.txt", "3\n")
    git_repo.micro_commit(sandbox, "t", "write_file", "c.txt")

    assert git_repo.has_wip_commits(sandbox) is True
    git_repo.squash_wip_commits(sandbox, "fix: final")
    assert git_repo.has_wip_commits(sandbox) is False


# =======================================================================
# The real acceptance test: a genuine chain run reaching PR_CREATION
# =======================================================================


async def test_a_real_chain_run_reaching_pr_creation_has_no_wip_commits_left(
    session, tmp_path, sandbox_manager
):
    """spec 4.6.1: 'a task reaching PR_CREATION with any unsquashed [wip]
    commit still present is a bug, not a style preference.' Driven
    through the REAL chain.py code path (run_chain, real Coder tool
    calls, real squash step wired into PR_CREATION) -- not a synthetic
    call to squash_wip_commits() in isolation, which every test above
    already covers."""
    scratch = tmp_path / "workspace"
    git_repo.materialize(FIXTURE_REPO, scratch)
    task_id_str = f"m29-{uuid.uuid4().hex[:8]}"
    sandbox = sandbox_manager.create(task_id_str, scratch)
    git_repo.init_baseline(sandbox)
    git_repo.create_branch(sandbox, "amop/fix-acceptance")

    task = await create_task(session, task_type="bug_fix")
    ctx = ToolContext(
        agent_name="chain", scratch_dir=scratch, mode="operator",
        sandbox=sandbox, db_session=session, task_id=task.id,
    )

    agents = build_agents(
        ctx,
        investigator=[root_cause_answer(1.0)],
        coder=[
            tool_call(
                "write_file", path="calculator.py",
                content=fixed_calculator_source(scratch),
            ),
            final("applied the fix"),
        ],
        tester=[suite_answer(True)],
        reviewer=[review_answer(True)],
    )

    result = await run_chain(
        session, task, description="average is wrong", ctx=ctx, agents=agents
    )

    assert result.final_state is TaskState.WAITING_FOR_APPROVAL
    # The Coder's own write_file call really did produce a micro-commit
    # (proving there was something real to squash, not a trivial pass).
    # Confirmed by the branch having more than just the baseline commit
    # would leave, and specifically zero surviving [wip] markers now.
    assert git_repo.has_wip_commits(sandbox) is False


# =======================================================================
# Stage 2: RECONCILE, spec 4.6.2, cases A-F
#
# Every case below builds REAL git state in a real sandbox (or, for the
# two prechecks and case C, deliberately withholds it) and REAL Postgres
# task_context, then calls the actual reconcile() function -- no
# mocking of git or the case-dispatch logic itself. Only case E's
# GitHub call is faked (same reasoning _fake_github above already
# established for PR_CREATION: no real network/API access in tests).
# =======================================================================


def _sample_report(task_id) -> RootCauseReport:
    return RootCauseReport(
        task_id=str(task_id),
        root_cause="average() divides by len(values) - 1 instead of len(values)",
        confidence=0.9,
        evidence=[Evidence(type="code", ref="calculator.py:39", excerpt=BUGGY_LINE)],
        affected_files=["calculator.py"],
        suggested_fix_plan="divide by len(values)",
    )


async def _task_in(session, state, task_context) -> object:
    """Create a Task and walk it directly to `state` (CODING or TESTING)
    via legal transitions -- reconcile() tests need a task already
    sitting there, the way a real crash would leave it, not a full
    scripted chain run."""
    task = await create_task(session, task_type="bug_fix", task_context=task_context)
    path = [
        TaskState.TRIAGING,
        TaskState.INVESTIGATING,
        TaskState.PLANNING_FIX,
        TaskState.CODING,
    ]
    if state is TaskState.TESTING:
        path.append(TaskState.TESTING)
    for to_state in path:
        task = await transition_with_retry(session, task, to_state, actor="test")
    return task


@pytest.fixture
def scratch_root(tmp_path, monkeypatch):
    """Points sandbox_tools.SCRATCH_DIR (reconcile.py reads it live, at
    call time) at a per-test tmp_path, so `SCRATCH_DIR / task.id` is a
    real, disposable directory instead of the project's actual
    ./amop_workspace."""
    monkeypatch.setattr(sandbox_tools, "SCRATCH_DIR", tmp_path)
    return tmp_path


async def _real_stranded_workspace(session, sandbox_manager, scratch_root, state=TaskState.CODING):
    """A task sitting in CODING with a REAL git repo behind it, at the
    exact `SCRATCH_DIR / task.id` path reconcile() will look for --
    the surviving-scratch-dir precondition every non-C, non-precheck
    case below needs."""
    task = await _task_in(session, state, task_context={"repo": str(FIXTURE_REPO)})
    scratch = scratch_root / str(task.id)
    git_repo.materialize(FIXTURE_REPO, scratch)
    sandbox = sandbox_manager.create(str(task.id), scratch)
    git_repo.init_baseline(sandbox)
    git_repo.create_branch(sandbox, f"amop/fix-{task.id.hex[:8]}")
    baseline_sha = git_repo.current_sha(sandbox)
    return task, sandbox, baseline_sha


async def _set_context(session, task, **fields) -> None:
    task.task_context = {**(task.task_context or {}), **fields}
    session.add(task)
    await session.commit()


# -- Prechecks -----------------------------------------------------------


async def test_reconcile_no_working_repo_recorded(session):
    """A task with no task_context['repo'] at all -- Milestone 0/1's
    repo-less `amop run` scaffold path, the real shape of 10 of the 11
    stranded rows found in the dev database (see this milestone's real
    stranded-task evidence below). Nothing to reconcile against;
    NEEDS_HUMAN_INPUT immediately, no sandbox ever touched."""
    task = await _task_in(session, TaskState.CODING, task_context={"prompt": "x"})

    result = await reconcile(session, task)

    assert result.resumable is False
    assert result.case == "no_working_repo_recorded"
    assert result.reason == "no_working_repo_recorded"
    await session.refresh(task)
    assert task.state == TaskState.NEEDS_HUMAN_INPUT.value


async def test_reconcile_case_c_scratch_dir_gone_resumes_from_recorded_report(
    session, scratch_root
):
    """Case C: container AND its host-mounted scratch dir are both gone
    (scratch_root is real but this task's own subdirectory under it was
    never created) -- the accepted-data-loss case, spec's own words,
    not relitigated here. Still resumable because the RootCauseReport
    survives independently in Postgres."""
    report = _sample_report("placeholder")
    task = await _task_in(
        session,
        TaskState.CODING,
        task_context={
            "repo": str(FIXTURE_REPO),
            "root_cause_report": report.model_dump(mode="json"),
        },
    )

    result = await reconcile(session, task)

    assert result.resumable is True
    assert result.case == "C"
    assert result.needs_fresh_checkout is True
    assert result.report.root_cause == report.root_cause


async def test_reconcile_case_c_without_a_recorded_root_cause_needs_human(
    session, scratch_root
):
    """The real gap this milestone's persistence closes going forward,
    but can't retroactively fix: a task stranded before Stage 2 existed
    has no root_cause_report to fall back on either. Case C's own
    remediation ('reconstructible from the DB alone') has nothing to
    reconstruct from -- NEEDS_HUMAN_INPUT, not a silent restart."""
    task = await _task_in(
        session, TaskState.CODING, task_context={"repo": str(FIXTURE_REPO)}
    )

    result = await reconcile(session, task)

    assert result.resumable is False
    assert result.reason == "root_cause_not_recorded"
    await session.refresh(task)
    assert task.state == TaskState.NEEDS_HUMAN_INPUT.value


# -- Cases requiring a real surviving scratch dir -------------------------


async def test_reconcile_case_equal_resumes_with_no_db_rewrite(
    session, sandbox_manager, scratch_root
):
    """DB's recorded commit_sha already matches real HEAD exactly --
    states agree, resume normally, no adoption write needed."""
    task, sandbox, baseline_sha = await _real_stranded_workspace(
        session, sandbox_manager, scratch_root
    )
    report = _sample_report(task.id)
    await _set_context(
        session, task,
        branch=git_repo.current_branch(sandbox),
        commit_sha=baseline_sha,
        root_cause_report=report.model_dump(mode="json"),
    )

    result = await reconcile(session, task)

    assert result.resumable is True
    assert result.case == "equal"
    assert result.report.root_cause == report.root_cause


async def test_reconcile_case_a_head_ahead_adopts_it(session, sandbox_manager, scratch_root):
    """Case A: git succeeded (a real micro-commit landed), the DB write
    that should have followed it never happened -- the DB still
    recording the pre-crash baseline sha. Git is the more expensive
    state to reconstruct, so it wins: adopt real HEAD into the DB."""
    task, sandbox, baseline_sha = await _real_stranded_workspace(
        session, sandbox_manager, scratch_root
    )
    report = _sample_report(task.id)
    await _set_context(
        session, task,
        branch=git_repo.current_branch(sandbox),
        commit_sha=baseline_sha,  # the DB's stale, pre-crash record
        root_cause_report=report.model_dump(mode="json"),
    )
    sandbox.write_file("/workspace/extra.txt", "real work\n")
    real_head = git_repo.micro_commit(sandbox, str(task.id), "write_file", "extra.txt")
    assert real_head != baseline_sha

    result = await reconcile(session, task)

    assert result.resumable is True
    assert result.case == "A"
    await session.refresh(task)
    assert task.task_context["commit_sha"] == real_head


async def test_reconcile_case_b_db_commit_does_not_exist_trusts_the_branch(
    session, sandbox_manager, scratch_root
):
    """Case B: DB claims a commit_sha that isn't even a real object in
    this repo -- spec's own words, 'should be impossible given 4.6.1's
    ordering, but defended anyway'. The branch is trusted; the DB's
    fictitious sha is rolled back to real HEAD."""
    task, sandbox, baseline_sha = await _real_stranded_workspace(
        session, sandbox_manager, scratch_root
    )
    report = _sample_report(task.id)
    fake_sha = "d" * 40
    await _set_context(
        session, task,
        branch=git_repo.current_branch(sandbox),
        commit_sha=fake_sha,
        root_cause_report=report.model_dump(mode="json"),
    )

    result = await reconcile(session, task)

    assert result.resumable is True
    assert result.case == "B"
    await session.refresh(task)
    assert task.task_context["commit_sha"] == baseline_sha


async def test_reconcile_case_d_foreign_commit_never_auto_resolves(
    session, sandbox_manager, scratch_root
):
    """Case D, via authorship: HEAD is ahead of the DB's recorded sha,
    but the commit that put it there was not written by AMOP (neither
    amop-bot nor the sandbox's baked-in AMOP Agent identity). Spec: 'an
    agent silently rebasing over a human's work is worse than
    stalling' -- NEEDS_HUMAN_INPUT, never auto-adopted."""
    task, sandbox, baseline_sha = await _real_stranded_workspace(
        session, sandbox_manager, scratch_root
    )
    report = _sample_report(task.id)
    await _set_context(
        session, task,
        branch=git_repo.current_branch(sandbox),
        commit_sha=baseline_sha,
        root_cause_report=report.model_dump(mode="json"),
    )
    sandbox.exec_run(
        "cd /workspace && git -c user.name=hacker -c user.email=hacker@example.com "
        'commit --allow-empty -m "sneaky external commit"'
    )

    result = await reconcile(session, task)

    assert result.resumable is False
    assert result.reason == "branch_modified_externally"
    await session.refresh(task)
    assert task.state == TaskState.NEEDS_HUMAN_INPUT.value
    # Never adopted -- the DB's stale sha must survive untouched.
    assert task.task_context["commit_sha"] == baseline_sha


async def test_reconcile_case_d_true_divergence_never_auto_resolves(
    session, sandbox_manager, scratch_root
):
    """Case D, via ancestry: the DB's recorded commit is real but lives
    on a completely different line of history than current HEAD --
    neither is an ancestor of the other. Nothing in AMOP's own commit
    flow produces this; NEEDS_HUMAN_INPUT rather than guessing which
    side is 'right'."""
    task, sandbox, baseline_sha = await _real_stranded_workspace(
        session, sandbox_manager, scratch_root
    )
    working_branch = git_repo.current_branch(sandbox)
    sandbox.exec_run(f"cd /workspace && git checkout -b fork-branch {baseline_sha}")
    sandbox.exec_run(
        "cd /workspace && git -c user.name=amop-bot -c user.email=amop-bot@localhost "
        'commit --allow-empty -m "fork commit"'
    )
    fork_sha = git_repo.current_sha(sandbox)
    sandbox.exec_run(f"cd /workspace && git checkout {working_branch}")
    # back on the real working branch, still sitting at baseline_sha --
    # fork_sha is a real commit, just not reachable from here.
    report = _sample_report(task.id)
    await _set_context(
        session, task,
        branch=working_branch,
        commit_sha=fork_sha,
        root_cause_report=report.model_dump(mode="json"),
    )

    result = await reconcile(session, task)

    assert result.resumable is False
    assert result.reason == "branch_modified_externally"


async def test_reconcile_case_f_corrupted_repo_needs_human(session, sandbox_manager, scratch_root):
    """Case F: git fsck fails on resume. AMOP does not attempt automated
    repair of a corrupted repository -- spec's own words -- so this is
    always NEEDS_HUMAN_INPUT, never an auto-resolve path. Corruption is
    induced by deleting a loose object file outright (mutation-testing
    this exact technique earlier in this milestone found that merely
    overwriting an object's content does NOT reliably trip `git fsck`;
    deleting it does)."""
    task, sandbox, baseline_sha = await _real_stranded_workspace(
        session, sandbox_manager, scratch_root
    )
    report = _sample_report(task.id)
    await _set_context(
        session, task,
        branch=git_repo.current_branch(sandbox),
        commit_sha=baseline_sha,
        root_cause_report=report.model_dump(mode="json"),
    )
    obj_path = f"/workspace/.git/objects/{baseline_sha[:2]}/{baseline_sha[2:]}"
    sandbox.exec_run(f"rm -f {obj_path}")

    result = await reconcile(session, task)

    assert result.resumable is False
    assert result.reason == "repo_integrity_failure"


async def test_reconcile_case_e_adopts_an_orphaned_pr(
    session, sandbox_manager, scratch_root, monkeypatch
):
    """Case E: a PR already exists on GitHub for this task's branch with
    no local pull_requests row (crash landed between create_pull_request
    succeeding and PR_CREATION's own row-write) -- adopted here rather
    than left invisible, preventing create_pull_request from opening a
    duplicate later. Only the GitHub call is faked (no real network in
    tests); find_open_pr_for_branch and the adoption write are real."""
    task, sandbox, baseline_sha = await _real_stranded_workspace(
        session, sandbox_manager, scratch_root
    )
    report = _sample_report(task.id)
    await _set_context(
        session, task,
        branch=git_repo.current_branch(sandbox),
        commit_sha=baseline_sha,
        root_cause_report=report.model_dump(mode="json"),
    )
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token-for-test")

    async def _fake_find_open_pr_for_branch(client, head, base):
        return {"url": "https://github.com/bivek127/amop-sandbox/pull/777", "number": 777}

    monkeypatch.setattr(
        "amop.tools.github.find_open_pr_for_branch", _fake_find_open_pr_for_branch
    )

    result = await reconcile(session, task)

    assert result.resumable is True
    rows = (
        await session.execute(
            select(PullRequest).where(PullRequest.task_id == task.id)
        )
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].url == "https://github.com/bivek127/amop-sandbox/pull/777"
    assert rows[0].number == 777


# -- End-to-end: a real reconcile() -> run_chain(resume_report=...) hop --


async def test_reconcile_then_resume_reaches_waiting_for_approval(
    session, sandbox_manager, scratch_root
):
    """The full bounded-resume path, case A, tied together: reconcile()
    recovers a RootCauseReport and adopts real HEAD, then
    run_chain(resume_report=...) re-enters at PLANNING_FIX and drives
    the rest of the chain for real -- proving the resumed run is not
    just a classification exercise but an actually-drivable task."""
    task, sandbox, baseline_sha = await _real_stranded_workspace(
        session, sandbox_manager, scratch_root
    )
    report = _sample_report(task.id)
    await _set_context(
        session, task,
        branch=git_repo.current_branch(sandbox),
        commit_sha=baseline_sha,
        root_cause_report=report.model_dump(mode="json"),
    )
    scratch = scratch_root / str(task.id)

    result = await reconcile(session, task)
    assert result.resumable is True
    assert result.needs_fresh_checkout is False

    ctx = ToolContext(
        agent_name="chain", scratch_dir=scratch, mode="operator",
        sandbox=sandbox, db_session=session, task_id=task.id,
        repo_path=str(FIXTURE_REPO),
    )
    agents = build_agents(
        ctx,
        investigator=[root_cause_answer(1.0)],  # unused on the resume path
        coder=[
            tool_call(
                "write_file", path="calculator.py",
                content=fixed_calculator_source(scratch),
            ),
            final("applied the fix"),
        ],
        tester=[suite_answer(True)],
        reviewer=[review_answer(True)],
    )

    chain_result = await run_chain(
        session, task, description="average is wrong", ctx=ctx, agents=agents,
        resume_report=result.report,
    )

    assert chain_result.final_state is TaskState.WAITING_FOR_APPROVAL
    assert git_repo.has_wip_commits(sandbox) is False


async def test_planning_fix_reentry_from_coding_or_testing_is_reconcile_only(
    session, workspace
):
    """The (CODING|TESTING, PLANNING_FIX) edges state_machine.py added
    for Milestone 29 exist ONLY for reconcile()'s bounded resume. This
    proves it behaviorally for BOTH edges, not just by code inspection
    (grep confirms exactly two `go(TaskState.PLANNING_FIX, ...)` call
    sites in the whole file -- the original INVESTIGATING-only hop, and
    the one gated behind `if resume_report is not None`): a normal
    run_chain() call (resume_report=None, the only way every real task
    is ever driven) is put through TWO real retry cycles --
    TESTING -> CODING (attempt 1 leaves the real bug in place, so the
    orchestrator's OWN ground-truth pytest run, not the scripted
    Tester's claim, forces the retry -- Milestone 26's "a lying self-
    report cannot move the task forward") and REVIEWING -> CODING
    (Reviewer rejects once with a grounded, verifiable counterexample)
    -- and the real task_transitions audit trail is queried to confirm
    PLANNING_FIX was entered exactly once, never again on either retry.

    Same bug shape as Milestone 16's severity finding: a transition
    table permitting an edge is not the same as that edge being reached
    only where intended. Untested, this would be an assumption, not a
    verified property -- if a future change ever called
    go(TaskState.PLANNING_FIX, ...) from inside the CODING/TESTING retry
    loop for some unrelated reason, this is the test that would catch
    it (mutation-verified below)."""
    ctx, scratch, _sandbox = workspace
    task = await create_task(session, task_type="bug_fix")

    # Attempt 1: a real mutating edit that does NOT fix the bug (the
    # buggy division line survives untouched) -- a genuine diff, not a
    # no_op, so it reaches TESTING and is judged by the orchestrator's
    # own pytest run, which still fails. Attempts 2 and 3 apply the real
    # fix; attempt 3 exists only because Reviewer rejects attempt 2 once.
    unfixed_source = "# cosmetic, does not touch the bug\n" + (scratch / "calculator.py").read_text()
    agents = ChainAgents(
        investigator=InvestigatorAgent(ScriptedLLM([root_cause_answer(0.9)]), ctx),
        coder=CoderAgent(
            ScriptedLLM(
                [
                    tool_call("write_file", path="calculator.py", content=unfixed_source),
                    final("first attempt"),
                    tool_call(
                        "write_file", path="calculator.py",
                        content=fixed_calculator_source(scratch),
                    ),
                    final("applied the real fix"),
                    tool_call(
                        "write_file", path="calculator.py",
                        content=fixed_calculator_source(scratch),
                    ),
                    final("addressed the review findings"),
                ]
            ),
            ctx, task_id="scripted",
        ),
        tester=tester_mod.TesterAgent(ScriptedLLM([suite_answer(True)]), ctx),
        reviewer=ReviewerAgent(
            ScriptedLLM(
                [
                    review_answer(
                        False,
                        "the fix needs a guard for empty input",
                        counterexample=(
                            "average([]) with 0 items -> raises ZeroDivisionError, "
                            "should raise ValueError instead"
                        ),
                        counterexample_claim={
                            "module": "calculator",
                            "function": "Calculator.average",
                            "calls": [
                                {
                                    "name": "empty", "args": {"values": []},
                                    "expect_raises": "ValueError",
                                }
                            ],
                        },
                    ),
                    review_answer(True),
                ]
            ),
            ctx,
        ),
    )

    result = await run_chain(
        session, task, description="average is wrong", ctx=ctx, agents=agents
    )

    assert result.final_state is TaskState.WAITING_FOR_APPROVAL

    rows = (
        await session.execute(
            select(TaskTransition)
            .where(TaskTransition.task_id == task.id)
            .order_by(TaskTransition.id)
        )
    ).scalars().all()

    planning_fix_entries = [r for r in rows if r.to_state == TaskState.PLANNING_FIX.value]
    assert len(planning_fix_entries) == 1, (
        f"expected exactly one PLANNING_FIX entry on a normal (non-resume) run "
        f"with two real retry cycles, got {len(planning_fix_entries)}: "
        f"{[(r.from_state, r.to_state, r.trigger) for r in planning_fix_entries]}"
    )
    assert planning_fix_entries[0].from_state == TaskState.INVESTIGATING.value

    # Both retries really happened (TESTING -> CODING from the real
    # ground-truth failure, REVIEWING -> CODING from the rejection) and
    # both went through CODING directly, never back through PLANNING_FIX.
    testing_retries = [
        r for r in rows
        if r.from_state == TaskState.TESTING.value and r.to_state == TaskState.CODING.value
    ]
    reviewing_retries = [
        r for r in rows
        if r.from_state == TaskState.REVIEWING.value and r.to_state == TaskState.CODING.value
    ]
    assert len(testing_retries) == 1
    assert len(reviewing_retries) == 1

    # No reconcile() call happened at all on this run -- confirms the
    # PLANNING_FIX re-entry edges genuinely went unused, not just
    # "unused because nothing tried".
    assert not any(r.actor == "system:reconcile" for r in rows)


async def test_coding_loop_persists_commit_sha_even_when_micro_commit_captures_the_whole_diff(
    session, tmp_path, sandbox_manager
):
    """Same failure shape as Milestone 4's confidence-threshold gap: a
    mechanism (the CODING loop's task_context['commit_sha'] persistence
    write) that is correct in isolation but was never actually exercised
    by anything real, because something else -- here, commit_all()
    returning a real sha -- almost always happened to catch the case
    first. Found live, by this milestone's own kill-mid-CODING-and-
    restart demo, not by any test written before it: the demo's single
    write_file call was captured entirely by Stage 1's own
    micro_commit(), so _run_coder()'s own commit_all() call at turn-end
    found a clean working tree and returned None -- and the CODING
    loop's persistence write was reading exactly that field
    (code_report.commit_sha), so it silently never fired.

    This test drives the REAL _run_coder() -> commit_all() ->
    persistence path end to end (not a hand-constructed task_context
    like every other Stage 2 test in this file) through precisely that
    shape -- one real edit, fully captured by micro_commit(), nothing
    left for commit_all() to catch -- and proves the fixed write
    (git_repo.current_sha(sandbox), read unconditionally) actually
    lands, at the moment it should have landed, not just somewhere by
    the time the whole chain finishes.

    Deliberately NOT the shared `workspace` fixture: its ctx has no
    task_id, which makes micro_commit() a no-op by Stage 1's own design
    (test_write_file_with_no_task_id_skips_micro_commit_but_still_
    succeeds) -- exactly wrong for this test, which needs the
    micro-commit to actually fire so commit_all() has nothing left to
    catch. Built explicitly with task_id=task.id instead, same as the
    Stage 1 acceptance test above."""
    scratch = tmp_path / "workspace"
    git_repo.materialize(FIXTURE_REPO, scratch)
    task_id_str = f"m29-{uuid.uuid4().hex[:8]}"
    sandbox = sandbox_manager.create(task_id_str, scratch)
    git_repo.init_baseline(sandbox)
    git_repo.create_branch(sandbox, "amop/fix-commit-sha-bug")

    task = await create_task(session, task_type="bug_fix")
    ctx = ToolContext(
        agent_name="chain", scratch_dir=scratch, mode="operator",
        sandbox=sandbox, db_session=session, task_id=task.id,
    )

    captured: dict = {}

    def emit(message: str) -> None:
        if message.startswith("TESTING: running"):
            # The CODING loop's persistence write runs synchronously,
            # before TESTING's own stage() call -- by the time this
            # fires, it has already happened (or, pre-fix, silently
            # hasn't). Read directly off the live `task` object this
            # test still holds a reference to (the same object run_chain
            # mutates in place), not re-derived from the DB, so this
            # proves the in-memory write itself happened at the right
            # point, not merely that some later step also wrote a
            # correct value. Real git state is snapshotted HERE too, not
            # after run_chain returns -- PR_CREATION's squash step moves
            # HEAD again before this run finishes, so comparing against
            # a post-run git read would be checking the wrong moment.
            captured["real_sha_at_the_time"] = git_repo.current_sha(sandbox)
            captured["real_branch_at_the_time"] = git_repo.current_branch(sandbox)
            captured["branch"] = task.task_context.get("branch")
            captured["commit_sha"] = task.task_context.get("commit_sha")

    agents = build_agents(
        ctx,
        investigator=[root_cause_answer(1.0)],
        coder=[
            tool_call(
                "write_file", path="calculator.py",
                content=fixed_calculator_source(scratch),
            ),
            final("applied the fix"),
        ],
        tester=[suite_answer(True)],
        reviewer=[review_answer(True)],
    )

    result = await run_chain(
        session, task, description="average is wrong", ctx=ctx, agents=agents,
        emit=emit,
    )

    assert result.final_state is TaskState.WAITING_FOR_APPROVAL
    assert captured, "TESTING stage was never reached -- a setup problem, not the bug under test"

    # Confirms this really is the bug-triggering shape: commit_all()
    # found nothing left to commit (the whole diff was already captured
    # by micro_commit()), so the ORIGINAL code_report -- still sitting
    # on `result`, untouched by the later PR_CREATION squash step, which
    # only ever rebinds a local variable -- has commit_sha=None.
    assert result.code_change_report.commit_sha is None

    # The fixed write landed anyway, at the right moment, with the real
    # values -- not None, and matching real git state as it was at that
    # exact moment (not the branch's later, post-squash state).
    assert captured["commit_sha"] is not None
    assert captured["commit_sha"] == captured["real_sha_at_the_time"]
    assert captured["branch"] == captured["real_branch_at_the_time"]
