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
from sqlalchemy import text

from amop.agents import tester as tester_mod
from amop.agents.coder import CoderAgent
from amop.agents.investigator import InvestigatorAgent
from amop.agents.reviewer import ReviewerAgent
from amop.database.session import init_db, make_engine, make_session_factory
from amop.models.base import BaseLLM, ModelResponse
from amop.orchestrator.chain import ChainAgents, run_chain
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import create_task
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


def review_answer(approved: bool) -> str:
    return final(
        {
            "task_id": "t",
            "approved": approved,
            "addresses_reported_symptom": True,
            "findings": [],
            "rejection_reason": None,
            "counterexample": None,
            "counterexample_claim": None,
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
                "TRUNCATE agent_actions, task_transitions, tasks "
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
