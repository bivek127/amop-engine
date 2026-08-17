"""Milestone 6 — Close Out Phase 1.

Two tiers, same discipline as Milestones 4/5 (spec 19.2/19.3):

  * Deterministic (always run): secret_scan / scope_guard as pure
    functions, create_pull_request's gate ordering (network mocked out),
    the flaky-test carve-out as pure functions, schema/state-machine
    additions, and two chain-level scripted-model runs (oversized diff,
    planted secret) that never touch the real network.
  * Real GitHub end-to-end, opt-in via AMOP_E2E_GITHUB=1: opens an actual
    pull request on github.com/bivek127/amop-sandbox. Never runs on a
    plain `pytest` invocation.

Requires: Postgres at TEST_DATABASE_URL, and a running Docker daemon (same
as Milestones 4/5). The opt-in test additionally needs GITHUB_TOKEN (from
.env) and real network access to github.com.
"""

import json
import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text

import amop.tools.github as github
from amop.agents import handoffs  # TestReport accessed via the module: pytest collects any bare Test* name as a test class
from amop.agents import tester as tester_mod
from amop.agents.coder import CoderAgent
from amop.agents.handoffs import CodeChangeReport
from amop.agents.investigator import InvestigatorAgent
from amop.agents.reviewer import ReviewerAgent
from amop.database.session import init_db, make_engine, make_session_factory
from amop.models.base import BaseLLM, ModelResponse
from amop.orchestrator import state_machine
from amop.orchestrator.chain import ChainAgents, run_chain
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import create_task, get_transitions
from amop.safety import scope_guard, secret_scan
from amop.sandbox import repo as git_repo
from amop.sandbox import tools as sandbox_tools  # noqa: F401 -- registers the sandboxed tools
from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext, invoke_tool

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://localhost/amop_test"
)
FIXTURE_REPO = Path(__file__).parent / "fixtures" / "buggy_calculator"
BUGGY_LINE = "return sum(values) / (len(values) - 1)"
FIXED_LINE = "return sum(values) / len(values)"

# Exactly 16 chars after AKIA -- matches secret_scan's _AWS_ACCESS_KEY_RE.
PLANTED_AWS_KEY = "AKIAABCDEFGHIJKLMNOP"


# ---------------------------------------------------------------------
# safety/secret_scan.py -- pure, no I/O
# ---------------------------------------------------------------------


def test_secret_scan_catches_planted_aws_key():
    diff = f"--- a/config.py\n+++ b/config.py\n+API_KEY = \"{PLANTED_AWS_KEY}\"\n"
    hits = secret_scan.scan_for_secrets(diff)
    assert hits
    assert any(h.pattern_name == "aws_access_key_id" for h in hits)


def test_secret_scan_catches_generic_planted_key():
    # No vendor-specific shape -- exercises the generic catch-all pattern.
    diff = '--- a/config.py\n+++ b/config.py\n+my_api_key = "totally-not-a-real-secret-value"\n'
    hits = secret_scan.scan_for_secrets(diff)
    assert hits
    assert any(h.pattern_name == "generic_secret_assignment" for h in hits)


def test_secret_scan_clean_diff_returns_empty():
    diff = "--- a/calc.py\n+++ b/calc.py\n+def add(a, b):\n+    return a + b\n"
    assert secret_scan.scan_for_secrets(diff) == []


def test_secret_scan_ignores_removed_and_context_lines():
    # The secret only appears on a removed ('-') line and the file-header
    # ('+++') line -- neither counts as newly added.
    diff = (
        f"--- a/config.py\n"
        f"+++ b/config.py\n"
        f"-API_KEY = \"{PLANTED_AWS_KEY}\"\n"
        f" unrelated_context_line = 1\n"
        f"+safe_value = 42\n"
    )
    assert secret_scan.scan_for_secrets(diff) == []


def test_secret_match_never_carries_the_matched_text():
    hits = secret_scan.scan_for_secrets(
        f"+++ b/x.py\n+KEY = \"{PLANTED_AWS_KEY}\"\n"
    )
    assert hits
    assert not hasattr(hits[0], "matched_text")
    assert PLANTED_AWS_KEY not in hits[0].reason


# ---------------------------------------------------------------------
# safety/scope_guard.py -- pure, no I/O
# ---------------------------------------------------------------------


def _diff_with_n_added_lines(n: int) -> str:
    lines = ["--- a/foo.py", "+++ b/foo.py"] + [f"+line {i}" for i in range(n)]
    return "\n".join(lines)


def test_scope_guard_changed_line_count_excludes_file_headers():
    diff = "--- a/foo.py\n+++ b/foo.py\n+added\n-removed\n context\n"
    assert scope_guard.changed_line_count(diff) == 2


def test_scope_guard_under_cap_not_over():
    assert not scope_guard.over_cap(_diff_with_n_added_lines(100))


def test_scope_guard_at_cap_not_over():
    assert not scope_guard.over_cap(_diff_with_n_added_lines(150))


def test_scope_guard_over_cap():
    assert scope_guard.over_cap(_diff_with_n_added_lines(151))


# ---------------------------------------------------------------------
# tools/github.py -- create_pull_request's gate ordering, network mocked
# ---------------------------------------------------------------------


@pytest.fixture
def fake_ctx(tmp_path):
    # A non-None sandbox sentinel is enough -- diff_against_baseline is
    # monkeypatched per-test, never actually called against a real
    # container here.
    return ToolContext(
        agent_name="test", scratch_dir=tmp_path, mode="operator", sandbox=object()
    )


def _refuse(*_args, **_kwargs):
    raise AssertionError("should not have been called -- an earlier gate should have blocked this")


async def test_create_pull_request_blocks_on_secret_before_any_network_call(
    monkeypatch, fake_ctx
):
    monkeypatch.setattr(
        github.git_repo,
        "diff_against_baseline",
        lambda sandbox, base: f"+++ b/x.py\n+KEY = \"{PLANTED_AWS_KEY}\"\n",
    )
    monkeypatch.setattr(github.git_repo, "push_to_remote", _refuse)
    monkeypatch.setattr(github, "_client", _refuse)

    result = await invoke_tool(
        "create_pull_request",
        {"title": "t", "body": "b", "head": "amop/x", "base": "main"},
        fake_ctx,
        agent_name="test",
    )
    assert not result.success
    assert result.error_code == "SECRET_DETECTED"


async def test_create_pull_request_blocks_on_oversized_diff_before_any_network_call(
    monkeypatch, fake_ctx
):
    monkeypatch.setattr(
        github.git_repo,
        "diff_against_baseline",
        lambda sandbox, base: _diff_with_n_added_lines(200),
    )
    monkeypatch.setattr(github.git_repo, "push_to_remote", _refuse)
    monkeypatch.setattr(github, "_client", _refuse)

    result = await invoke_tool(
        "create_pull_request",
        {"title": "t", "body": "b", "head": "amop/x", "base": "main"},
        fake_ctx,
        agent_name="test",
    )
    assert not result.success
    assert result.error_code == "DIFF_TOO_LARGE"


class _FakePR:
    def __init__(self, number, url):
        self.number = number
        self.html_url = url


class _FakeRepoWithExistingPR:
    def get_pulls(self, state=None, head=None, base=None):
        return [_FakePR(42, "https://github.com/bivek127/amop-sandbox/pull/42")]

    def create_pull(self, **_kwargs):
        raise AssertionError("create_pull must not be called when an open PR already exists")


async def test_create_pull_request_adopts_existing_open_pr_instead_of_duplicating(
    monkeypatch, fake_ctx
):
    monkeypatch.setattr(
        github.git_repo,
        "diff_against_baseline",
        lambda sandbox, base: "+++ b/x.py\n+print('ok')\n",
    )
    monkeypatch.setattr(github.git_repo, "push_to_remote", lambda *a, **k: None)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake0000000000000000000000000000000")
    monkeypatch.setattr(
        github, "_client", lambda token: type("C", (), {"get_repo": lambda self, name: _FakeRepoWithExistingPR()})()
    )

    result = await invoke_tool(
        "create_pull_request",
        {"title": "t", "body": "b", "head": "amop/x", "base": "main"},
        fake_ctx,
        agent_name="test",
    )
    assert result.success, result.message
    assert result.output["created"] is False
    assert result.output["url"] == "https://github.com/bivek127/amop-sandbox/pull/42"


async def test_create_pull_request_missing_token_blocked_before_push(monkeypatch, fake_ctx):
    monkeypatch.setattr(
        github.git_repo,
        "diff_against_baseline",
        lambda sandbox, base: "+++ b/x.py\n+print('ok')\n",
    )
    monkeypatch.setattr(github.git_repo, "push_to_remote", _refuse)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    result = await invoke_tool(
        "create_pull_request",
        {"title": "t", "body": "b", "head": "amop/x", "base": "main"},
        fake_ctx,
        agent_name="test",
    )
    assert not result.success
    assert result.error_code == "GITHUB_AUTH_MISSING"


# ---------------------------------------------------------------------
# agents/tester.py -- flaky-test carve-out, pure, no I/O
# ---------------------------------------------------------------------


def test_flaky_pre_existing_test_is_excluded():
    excluded, effective = tester_mod.classify_failures(
        fix_branch_failures=["test_unrelated_flake"],
        base_branch_results={"test_unrelated_flake": True},
        description="average() divides by len(values)-1 instead of len(values)",
        new_tests_added=[],
    )
    assert excluded == ["test_unrelated_flake"]
    assert effective == []


def test_bug_report_named_test_is_never_excluded_even_failing_both_branches():
    excluded, effective = tester_mod.classify_failures(
        fix_branch_failures=["test_average_empty_list"],
        base_branch_results={"test_average_empty_list": True},
        description="test_average_empty_list is failing on main, please fix",
        new_tests_added=[],
    )
    assert excluded == []
    assert effective == ["test_average_empty_list"]


def test_new_regression_test_is_never_excluded_even_failing_both_branches():
    excluded, effective = tester_mod.classify_failures(
        fix_branch_failures=["test_new_regression"],
        base_branch_results={"test_new_regression": True},
        description="unrelated free-text bug report",
        new_tests_added=["test_new_regression"],
    )
    assert excluded == []
    assert effective == ["test_new_regression"]


def test_fix_branch_only_failure_is_never_excluded():
    # Fails on the fix branch but base_branch_results says it does NOT
    # fail on base -- real signal, must stay in effective regardless.
    excluded, effective = tester_mod.classify_failures(
        fix_branch_failures=["test_real_bug"],
        base_branch_results={"test_real_bug": False},
        description="",
        new_tests_added=[],
    )
    assert excluded == []
    assert effective == ["test_real_bug"]


def test_inconclusive_base_rerun_is_not_excluded():
    # No entry in base_branch_results at all (an errored/timed-out
    # re-run) -- fail-safe default keeps it as real signal.
    excluded, effective = tester_mod.classify_failures(
        fix_branch_failures=["test_maybe_flaky"],
        base_branch_results={},
        description="",
        new_tests_added=[],
    )
    assert excluded == []
    assert effective == ["test_maybe_flaky"]


def test_is_carveout_protected_checks_both_new_tests_and_description():
    assert tester_mod.is_carveout_protected("test_x", "", ["test_x"])
    assert tester_mod.is_carveout_protected("test_y", "please fix test_y", [])
    assert not tester_mod.is_carveout_protected("test_z", "unrelated report", [])


# ---------------------------------------------------------------------
# Schema / state-machine additions
# ---------------------------------------------------------------------


def test_code_change_report_accepts_needs_decomposition_status():
    report = CodeChangeReport(
        task_id="t", status="needs_decomposition", branch="amop/x", files_changed=[]
    )
    assert report.status == "needs_decomposition"


def test_test_report_defaults_and_accepts_new_fields():
    default_report = handoffs.TestReport(task_id="t", all_passed=True)
    assert default_report.new_tests_added == []
    assert default_report.regression_confirmed is False

    populated = handoffs.TestReport(
        task_id="t",
        all_passed=True,
        new_tests_added=["test_x"],
        regression_confirmed=True,
    )
    assert populated.new_tests_added == ["test_x"]
    assert populated.regression_confirmed is True


def test_needs_decomposition_state_transitions_are_legal():
    # Would raise IllegalTransitionError if not registered.
    state_machine.validate_transition(TaskState.CODING, TaskState.NEEDS_HUMAN_INPUT)
    state_machine.validate_transition(TaskState.PR_CREATION, TaskState.NEEDS_HUMAN_INPUT)


# ---------------------------------------------------------------------
# Chain-level scripted tests -- real Postgres + Docker, no Ollama, no
# GitHub network (the secret-block test exercises the REAL secret_scan
# gate inside the real create_pull_request tool; it never reaches push
# because that gate fires first).
# ---------------------------------------------------------------------


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


def root_cause_answer(confidence: float, affected=None) -> str:
    return final(
        {
            "task_id": "t",
            "root_cause": "average() divides by len(values) - 1 instead of len(values)",
            "confidence": confidence,
            "evidence": [
                {"type": "code", "ref": "calculator.py:39", "excerpt": BUGGY_LINE}
            ],
            "affected_files": affected if affected is not None else ["calculator.py"],
            "suggested_fix_plan": "divide by len(values)",
        }
    )


def review_answer(approved: bool, reason: str | None = None) -> str:
    return final(
        {
            "task_id": "t",
            "approved": approved,
            "addresses_reported_symptom": True,
            "findings": [],
            "rejection_reason": reason,
            "counterexample": None,
        }
    )


def suite_answer(all_passed: bool, details: str = "ok") -> str:
    return final({"task_id": "t", "all_passed": all_passed, "details": details})


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
            text("TRUNCATE task_transitions, tasks RESTART IDENTITY CASCADE")
        )


@pytest.fixture
def workspace(tmp_path):
    """Same shape as test_milestone4.py's fixture, with a UNIQUE branch
    name per test run (not a fixed "amop/fix-test") -- the opt-in real
    E2E test below pushes this branch to a real remote, and a fixed name
    would collide with a previous run's unrelated git history."""
    scratch = tmp_path / "workspace"
    git_repo.materialize(FIXTURE_REPO, scratch)

    manager = SandboxManager()
    task_id = f"m6-{uuid.uuid4().hex[:8]}"
    sandbox = manager.create(task_id, scratch)
    try:
        git_repo.init_baseline(sandbox)
        git_repo.create_branch(sandbox, f"amop/fix-{uuid.uuid4().hex[:8]}")
        ctx = ToolContext(
            agent_name="chain", scratch_dir=scratch, mode="operator", sandbox=sandbox
        )
        yield ctx, scratch, sandbox
    finally:
        manager.destroy(task_id)


def fixed_calculator_source(scratch: Path) -> str:
    source = (scratch / "calculator.py").read_text()
    assert BUGGY_LINE in source
    return source.replace(BUGGY_LINE, FIXED_LINE)


def oversized_calculator_source(scratch: Path) -> str:
    """A whole-file rewrite of calculator.py, fixed but padded well past
    the 150-line cap -- exercises scope_guard.py's abort path."""
    fixed = fixed_calculator_source(scratch)
    padding = "\n".join(f"    # padding line {i} to blow the diff-size cap" for i in range(200))
    return f"{fixed}\n\n\nclass _Padding:\n{padding}\n"


def calculator_source_with_planted_secret(scratch: Path) -> str:
    fixed = fixed_calculator_source(scratch)
    return f'{fixed}\n\n# leftover debug credential, never do this:\nAWS_KEY = "{PLANTED_AWS_KEY}"\n'


def build_agents(ctx, *, investigator, coder, tester, reviewer) -> ChainAgents:
    return ChainAgents(
        investigator=InvestigatorAgent(ScriptedLLM(investigator), ctx),
        coder=CoderAgent(ScriptedLLM(coder), ctx, task_id="scripted"),
        tester=tester_mod.TesterAgent(ScriptedLLM(tester), ctx),
        reviewer=ReviewerAgent(ScriptedLLM(reviewer), ctx),
    )


async def state_history(session, task_id) -> list[str]:
    return [t.to_state for t in await get_transitions(session, task_id)]


async def test_oversized_diff_triggers_needs_decomposition(session, workspace):
    ctx, scratch, _sandbox = workspace
    task = await create_task(session, task_type="bug_fix")

    agents = build_agents(
        ctx,
        investigator=[root_cause_answer(0.9)],
        coder=[
            tool_call(
                "write_file", path="calculator.py", content=oversized_calculator_source(scratch)
            ),
            final("fixed it (this should never be reached -- aborted before TESTING)"),
        ],
        tester=[suite_answer(True)],
        reviewer=[review_answer(True)],
    )

    result = await run_chain(
        session, task, description="average is wrong", ctx=ctx, agents=agents
    )

    assert result.final_state is TaskState.NEEDS_HUMAN_INPUT
    assert result.code_change_report.status == "needs_decomposition"

    history = await state_history(session, task.id)
    assert TaskState.TESTING.value not in history  # aborted before ever testing
    assert TaskState.PR_CREATION.value not in history


async def test_planted_secret_blocks_pr_creation_before_any_network_call(session, workspace):
    ctx, scratch, _sandbox = workspace
    task = await create_task(session, task_type="bug_fix")

    agents = build_agents(
        ctx,
        investigator=[root_cause_answer(0.9)],
        coder=[
            tool_call(
                "write_file",
                path="calculator.py",
                content=calculator_source_with_planted_secret(scratch),
            ),
            final("fixed the denominator"),
        ],
        tester=[suite_answer(True, "all green")],
        reviewer=[tool_call("get_diff"), review_answer(True)],
    )

    result = await run_chain(
        session,
        task,
        description="average() returns the wrong number",
        ctx=ctx,
        agents=agents,
    )

    # Blocked by the REAL secret_scan gate inside the REAL
    # create_pull_request tool (not mocked) -- proves the gate fires
    # before any push/API call would even be attempted, since a push
    # here (no valid remote for this local-only fixture branch) would
    # fail loudly and differently if the scan hadn't already stopped it.
    assert result.final_state is TaskState.NEEDS_HUMAN_INPUT
    assert "secret" in (result.error or "").lower()
    assert "aws_access_key_id" in (result.error or "")

    history = await state_history(session, task.id)
    assert TaskState.PR_CREATION.value in history
    assert TaskState.WAITING_FOR_APPROVAL.value not in history


# ---------------------------------------------------------------------
# Opt-in real GitHub E2E (mirrors AMOP_E2E_OLLAMA's exact pattern)
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("AMOP_E2E_GITHUB") != "1",
    reason="real GitHub E2E: set AMOP_E2E_GITHUB=1 (opens a real PR on github.com/bivek127/amop-sandbox)",
)
async def test_real_github_pr_opens_on_sandbox_repo(session, workspace):
    ctx, scratch, _sandbox = workspace
    task = await create_task(session, task_type="bug_fix")

    agents = build_agents(
        ctx,
        investigator=[root_cause_answer(0.9)],
        coder=[
            tool_call("write_file", path="calculator.py", content=fixed_calculator_source(scratch)),
            final("fixed the denominator"),
        ],
        tester=[suite_answer(True, "all green")],
        reviewer=[tool_call("get_diff"), review_answer(True)],
    )

    result = await run_chain(
        session,
        task,
        description="average() divides by len(values)-1 instead of len(values)",
        ctx=ctx,
        agents=agents,
    )

    assert result.final_state is TaskState.WAITING_FOR_APPROVAL, result.error
    assert result.pr_url, result.error
    assert result.pr_url.startswith("https://github.com/bivek127/amop-sandbox/pull/")
    print(f"\nOpened: {result.pr_url}\n")
