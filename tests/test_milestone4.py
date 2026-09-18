"""Milestone 4 — the full agent chain on a fixture repo.

Two tiers, per spec Sections 19.2 and 19.3:

  * Deterministic (always run): routing logic as pure functions, and full
    chain runs driven by a **scripted** model (19.2's "mock Model Router
    that returns scripted responses ... without spending API budget or
    introducing model non-determinism into CI"). These use a real
    Postgres, a real Docker sandbox and the real state machine -- only
    the model is scripted.
  * Real-model end-to-end (19.3), opt-in via AMOP_E2E_OLLAMA=1, because a
    four-agent chain on a local 14B model takes minutes and varies run to
    run. Asserted on outcome (final state, which file changed), never on
    reasoning text.

Requires: Postgres at TEST_DATABASE_URL, and a running Docker daemon.
"""

import json
import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text

from amop.agents.coder import CoderAgent
from amop.agents.handoffs import CONFIDENCE_THRESHOLD, ReviewVerdict, RootCauseReport
from amop.agents import handoffs  # TestReport accessed via the module: pytest collects any bare Test* name as a test class
from amop.agents.investigator import InvestigatorAgent
from amop.agents.reviewer import ReviewerAgent
from amop.agents import tester as tester_mod  # same reason: 'TesterAgent' matches pytest's Test* class pattern
from amop.database.session import init_db, make_engine, make_session_factory
from amop.models.base import BaseLLM, ModelResponse
from amop.orchestrator.chain import (
    MAX_FIX_ITERATIONS,
    MAX_NOOP_ATTEMPTS,
    enforce_review_checks,
    has_no_citations,
    missing_cited_files,
    MAX_REVIEW_CYCLES,
    ChainAgents,
    normalize_repo_path,
    out_of_scope_files,
    route_after_investigation,
    route_after_review,
    route_after_testing,
    run_chain,
)
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import create_task, get_transitions
from amop.sandbox import repo as git_repo
from amop.sandbox import tools as sandbox_tools  # noqa: F401 -- registers the sandboxed tools
from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext, ToolResult, get_tool, invoke_tool

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://localhost/amop_test"
)
FIXTURE_REPO = Path(__file__).parent / "fixtures" / "buggy_calculator"

# The known-good fix, per Section 19.3's "known-good fix so we can score
# pass/fail". Scripted Coders write this; the real-model E2E has to
# arrive at something equivalent on its own.
BUGGY_LINE = "return sum(values) / (len(values) - 1)"
FIXED_LINE = "return sum(values) / len(values)"


# ---------------------------------------------------------------------
# Pure routing tests — no LLM, no DB, no container (Done-When #3's
# foundation: the threshold is enforced in code, not by a prompt)
# ---------------------------------------------------------------------


def _report(confidence: float, affected=None) -> RootCauseReport:
    return RootCauseReport(
        task_id="t",
        root_cause="off-by-one in average()",
        confidence=confidence,
        evidence=[],
        affected_files=affected if affected is not None else ["calculator.py"],
        suggested_fix_plan="divide by len(values)",
    )


@pytest.mark.parametrize(
    "confidence,expected",
    [
        (0.0, TaskState.NEEDS_HUMAN_INPUT),
        (0.59, TaskState.NEEDS_HUMAN_INPUT),
        (0.599999, TaskState.NEEDS_HUMAN_INPUT),
        (0.6, TaskState.PLANNING_FIX),  # threshold is inclusive: ">= 0.6" proceeds
        (0.95, TaskState.PLANNING_FIX),
    ],
)
def test_confidence_threshold_boundary(confidence, expected):
    assert route_after_investigation(_report(confidence)) is expected


def test_confidence_threshold_matches_spec_value():
    # Guards against the threshold silently drifting away from Section
    # 6.2's stated 0.6 in a future refactor.
    assert CONFIDENCE_THRESHOLD == 0.6


@pytest.mark.parametrize(
    "all_passed,iterations,expected",
    [
        (True, 0, TaskState.REVIEWING),
        (True, MAX_FIX_ITERATIONS, TaskState.REVIEWING),  # passing wins over budget
        (False, 0, TaskState.CODING),
        (False, MAX_FIX_ITERATIONS - 1, TaskState.CODING),
        (False, MAX_FIX_ITERATIONS, TaskState.FAILED),
    ],
)
def test_route_after_testing(all_passed, iterations, expected):
    report = handoffs.TestReport(task_id="t", all_passed=all_passed, details="")
    assert route_after_testing(report, iterations) is expected


@pytest.mark.parametrize(
    "approved,cycles,expected",
    [
        (True, 0, TaskState.PR_CREATION),
        (False, 0, TaskState.CODING),
        (False, MAX_REVIEW_CYCLES - 1, TaskState.CODING),
        (False, MAX_REVIEW_CYCLES, TaskState.FAILED),
    ],
)
def test_route_after_review(approved, cycles, expected):
    verdict = ReviewVerdict(
        task_id="t", approved=approved, addresses_reported_symptom=True,
        findings=[], rejection_reason=None,
    )
    assert route_after_review(verdict, cycles) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("calculator.py", "calculator.py"),
        ("./calculator.py", "calculator.py"),
        ("/workspace/calculator.py", "calculator.py"),
        ("  calculator.py  ", "calculator.py"),
    ],
)
def test_normalize_repo_path(raw, expected):
    assert normalize_repo_path(raw) == expected


def test_out_of_scope_detects_test_file_edit():
    # The case this check exists for: a Coder that "fixes" the bug by
    # editing the test instead of the code under test.
    strayed = out_of_scope_files(
        ["calculator.py", "test_calculator.py"], ["calculator.py"]
    )
    assert strayed == ["test_calculator.py"]


def test_out_of_scope_tolerates_path_style_differences():
    assert out_of_scope_files(["calculator.py"], ["/workspace/calculator.py"]) == []


def test_out_of_scope_is_empty_when_nothing_was_declared():
    assert out_of_scope_files(["calculator.py"], []) == []


# ---------------------------------------------------------------------
# Scripted model + real DB + real sandbox
# ---------------------------------------------------------------------


class ScriptedLLM(BaseLLM):
    """Returns each response in order. Section 19.2's mock Model Router:
    validates the state machine without model non-determinism."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.prompts: list[list[dict]] = []

    async def complete(self, messages, tools=None) -> ModelResponse:
        self.prompts.append(list(messages))
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
            "task_id": "will-be-overwritten",
            "root_cause": "average() divides by len(values) - 1 instead of len(values)",
            "confidence": confidence,
            "evidence": [
                {"type": "code", "ref": "calculator.py:39", "excerpt": BUGGY_LINE}
            ],
            "affected_files": affected if affected is not None else ["calculator.py"],
            "suggested_fix_plan": "divide by len(values)",
        }
    )


def review_answer(
    approved: bool,
    reason: str | None = None,
    addresses_symptom: bool = True,
    counterexample: str | None = None,
    counterexample_claim: dict | None = None,
) -> str:
    return final(
        {
            "task_id": "t",
            "approved": approved,
            "addresses_reported_symptom": addresses_symptom,
            "findings": []
            if approved
            else [
                {
                    "severity": "high",
                    "description": "the fix changes the wrong denominator",
                    "file": "calculator.py",
                    "line": 39,
                }
            ],
            "rejection_reason": reason,
            "counterexample": counterexample,
            "counterexample_claim": counterexample_claim,
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


@pytest.fixture(autouse=True)
def _fake_github(monkeypatch):
    """Milestone 6: PR_CREATION now calls the real create_pull_request tool
    (amop/tools/github.py) instead of Milestone 4's simulated print. Left
    unmocked, every test in this file that drives the chain to a genuine
    fix would attempt a REAL network push + GitHub API call against
    github.com/bivek127/amop-sandbox -- GITHUB_TOKEN loads from .env via
    amop.database.session's import-time load_dotenv() call, so it's
    present even though nothing here opts into AMOP_E2E_GITHUB. This
    file's tests predate GitHub entirely and have no reason to touch it,
    so the registered tool's function is faked here: the chain still
    exercises real PR_CREATION routing (a genuine ToolResult flows back
    through invoke_tool, WAITING_FOR_APPROVAL is a real transition, not a
    skip) without ever making a network call. Autouse -- every test in
    this file gets this, since GitHub involvement is incidental to what
    all of them are actually testing.
    """

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
def workspace(tmp_path):
    """A materialized fixture repo in a real container, git-initialized
    and on a working branch — the same setup run_fix() performs, minus
    the chain itself."""
    scratch = tmp_path / "workspace"
    git_repo.materialize(FIXTURE_REPO, scratch)

    manager = SandboxManager()
    task_id = f"m4-{uuid.uuid4().hex[:8]}"
    sandbox = manager.create(task_id, scratch)
    try:
        git_repo.init_baseline(sandbox)
        git_repo.create_branch(sandbox, "amop/fix-test")
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


def build_agents(ctx, *, investigator, coder, tester, reviewer) -> ChainAgents:
    return ChainAgents(
        investigator=InvestigatorAgent(ScriptedLLM(investigator), ctx),
        coder=CoderAgent(ScriptedLLM(coder), ctx, task_id="scripted"),
        tester=tester_mod.TesterAgent(ScriptedLLM(tester), ctx),
        reviewer=ReviewerAgent(ScriptedLLM(reviewer), ctx),
    )


async def state_history(session, task_id) -> list[str]:
    return [t.to_state for t in await get_transitions(session, task_id)]


# -- the low-confidence guardrail, end to end -------------------------


async def test_low_confidence_halts_the_chain_at_needs_human_input(session, workspace):
    """Done-When #3. Not just 'the router returns the right enum' — the
    whole chain has to actually stop: no coding, no commit, terminal
    state in the database."""
    ctx, scratch, sandbox = workspace
    task = await create_task(session, task_type="bug_fix")

    agents = build_agents(
        ctx,
        investigator=[root_cause_answer(0.3)],
        coder=[final("should never run")],
        tester=[suite_answer(True)],
        reviewer=[review_answer(True)],
    )

    result = await run_chain(
        session, task, description="something is off", ctx=ctx, agents=agents
    )

    assert result.final_state is TaskState.NEEDS_HUMAN_INPUT
    assert result.task.state == TaskState.NEEDS_HUMAN_INPUT.value

    history = await state_history(session, task.id)
    assert history == [
        TaskState.TRIAGING.value,
        TaskState.INVESTIGATING.value,
        TaskState.NEEDS_HUMAN_INPUT.value,
    ]
    # The chain genuinely stopped rather than routing around the guard.
    assert TaskState.CODING.value not in history
    assert TaskState.PLANNING_FIX.value not in history

    # Nothing was written or committed.
    assert not git_repo.has_changes(sandbox)
    assert (scratch / "calculator.py").read_text().count(BUGGY_LINE) == 1


async def test_confidence_at_threshold_proceeds_to_coding(session, workspace):
    """The mirror of the test above: 0.6 exactly must NOT halt, or the
    guardrail would just be a chain that never runs."""
    ctx, scratch, _sandbox = workspace
    task = await create_task(session, task_type="bug_fix")

    agents = build_agents(
        ctx,
        investigator=[root_cause_answer(0.6)],
        coder=[
            tool_call("write_file", path="calculator.py", content=fixed_calculator_source(scratch)),
            final("fixed the denominator"),
        ],
        tester=[suite_answer(True)],
        reviewer=[review_answer(True)],
    )

    result = await run_chain(
        session, task, description="average is wrong", ctx=ctx, agents=agents
    )

    history = await state_history(session, task.id)
    assert TaskState.CODING.value in history
    # Milestone 6: a successful chain now ends at WAITING_FOR_APPROVAL (a
    # real PR opened, human must merge) rather than the old simulated
    # MERGED -> RESOLVED path.
    assert result.final_state is TaskState.WAITING_FOR_APPROVAL


# -- the happy path ---------------------------------------------------


async def test_full_chain_resolves_and_touches_only_the_buggy_file(session, workspace):
    ctx, scratch, sandbox = workspace
    task = await create_task(session, task_type="bug_fix")

    agents = build_agents(
        ctx,
        investigator=[root_cause_answer(0.9)],
        coder=[
            tool_call("run_tests"),
            tool_call("write_file", path="calculator.py", content=fixed_calculator_source(scratch)),
            final("changed the denominator to len(values)"),
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

    # Milestone 6: a successful chain now ends at WAITING_FOR_APPROVAL (a
    # real PR opened, human must merge) rather than the old simulated
    # MERGED -> RESOLVED path -- see the _fake_github fixture for why this
    # PR creation doesn't touch the real network.
    assert result.final_state is TaskState.WAITING_FOR_APPROVAL, result.error
    assert result.pr_url == "https://github.com/bivek127/amop-sandbox/pull/999"

    # The full legal path was walked, in order, with no shortcuts.
    assert await state_history(session, task.id) == [
        TaskState.TRIAGING.value,
        TaskState.INVESTIGATING.value,
        TaskState.PLANNING_FIX.value,
        TaskState.CODING.value,
        TaskState.TESTING.value,
        TaskState.REVIEWING.value,
        TaskState.PR_CREATION.value,
        TaskState.WAITING_FOR_APPROVAL.value,
    ]

    # Diff scope: only the buggy file, read from git rather than claimed.
    assert result.code_change_report.files_changed == ["calculator.py"]
    assert result.code_change_report.commit_sha
    assert FIXED_LINE in (scratch / "calculator.py").read_text()

    # And the fix is real: the suite passes in the container.
    ground_truth = await invoke_tool("run_tests", {}, ctx, agent_name="test")
    assert ground_truth.output["all_passed"]
    assert ground_truth.output["failed"] == 0


# -- ground truth beats self-report -----------------------------------


async def test_tester_claiming_success_cannot_override_a_failing_suite(session, workspace):
    """The sharpest form of "software decides what's allowed": the Tester
    model insists everything passes while the suite is genuinely red. The
    chain must believe pytest, not the model."""
    ctx, scratch, _sandbox = workspace
    task = await create_task(session, task_type="bug_fix")

    agents = build_agents(
        ctx,
        investigator=[root_cause_answer(0.9)],
        # Coder does nothing at all -- the bug is still present.
        coder=[final("I decided no change was needed")],
        # ...and the Tester lies about it, every time it's asked.
        tester=[suite_answer(True, "everything passes, ship it")],
        reviewer=[review_answer(True)],
    )

    result = await run_chain(
        session, task, description="average is wrong", ctx=ctx, agents=agents
    )

    # It must NOT reach RESOLVED on the strength of the model's claim.
    assert result.final_state is TaskState.FAILED
    assert result.test_report.all_passed is False

    history = await state_history(session, task.id)
    assert TaskState.REVIEWING.value not in history  # never got past testing
    # This Coder never makes a single mutating tool call, on any attempt --
    # that's the no_op guard's budget (MAX_NOOP_ATTEMPTS), not the
    # test-failure retry budget (MAX_FIX_ITERATIONS): a Coder that isn't
    # even trying shouldn't get the full fix-iteration budget spent on it.
    # Ground truth (pytest, not the lying Tester) still decided every one
    # of those attempts, which is the property this test exists to check.
    assert history.count(TaskState.CODING.value) == MAX_NOOP_ATTEMPTS


# -- reviewer rejection loops back ------------------------------------


async def test_reviewer_rejection_routes_back_to_coding_with_findings(session, workspace):
    ctx, scratch, _sandbox = workspace
    task = await create_task(session, task_type="bug_fix")

    coder_model = ScriptedLLM(
        [
            tool_call("write_file", path="calculator.py", content=fixed_calculator_source(scratch)),
            final("first attempt"),
            # The second attempt must make its own mutating tool call too --
            # a final_answer with no write_file/patch_file call is a no_op
            # (Milestone 5 bugfix: chain.py's no_op guard), and would no
            # longer be forwarded to Reviewer as if it were a fresh diff.
            tool_call("write_file", path="calculator.py", content=fixed_calculator_source(scratch)),
            final("addressed the review findings"),
        ]
    )
    agents = ChainAgents(
        investigator=InvestigatorAgent(ScriptedLLM([root_cause_answer(0.9)]), ctx),
        coder=CoderAgent(coder_model, ctx, task_id="scripted"),
        tester=tester_mod.TesterAgent(ScriptedLLM([suite_answer(True)]), ctx),
        reviewer=ReviewerAgent(
            ScriptedLLM(
                [
                    # A grounded rejection -- this test is about a
                    # rejection SURVIVING the mechanical override and
                    # getting addressed on retry, so the counterexample
                    # has to be one that actually holds up.
                    #
                    # Milestone 24: prose alone no longer does that. The
                    # structured claim below is executed against the real
                    # `average` in the sandbox, where calling it with []
                    # genuinely raises ZeroDivisionError -- so it
                    # verifies, and the rejection stands.
                    review_answer(
                        False,
                        "the fix needs a guard for empty input",
                        counterexample="average([]) with 0 items -> raises ZeroDivisionError, should raise ValueError instead",
                        counterexample_claim={
                            "module": "calculator",
                            "function": "Calculator.average",
                            "calls": [
                                {
                                    "name": "empty",
                                    "args": {"values": []},
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

    history = await state_history(session, task.id)
    # REVIEWING -> CODING happened, then the second review approved.
    assert history.count(TaskState.CODING.value) == 2
    assert history.count(TaskState.REVIEWING.value) == 2
    # Milestone 6: WAITING_FOR_APPROVAL, not RESOLVED -- see
    # test_confidence_at_threshold_proceeds_to_coding's comment.
    assert result.final_state is TaskState.WAITING_FOR_APPROVAL

    # Section 6.5: the findings reach the next Coder invocation verbatim.
    later_prompts = "".join(
        m["content"] for prompt in coder_model.prompts for m in prompt if m["role"] == "user"
    )
    assert "empty input" in later_prompts


async def test_scope_violation_overrides_reviewer_approval(session, workspace):
    """A Coder that edits the tests to make them pass, and a Reviewer that
    waves it through. The mechanical scope check has to catch it anyway --
    an approval is a judgment, but "which files changed" is a fact."""
    ctx, scratch, _sandbox = workspace
    task = await create_task(session, task_type="bug_fix")

    neutered_tests = (scratch / "test_calculator.py").read_text().replace(
        "assert calc.average([1, 2, 3]) == 2.0", "assert calc.average([1, 2, 3]) == 3.0"
    ).replace(
        "assert calc.average([7]) == 7.0", "assert True"
    )

    agents = build_agents(
        ctx,
        investigator=[root_cause_answer(0.9, affected=["calculator.py"])],
        coder=[
            tool_call("write_file", path="test_calculator.py", content=neutered_tests),
            final("adjusted the tests"),
        ],
        tester=[suite_answer(True)],
        reviewer=[review_answer(True)],  # reviewer approves the cheat
    )

    result = await run_chain(
        session, task, description="average is wrong", ctx=ctx, agents=agents
    )

    assert result.review_verdict.approved is False
    assert "scope violation" in result.review_verdict.rejection_reason
    assert "test_calculator.py" in result.review_verdict.rejection_reason
    assert result.final_state is not TaskState.RESOLVED


# -- handoff validation is a failure, not a pass-through --------------


async def test_invalid_handoff_fails_the_task(session, workspace):
    """Section 4.7: a handoff that fails schema validation is an agent
    failure. INVESTIGATING has no legal edge to FAILED, so the task lands
    on NEEDS_HUMAN_INPUT — the state machine, not a convenient default."""
    ctx, _scratch, _sandbox = workspace
    task = await create_task(session, task_type="bug_fix")

    agents = build_agents(
        ctx,
        # confidence is required and must be 0..1; 7.5 is neither.
        investigator=[final({"root_cause": "dunno", "confidence": 7.5})],
        coder=[final("never runs")],
        tester=[suite_answer(True)],
        reviewer=[review_answer(True)],
    )

    result = await run_chain(
        session, task, description="anything", ctx=ctx, agents=agents
    )

    assert result.final_state is TaskState.NEEDS_HUMAN_INPUT
    assert "schema validation" in (result.error or "")
    assert TaskState.CODING.value not in await state_history(session, task.id)


# -- the tool allowlist is code, not a prompt -------------------------


async def test_investigator_may_not_write_files(workspace):
    ctx, scratch, _sandbox = workspace
    agent = InvestigatorAgent(ScriptedLLM([""]), ctx)

    result = await invoke_tool(
        "write_file",
        {"path": "calculator.py", "content": "wiped"},
        ctx,
        agent_name=agent.name,
        allowed_tools=agent.tools,
    )

    assert not result.success
    assert result.error_code == "TOOL_NOT_PERMITTED"
    # And the file really wasn't touched.
    assert BUGGY_LINE in (scratch / "calculator.py").read_text()


async def test_reviewer_may_not_write_files(workspace):
    ctx, _scratch, _sandbox = workspace
    agent = ReviewerAgent(ScriptedLLM([""]), ctx)

    result = await invoke_tool(
        "write_file", {"path": "x.py", "content": "x"}, ctx,
        agent_name=agent.name, allowed_tools=agent.tools,
    )

    assert result.error_code == "TOOL_NOT_PERMITTED"


async def test_coder_may_write_files(workspace):
    ctx, scratch, _sandbox = workspace
    agent = CoderAgent(ScriptedLLM([""]), ctx, task_id="allowlist")

    result = await invoke_tool(
        "write_file", {"path": "scratch_note.txt", "content": "ok"}, ctx,
        agent_name=agent.name, allowed_tools=agent.tools,
    )

    assert result.success
    assert (scratch / "scratch_note.txt").read_text() == "ok"


# -- run_tests reports the truth --------------------------------------


async def test_run_tests_reports_the_seeded_failures(workspace):
    ctx, _scratch, _sandbox = workspace

    result = await invoke_tool("run_tests", {}, ctx, agent_name="test")

    assert result.success
    assert result.output["all_passed"] is False
    assert result.output["failed"] == 2
    assert result.output["passed"] == 8
    failing = {f["test_name"] for f in result.output["failures"]}
    assert failing == {"test_average_of_three_numbers", "test_average_of_single_value"}


async def test_run_tests_reports_green_after_the_known_good_fix(workspace):
    ctx, scratch, _sandbox = workspace
    (scratch / "calculator.py").write_text(fixed_calculator_source(scratch))

    result = await invoke_tool("run_tests", {}, ctx, agent_name="test")

    assert result.output["all_passed"] is True
    assert result.output["failed"] == 0
    assert result.output["passed"] == 10


async def test_run_tests_rejects_a_path_outside_the_workspace(workspace):
    ctx, _scratch, _sandbox = workspace

    result = await invoke_tool(
        "run_tests", {"path": "../../etc"}, ctx, agent_name="test"
    )

    assert not result.success
    assert result.error_code == "DENIED"


# ---------------------------------------------------------------------
# Real-model end-to-end (Section 19.3) — opt-in
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("AMOP_E2E_OLLAMA") != "1",
    reason="real-model E2E: set AMOP_E2E_OLLAMA=1 (needs Ollama running)",
)
async def test_real_model_chain_fixes_the_seeded_bug(session, tmp_path):
    from amop.models.ollama import OllamaProvider
    from amop.orchestrator.chain import run_fix

    task = await create_task(session, task_type="bug_fix")
    result = await run_fix(
        session,
        task,
        description=(
            "Calculator.average() returns the wrong value. "
            "average([1, 2, 3]) gives 3.0 instead of 2.0, and average([7]) "
            "raises ZeroDivisionError."
        ),
        repo_path=FIXTURE_REPO,
        model=OllamaProvider(),
        scratch_root=tmp_path,
        emit=print,
    )

    # Asserted on outcome, not on reasoning text (19.3). Milestone 6:
    # WAITING_FOR_APPROVAL, not RESOLVED -- this test is about whether
    # Ollama actually fixes the bug, not about GitHub, so create_pull_
    # request is faked here too (see _fake_github) rather than adding a
    # second live-network dependency to an already-slow real-model test.
    assert result.final_state is TaskState.WAITING_FOR_APPROVAL, result.error
    assert result.code_change_report.files_changed == ["calculator.py"]
    # Milestone 31: run_fix() now removes its scratch dir on return
    # (SandboxManager.destroy(..., remove_scratch_dir=True) -- nothing
    # in real production use needs it to survive), so this can no longer
    # re-read the file from disk afterward. result.diff is real git
    # ground truth (get_diff, not the model's self-report) and is still
    # available -- checked on the '+' line specifically, not the whole
    # diff blob, so this can't pass for the wrong reason (e.g. a diff
    # that showed the fix being removed instead of added).
    plus_lines = [
        line for line in result.diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    assert any(FIXED_LINE in line for line in plus_lines), result.diff


# ---------------------------------------------------------------------
# Container-path translation (regression: found by a live chain run, not
# by the scripted tests — the Coder was denied every file it touched
# because it used the /workspace paths its own prompt gave it)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["calculator.py", "./calculator.py", "/workspace/calculator.py"],
)
async def test_agent_can_read_the_repo_by_any_path_style(workspace, path):
    ctx, _scratch, _sandbox = workspace

    result = await invoke_tool("read_file", {"path": path}, ctx, agent_name="test")

    assert result.success, result.message
    assert BUGGY_LINE in result.output


async def test_agent_can_write_using_the_container_path(workspace):
    ctx, scratch, _sandbox = workspace

    result = await invoke_tool(
        "write_file",
        {"path": "/workspace/calculator.py", "content": fixed_calculator_source(scratch)},
        ctx,
        agent_name="test",
    )

    assert result.success, result.message
    assert FIXED_LINE in (scratch / "calculator.py").read_text()


@pytest.mark.parametrize(
    "path",
    [
        "/workspace/../etc/passwd",  # traversal out of the mount
        "/workspaceevil/secrets",  # lookalike prefix, must NOT be translated
        "/etc/passwd",
        "../../etc/passwd",
    ],
)
async def test_container_path_translation_still_denies_escapes(workspace, path):
    ctx, _scratch, _sandbox = workspace

    result = await invoke_tool("read_file", {"path": path}, ctx, agent_name="test")

    assert not result.success
    assert result.error_code == "DENIED"


# ---------------------------------------------------------------------
# Mechanical checks added after two live runs showed self-reported
# confidence is not a usable guardrail on its own: 1.00 while fixing an
# unrelated bug, and 0.80 while citing a file that does not exist.
# ---------------------------------------------------------------------


def test_missing_cited_files_flags_a_hallucinated_path(tmp_path):
    (tmp_path / "calculator.py").write_text("x = 1")
    report = _report(0.8, affected=["calculator.py", "src/export.py"])

    assert missing_cited_files(report, tmp_path) == ["src/export.py"]


def test_missing_cited_files_accepts_container_paths(tmp_path):
    (tmp_path / "calculator.py").write_text("x = 1")
    report = _report(0.8, affected=["/workspace/calculator.py"])

    assert missing_cited_files(report, tmp_path) == []


def test_missing_cited_files_flags_paths_outside_the_repo(tmp_path):
    report = _report(0.8, affected=["/etc/passwd"])

    # Exists on the host, but is not a file the Coder could ever touch.
    assert missing_cited_files(report, tmp_path) == ["/etc/passwd"]


def test_hallucinated_citation_routes_to_needs_human_input_despite_high_confidence():
    # This is the Run B case: confidence 0.80, well over the threshold.
    report = _report(0.8, affected=["src/export.py"])
    assert route_after_investigation(report, missing_files=["src/export.py"]) is (
        TaskState.NEEDS_HUMAN_INPUT
    )
    # ...and without the missing-file signal it would have proceeded,
    # which is exactly what happened live.
    assert route_after_investigation(report, missing_files=[]) is TaskState.PLANNING_FIX


def _verdict(
    approved: bool, addresses: bool = True, counterexample: str | None = None
) -> ReviewVerdict:
    return ReviewVerdict(
        task_id="t",
        approved=approved,
        addresses_reported_symptom=addresses,
        findings=[],
        rejection_reason="looks wrong to me" if not approved else None,
        counterexample=counterexample,
    )


def test_symptom_mismatch_overrides_an_approval():
    # Run A: correct code change, green suite, approving Reviewer -- but
    # it fixes a different problem than the one reported.
    checked = enforce_review_checks(
        _verdict(approved=True, addresses=False), ["calculator.py"], _report(1.0)
    )
    assert checked.approved is False
    assert "does not address the reported symptom" in checked.rejection_reason


def test_scope_violation_still_overrides_an_approval():
    checked = enforce_review_checks(
        _verdict(approved=True), ["calculator.py", "test_calculator.py"], _report(1.0)
    )
    assert checked.approved is False
    assert "scope violation" in checked.rejection_reason


def test_clean_approval_survives_the_mechanical_checks():
    checked = enforce_review_checks(_verdict(approved=True), ["calculator.py"], _report(1.0))
    assert checked.approved is True
    assert checked.rejection_reason is None


def test_downgraded_approval_is_never_re_upgraded_by_the_override():
    # A stricter Reviewer is always respected for the checks that produce
    # the downgrade themselves -- out_of_scope/addresses_reported_symptom
    # are hard facts about the repo, and the override (below) never runs
    # on their output, only on a rejection the model made unprompted.
    checked = enforce_review_checks(
        _verdict(approved=True, addresses=False), ["calculator.py"], _report(1.0),
        tests_passed=True,
    )
    assert checked.approved is False


# -- mechanical override of an ungrounded rejection --------------------
#
# Added after 3 live runs showed the Reviewer rejecting correct,
# in-scope, test-passing diffs with no identified defect -- 0/9
# approvals (docs-internal/ROADMAP.md). This is the one check that can
# turn approved=False into approved=True; every test below pins exactly
# how narrow that is.


def test_ungrounded_rejection_is_overridden_when_tests_pass_and_diff_in_scope():
    checked = enforce_review_checks(
        _verdict(approved=False),  # no counterexample
        ["calculator.py"],
        _report(1.0),
        tests_passed=True,
    )
    assert checked.approved is True
    assert "MECHANICALLY OVERRIDDEN" in checked.rejection_reason


def test_rejection_with_a_prose_only_counterexample_IS_now_overridden():
    """CONTRACT CHANGED IN MILESTONE 24 -- this test previously asserted
    the opposite, and the change is the point of that milestone.

    Prose used to keep a rejection alive if it merely had digits and
    enough length. Milestone 14 proved that gameable: a model supplied a
    well-shaped, arithmetically FALSE claim and its rejection of a
    correct diff stood. (This test's own former counterexample was an
    example -- it claims (9,8,1) scores 31.0 and (2,1,10) scores 8.0;
    under the fixture's formula those are 43.0 and 0.8.)

    Only an EXECUTED, verified claim keeps a rejection now. Prose is
    still carried for humans to read, but it no longer decides anything.
    """
    checked = enforce_review_checks(
        _verdict(
            approved=False,
            counterexample=(
                "urgency=9, impact=8, effort=1 -> score 31.0, but "
                "urgency=2, impact=1, effort=10 (score 8.0) should not "
                "outrank it and does"
            ),
        ),
        ["calculator.py"],
        _report(1.0),
        tests_passed=True,
    )
    assert checked.approved is True
    assert "no concrete input/output counterexample" in checked.rejection_reason


def test_rejection_is_not_overridden_when_tests_are_failing():
    checked = enforce_review_checks(
        _verdict(approved=False), ["calculator.py"], _report(1.0), tests_passed=False
    )
    assert checked.approved is False


def test_rejection_is_not_overridden_when_the_diff_is_out_of_scope():
    checked = enforce_review_checks(
        _verdict(approved=False),
        ["calculator.py", "test_calculator.py"],
        _report(1.0),  # affected_files=["calculator.py"] only
        tests_passed=True,
    )
    assert checked.approved is False


def test_rejection_is_not_overridden_when_nothing_was_declared_affected():
    # No affected_files means there's nothing to confirm the diff stayed
    # in scope against -- refuse to guess, same call the pre-existing
    # out_of_scope_files() makes for the downgrade direction.
    checked = enforce_review_checks(
        _verdict(approved=False), ["calculator.py"], _report(1.0, affected=[]),
        tests_passed=True,
    )
    assert checked.approved is False


@pytest.mark.parametrize(
    "text,expected",
    [
        (None, False),
        ("", False),
        ("looks wrong to me", False),  # no digits -- not a real input/output pair
        ("this diff is bad", False),
        ("urgency=9, impact=8, effort=1 -> score 31.0, too high", True),
    ],
)
def test_has_concrete_counterexample_is_a_format_check_not_a_correctness_check(
    text, expected
):
    from amop.orchestrator.chain import _has_concrete_counterexample

    assert _has_concrete_counterexample(_verdict(approved=False, counterexample=text)) is (
        expected
    )


def test_review_verdict_requires_an_explicit_symptom_answer():
    # No default: a model that ignores the question fails validation and
    # gets asked again, rather than passing on silence.
    with pytest.raises(Exception):
        ReviewVerdict(task_id="t", approved=True, findings=[], rejection_reason=None)


async def test_chain_halts_when_the_investigator_cites_a_nonexistent_file(session, workspace):
    """End-to-end version of Run B: high confidence, imaginary file. The
    chain must stop at INVESTIGATING instead of dispatching the Coder to
    edit something that isn't there."""
    ctx, _scratch, sandbox = workspace
    task = await create_task(session, task_type="bug_fix")

    agents = build_agents(
        ctx,
        investigator=[root_cause_answer(0.8, affected=["src/export.py"])],
        coder=[final("should never run")],
        tester=[suite_answer(True)],
        reviewer=[review_answer(True)],
    )

    result = await run_chain(
        session, task, description="CSV export drops rows", ctx=ctx, agents=agents
    )

    assert result.final_state is TaskState.NEEDS_HUMAN_INPUT
    history = await state_history(session, task.id)
    assert TaskState.CODING.value not in history
    assert not git_repo.has_changes(sandbox)


async def test_chain_rejects_a_fix_that_misses_the_reported_symptom(session, workspace):
    """End-to-end version of Run A: the Coder makes a real, correct,
    in-scope change and the suite goes green -- but the Reviewer says it
    doesn't address what was reported. The task must NOT reach RESOLVED."""
    ctx, scratch, _sandbox = workspace
    task = await create_task(session, task_type="bug_fix")

    agents = build_agents(
        ctx,
        investigator=[root_cause_answer(1.0)],
        coder=[
            tool_call("write_file", path="calculator.py", content=fixed_calculator_source(scratch)),
            final("fixed the denominator"),
        ],
        tester=[suite_answer(True)],
        reviewer=[review_answer(True, addresses_symptom=False)],
    )

    result = await run_chain(
        session,
        task,
        description="The CSV export sometimes drops the last row.",
        ctx=ctx,
        agents=agents,
    )

    assert result.final_state is not TaskState.RESOLVED
    assert result.final_state is TaskState.FAILED
    assert result.review_verdict.approved is False
    assert "does not address the reported symptom" in result.review_verdict.rejection_reason


# -- the degenerate citation case (observed in Run B) ------------------


@pytest.mark.parametrize("affected", [[], ["  "], ["", ""]])
def test_has_no_citations_detects_an_empty_file_list(affected):
    assert has_no_citations(_report(0.8, affected=affected)) is True


def test_has_no_citations_is_false_when_a_file_is_named():
    assert has_no_citations(_report(0.8, affected=["calculator.py"])) is False


def test_uncited_root_cause_routes_to_needs_human_input_despite_high_confidence():
    # Run B's actual shape: confidence 0.80, affected_files [] -- which
    # slipped past BOTH mechanical checks before this guard existed.
    report = _report(0.8, affected=[])
    assert route_after_investigation(report, missing_files=[], no_citations=True) is (
        TaskState.NEEDS_HUMAN_INPUT
    )


async def test_chain_halts_when_the_investigator_names_no_files(session, workspace):
    ctx, _scratch, sandbox = workspace
    task = await create_task(session, task_type="bug_fix")

    agents = build_agents(
        ctx,
        investigator=[root_cause_answer(0.8, affected=[])],
        coder=[final("should never run")],
        tester=[suite_answer(True)],
        reviewer=[review_answer(True)],
    )

    result = await run_chain(
        session, task, description="CSV export drops rows", ctx=ctx, agents=agents
    )

    assert result.final_state is TaskState.NEEDS_HUMAN_INPUT
    history = await state_history(session, task.id)
    assert TaskState.CODING.value not in history
    assert not git_repo.has_changes(sandbox)
