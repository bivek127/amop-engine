"""The real agent chain — Investigator → Coder → Tester → Reviewer,
driven by Milestone 1's state machine.

Section 4.7: agents never call each other. Everything moves through this
orchestrator as a validated `handoff_payload`, and a handoff that fails
schema validation is an agent failure, not a silent pass-through.

Three things here are deliberately *not* decided by a model:

  1. Routing. route_after_investigation/testing/review are pure
     functions of a handoff plus a counter. They contain no LLM call and
     are unit-testable in isolation, the same discipline as
     safety/engine.py's evaluate().
  2. Test outcomes. The orchestrator runs the suite itself and overwrites
     TestReport.all_passed with the parsed pytest result. A model may
     describe the failures; it may not decide whether there were any.
  3. Scope. The Reviewer's approval is re-checked against `git diff
     --name-only`: a diff touching files the root cause never named is
     rejected regardless of what the Reviewer concluded (Section 6.3.9 --
     Coder's self-report is not trusted, and neither is an approval that
     contradicts the repo).

This is "LLMs decide what to do; software decides what's allowed" at the
orchestration layer: the models supply judgment, the code supplies the
consequences.
"""

import asyncio
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from amop.agents.coder import CoderAgent
from amop.agents.handoffs import (
    CONFIDENCE_THRESHOLD,
    CodeChangeReport,
    ReviewVerdict,
    RootCauseReport,
    TestReport,
)
from amop.agents.investigator import InvestigatorAgent
from amop.agents.reviewer import ReviewerAgent
from amop.agents.tester import TesterAgent
from amop.database.models import Task
from amop.orchestrator.state_machine import TRANSITIONS, TaskState
from amop.orchestrator.task import transition
from amop.safety.engine import resolve_within_scratch
from amop.sandbox import repo as git_repo
from amop.sandbox import tools as sandbox_tools  # noqa: F401 -- registers the sandboxed tools
from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext, invoke_tool

# Section 4.2's counters: "test failure AND retry_count < max_fix_iterations
# (default 4)" and "Reviewer rejects with actionable feedback AND
# review_cycles < 2".
MAX_FIX_ITERATIONS = 4
MAX_REVIEW_CYCLES = 2


# ---------------------------------------------------------------------
# Pure routing — no LLM, no I/O, no database. Unit-tested directly.
# ---------------------------------------------------------------------


def route_after_investigation(
    report: RootCauseReport,
    missing_files: Sequence[str] = (),
    no_citations: bool = False,
) -> TaskState:
    """Section 6.2's confidence guardrail, and Section 4.2's
    INVESTIGATING transitions. Below the threshold the task stops and
    asks a human rather than handing a guess to the Coder.

    `missing_files` are cited affected_files that don't exist in the
    repo. That check exists because the confidence number alone proved
    unreliable in practice: on a bug report with nothing matching it in
    the repo, the model reported 0.80 confidence while citing
    "/workspace/src/export.py", a file that does not exist. Confidence
    is self-reported and a model can be sure of something false;
    "does this path exist" is a fact about the repo and cannot be
    talked around. Section 6.2 calls a gap like this an evidence gap,
    which lands the task at NEEDS_HUMAN_INPUT.
    """
    if missing_files or no_citations:
        return TaskState.NEEDS_HUMAN_INPUT
    if report.confidence < CONFIDENCE_THRESHOLD:
        return TaskState.NEEDS_HUMAN_INPUT
    return TaskState.PLANNING_FIX


def route_after_testing(report: TestReport, fix_iterations: int) -> TaskState:
    if report.all_passed:
        return TaskState.REVIEWING
    if fix_iterations < MAX_FIX_ITERATIONS:
        return TaskState.CODING
    return TaskState.FAILED


def route_after_review(verdict: ReviewVerdict, review_cycles: int) -> TaskState:
    if verdict.approved:
        return TaskState.PR_CREATION
    if review_cycles < MAX_REVIEW_CYCLES:
        return TaskState.CODING
    return TaskState.FAILED


def normalize_repo_path(path: str) -> str:
    """Reduce a model-supplied or git-supplied path to a comparable form.

    Models refer to the same file as "calculator.py", "/workspace/
    calculator.py", or "./calculator.py" interchangeably; git always
    reports repo-relative. Normalizing before comparison keeps the scope
    check from firing on a naming style rather than a real out-of-scope
    edit.
    """
    cleaned = path.strip().removeprefix("/workspace/").removeprefix("./")
    return cleaned.strip("/")


def out_of_scope_files(changed: list[str], affected: list[str]) -> list[str]:
    """Files the diff touched that the root cause never named.

    Returns [] when `affected` is empty -- with nothing declared there is
    nothing to check against, and inventing a violation from an absent
    claim would just be noise. Catching a Coder that edited the *tests*
    to make them pass is the main thing this earns.
    """
    if not affected:
        return []
    declared = {normalize_repo_path(p) for p in affected}
    return sorted(
        {normalize_repo_path(c) for c in changed if normalize_repo_path(c) not in declared}
    )


def has_no_citations(report: RootCauseReport) -> bool:
    """True if the Investigator named no affected file at all.

    This is the degenerate case of missing_cited_files() and it is worth
    its own guard, because an empty list silently defeats *two* checks:
    nothing cited means nothing can be missing, and out_of_scope_files()
    treats "nothing declared" as "nothing out of scope". Observed live --
    asked about a CSV export the repo doesn't have, the Investigator
    returned confidence 0.80 with affected_files: [], sailed through both
    guards, and burned three coding cycles producing empty diffs.

    A root cause that can't name a single file is an evidence gap by
    Section 6.2's own framing, not a fix plan.
    """
    return not [p for p in report.affected_files if p.strip()]


def missing_cited_files(report: RootCauseReport, scratch_dir: Path) -> list[str]:
    """Cited affected_files that don't exist in the repo.

    Purely mechanical: the model claims a file is implicated, and either
    that file is on disk or it isn't. Catches a hallucinated citation
    before the Coder is dispatched to edit something imaginary.
    """
    missing = []
    for cited in report.affected_files:
        resolved = resolve_within_scratch(cited, scratch_dir)
        # An out-of-scope path resolves to None -- also not a file the
        # Coder could legitimately touch, so it counts as missing here
        # and the Safety Engine would deny it downstream anyway.
        if resolved is None or not resolved.exists():
            missing.append(cited)
    return missing


def enforce_review_checks(
    verdict: ReviewVerdict, changed_files: list[str], report: RootCauseReport
) -> ReviewVerdict:
    """Apply the mechanical checks that sit on top of the Reviewer's
    judgment. An approval is a model opinion; these are not.

    Only ever downgrades an approval -- it can never turn a rejection
    into an approval, so a stricter Reviewer is always respected.
    """
    if not verdict.approved:
        return verdict

    strayed = out_of_scope_files(changed_files, report.affected_files)
    if strayed:
        return verdict.model_copy(
            update={
                "approved": False,
                "rejection_reason": (
                    f"scope violation: the diff touches {strayed}, which the "
                    f"root cause did not list as affected "
                    f"({report.affected_files}). Fix the cause, do not modify "
                    "unrelated files or tests."
                ),
            }
        )

    if not verdict.addresses_reported_symptom:
        return verdict.model_copy(
            update={
                "approved": False,
                "rejection_reason": (
                    "the change does not address the reported symptom. The "
                    "code may be correct and the tests may pass, but this "
                    "fixes a different problem than the one reported "
                    f"(claimed root cause: {report.root_cause[:120]})."
                ),
            }
        )

    return verdict


# ---------------------------------------------------------------------
# Chain wiring
# ---------------------------------------------------------------------


@dataclass
class ChainAgents:
    """The four agents, injectable so tests can drive the real chain with
    scripted models instead of a live Ollama (spec 19.2)."""

    investigator: Any
    coder: Any
    tester: Any
    reviewer: Any

    @classmethod
    def build(cls, model, ctx: ToolContext, task_id: str) -> "ChainAgents":
        return cls(
            investigator=InvestigatorAgent(model, ctx),
            coder=CoderAgent(model, ctx, task_id=task_id),
            tester=TesterAgent(model, ctx),
            reviewer=ReviewerAgent(model, ctx),
        )


@dataclass
class ChainResult:
    task: Task
    final_state: TaskState
    root_cause_report: RootCauseReport | None = None
    code_change_report: CodeChangeReport | None = None
    test_report: TestReport | None = None
    review_verdict: ReviewVerdict | None = None
    diff: str = ""
    container_id: str | None = None
    error: str | None = None
    stages: list[str] = field(default_factory=list)


def _noop(_message: str) -> None:
    pass


class _ChainFailure(Exception):
    """Internal: an agent failed or a handoff didn't validate. Carries the
    reason to the single place that records the terminal state."""


# Preference order for where a mid-chain failure lands: FAILED says the
# most, NEEDS_HUMAN_INPUT is the soft-terminal fallback (Section 4.1),
# CANCELLED is the last resort. The first one that's a legal edge from
# the current state wins -- read out of the real transition table rather
# than hardcoded per state, so it can't drift from the state machine.
_FAILURE_PREFERENCE = (
    TaskState.FAILED,
    TaskState.NEEDS_HUMAN_INPUT,
    TaskState.CANCELLED,
)


def _failure_state(current: TaskState) -> TaskState:
    for candidate in _FAILURE_PREFERENCE:
        if (current, candidate) in TRANSITIONS:
            return candidate
    return TaskState.CANCELLED


async def run_chain(
    session: AsyncSession,
    task: Task,
    *,
    description: str,
    ctx: ToolContext,
    agents: ChainAgents,
    emit: Callable[[str], None] = _noop,
) -> ChainResult:
    """Drive `task` through the real state machine using `agents`.

    Assumes a live sandbox on `ctx.sandbox` with the repo already
    materialized, git-initialized and on a working branch — see
    run_fix() for the lifecycle around this.
    """
    result = ChainResult(task=task, final_state=TaskState(task.state))
    result.container_id = ctx.sandbox.short_id if ctx.sandbox else None

    async def go(to_state: TaskState, actor: str, trigger: str | None = None) -> None:
        nonlocal task
        task = await transition(session, task, to_state, trigger=trigger, actor=actor)
        result.task = task
        result.final_state = to_state

    def stage(message: str) -> None:
        result.stages.append(message)
        emit(message)

    try:
        await go(TaskState.TRIAGING, actor="system")
        stage("TRIAGING: bug report accepted, treating as novel")
        await go(TaskState.INVESTIGATING, actor="system")

        # -- INVESTIGATING -------------------------------------------
        stage("INVESTIGATING: investigator examining the repo")
        report = await _run_agent(agents.investigator, _investigator_prompt(description))
        report = _retag(report, str(task.id))
        result.root_cause_report = report
        stage(
            f"INVESTIGATING: root cause -> {report.root_cause[:120]} "
            f"(confidence {report.confidence:.2f}, "
            f"affected {report.affected_files or '[]'})"
        )

        missing = missing_cited_files(report, ctx.scratch_dir)
        uncited = has_no_citations(report)
        if (
            route_after_investigation(report, missing, uncited)
            is TaskState.NEEDS_HUMAN_INPUT
        ):
            if uncited:
                reason = (
                    "investigator named no affected files "
                    "(Section 6.2 evidence gap)"
                )
                detail = (
                    "NEEDS_HUMAN_INPUT: the investigator could not name a single "
                    "affected file — there is nothing for the coder to act on"
                )
            elif missing:
                reason = (
                    f"cited files do not exist in the repo: {missing} "
                    "(Section 6.2 evidence gap)"
                )
                detail = (
                    f"NEEDS_HUMAN_INPUT: the investigator cited {missing}, which "
                    "does not exist in this repo — refusing to dispatch a fix "
                    "for an imaginary file"
                )
            else:
                reason = (
                    f"confidence {report.confidence:.2f} < {CONFIDENCE_THRESHOLD} "
                    "(Section 6.2 guardrail)"
                )
                detail = (
                    f"NEEDS_HUMAN_INPUT: confidence {report.confidence:.2f} is below "
                    f"{CONFIDENCE_THRESHOLD} — stopping rather than guessing at a fix"
                )
            await go(
                TaskState.NEEDS_HUMAN_INPUT,
                actor=f"agent:{agents.investigator.name}",
                trigger=reason,
            )
            stage(detail)
            return result

        await go(TaskState.PLANNING_FIX, actor=f"agent:{agents.investigator.name}")
        stage("PLANNING_FIX: handing the root cause to the coder")

        fix_iterations = 0
        review_cycles = 0
        findings_feedback = ""
        # Who/what caused the *next* entry into CODING. Seeded with the
        # PLANNING_FIX -> CODING hop; retry branches below rewrite it
        # before looping, so every re-entry is attributed to the agent
        # that actually sent the task back.
        coding_actor = f"agent:{agents.investigator.name}"
        coding_trigger: str | None = None

        while True:
            # -- CODING ----------------------------------------------
            # The only place the chain enters CODING, so PLANNING_FIX ->
            # CODING, TESTING -> CODING and REVIEWING -> CODING all go
            # through one legal transition instead of a branch trying to
            # re-enter a state it's already in.
            await go(TaskState.CODING, actor=coding_actor, trigger=coding_trigger)
            stage(f"CODING: coder applying fix (attempt {fix_iterations + 1})")
            code_report = await _run_coder(
                agents.coder, ctx, report, findings_feedback, str(task.id)
            )
            result.code_change_report = code_report
            stage(
                f"CODING: {code_report.status}, files changed: "
                f"{code_report.files_changed or '[]'}"
            )

            # -- TESTING ---------------------------------------------
            await go(TaskState.TESTING, actor=f"agent:{agents.coder.name}")
            stage("TESTING: running the repo's own test suite")
            test_report = await _run_tester(agents.tester, ctx, str(task.id))
            result.test_report = test_report
            stage(f"TESTING: all_passed={test_report.all_passed} — {test_report.details[:160]}")

            next_state = route_after_testing(test_report, fix_iterations)
            if next_state is TaskState.CODING:
                fix_iterations += 1
                findings_feedback = (
                    "The test suite still fails after your last change:\n"
                    f"{test_report.details}"
                )
                coding_actor = f"agent:{agents.tester.name}"
                coding_trigger = (
                    f"test failure AND retry_count {fix_iterations} "
                    f"< {MAX_FIX_ITERATIONS}"
                )
                continue
            if next_state is TaskState.FAILED:
                await go(
                    TaskState.FAILED,
                    actor=f"agent:{agents.tester.name}",
                    trigger="test failure AND retries exhausted",
                )
                stage("FAILED: tests still failing after the retry budget was exhausted")
                return result

            # -- REVIEWING -------------------------------------------
            await go(TaskState.REVIEWING, actor=f"agent:{agents.tester.name}")
            stage("REVIEWING: reviewer checking the diff against the root cause")
            verdict = await _run_reviewer(
                agents.reviewer, ctx, report, test_report, description, str(task.id)
            )

            # Mechanical checks on top of the model's verdict.
            checked = enforce_review_checks(verdict, code_report.files_changed, report)
            if checked.approved != verdict.approved:
                stage(f"REVIEWING: approval overridden — {checked.rejection_reason}")
            verdict = checked
            result.review_verdict = verdict
            stage(
                f"REVIEWING: approved={verdict.approved}"
                + (f" — {verdict.rejection_reason}" if verdict.rejection_reason else "")
            )

            next_state = route_after_review(verdict, review_cycles)
            if next_state is TaskState.CODING:
                review_cycles += 1
                # Section 6.5: findings are fed into the next Coder
                # invocation verbatim -- Coder addresses them, it does
                # not re-investigate.
                findings_feedback = _render_findings(verdict)
                coding_actor = f"agent:{agents.reviewer.name}"
                coding_trigger = (
                    f"Reviewer rejects with actionable feedback AND "
                    f"review_cycles {review_cycles} < {MAX_REVIEW_CYCLES}"
                )
                continue
            if next_state is TaskState.FAILED:
                await go(
                    TaskState.FAILED,
                    actor=f"agent:{agents.reviewer.name}",
                    trigger="Reviewer rejects and cycles exhausted",
                )
                stage("FAILED: reviewer rejected and the review budget was exhausted")
                return result
            break

        # -- PR_CREATION (simulated) ---------------------------------
        await go(TaskState.PR_CREATION, actor=f"agent:{agents.reviewer.name}")
        result.diff = await _get_diff(ctx)
        stage("PR_CREATION: (simulated — no GitHub API call this milestone)")

        # Section 4.2: PR_CREATION -> MERGED requires "mode >= operator
        # and auto_merge". The doc's "PR_CREATION -> RESOLVED" is not a
        # legal edge, so the chain takes the real path through MERGED.
        await go(
            TaskState.MERGED,
            actor="system",
            trigger="mode >= operator and auto_merge:true (simulated merge, no remote)",
        )
        stage("MERGED: fix committed on the working branch (merge simulated locally)")

        # MERGED -> RESOLVED's trigger is "post-merge checks pass", so
        # the suite is genuinely re-run here rather than assumed green.
        post_merge = await _run_tests_ground_truth(ctx)
        if not post_merge["all_passed"]:
            await go(
                TaskState.CANCELLED,
                actor="system",
                trigger="post-merge checks failed",
            )
            stage("CANCELLED: post-merge test run did not pass")
            return result

        await go(TaskState.RESOLVED, actor="system")
        stage(
            f"RESOLVED: post-merge suite green "
            f"({post_merge['passed']} passed, {post_merge['failed']} failed)"
        )
        return result

    except _ChainFailure as exc:
        # Section 5.3: an agent invocation can terminate in six ways, and
        # "the orchestrator decides the resulting task state per Section
        # 4.2's transition table (most map to NEEDS_HUMAN_INPUT or
        # FAILED)". Which one is legal depends on where the task is --
        # e.g. INVESTIGATING has no edge to FAILED at all, so an
        # Investigator that dies mid-run parks at NEEDS_HUMAN_INPUT.
        failure_state = _failure_state(TaskState(task.state))
        await go(failure_state, actor="system", trigger=str(exc))
        result.error = str(exc)
        stage(f"{failure_state.value}: {exc}")
        return result


# ---------------------------------------------------------------------
# Agent invocation helpers
# ---------------------------------------------------------------------


async def _run_agent(agent, prompt: str):
    """Run an agent and return its validated handoff, or raise
    _ChainFailure. Section 4.7: a handoff that fails validation is an
    agent failure, never a silent pass-through."""
    result = await agent.run(prompt)
    if not result.success:
        raise _ChainFailure(f"{agent.name} failed: {result.error}")
    if result.handoff is None:
        raise _ChainFailure(f"{agent.name} produced no valid handoff payload")
    return result.handoff


def _retag(handoff, task_id: str):
    """Force the handoff's task_id to the real task. The model is asked
    for it but has no way to know it -- and a mismatched id in an audit
    trail is worse than one the orchestrator stamps itself."""
    return handoff.model_copy(update={"task_id": task_id})


def _investigator_prompt(description: str) -> str:
    return (
        f"Bug report: {description}\n\n"
        "The repository is checked out at /workspace. Investigate and "
        "report the root cause."
    )


async def _run_coder(agent, ctx, report: RootCauseReport, feedback: str, task_id: str):
    prompt = (
        f"Root cause: {report.root_cause}\n"
        f"Suggested fix plan: {report.suggested_fix_plan}\n"
        f"Affected files: {report.affected_files}\n\n"
        "Apply the smallest change that fixes this root cause. Fix the "
        "source code, not the tests. The repository is at /workspace."
    )
    if feedback:
        prompt += f"\n\nFeedback you must address:\n{feedback}"

    agent_result = await agent.run(prompt)

    # A failed Coder is deliberately NOT a chain failure. Section 6.3.3:
    # "Coder hands off a CodeChangeReport with status: 'failed' and its
    # best diagnostic -- the orchestrator, not the Coder, decides whether
    # that's a task failure (Section 4.2: TESTING -> FAILED)." So the run
    # continues into TESTING, where the real suite gets the final word;
    # the retry budget handles it from there.
    diagnostic = None if agent_result.success else f"coder agent failed: {agent_result.error}"

    # CodeChangeReport is assembled from git, not from what the model
    # said it did (Section 6.3.9). Docker calls are blocking, so they go
    # off the event loop.
    sandbox = ctx.sandbox
    branch = await asyncio.to_thread(git_repo.current_branch, sandbox)
    sha = await asyncio.to_thread(
        git_repo.commit_all, sandbox, f"fix: {report.root_cause[:60]}"
    )
    changed = await asyncio.to_thread(git_repo.changed_files, sandbox)
    full_diff = await asyncio.to_thread(git_repo.diff_against_baseline, sandbox)

    if not changed and diagnostic is None:
        diagnostic = "coder made no file changes"

    return CodeChangeReport(
        task_id=task_id,
        status="success" if changed and agent_result.success else "failed",
        branch=branch,
        commit_sha=sha,
        files_changed=changed,
        diff_summary=full_diff[:2000],
        iterations_used=agent_result.iterations_used,
        failure_diagnostic=diagnostic,
    )


async def _run_tester(agent, ctx, task_id: str) -> TestReport:
    """Run the Tester, then overwrite its all_passed with ground truth.

    The model contributes interpretation; pytest contributes the verdict.
    A Tester that claims success while the suite is red cannot move the
    task forward, because the value the router reads never came from it.
    """
    ground_truth = await _run_tests_ground_truth(ctx)

    prompt = (
        "A fix has just been applied to the repository at /workspace. "
        "Run the test suite and summarize the current state of it."
    )
    agent_result = await agent.run(prompt)

    details = ""
    if agent_result.success and agent_result.handoff is not None:
        details = agent_result.handoff.details
    else:
        # A failed Tester agent is not a failed task: the authoritative
        # result already exists. Record the degradation instead of
        # discarding a perfectly good pytest run.
        details = f"(tester agent unavailable: {agent_result.error})"

    if ground_truth["failures"]:
        failed_names = ", ".join(f["test_name"] for f in ground_truth["failures"])
        details = f"{details} | failing: {failed_names}"

    return TestReport(
        task_id=task_id,
        all_passed=ground_truth["all_passed"],
        details=details.strip(),
    )


async def _run_reviewer(
    agent,
    ctx,
    report: RootCauseReport,
    test_report: TestReport,
    description: str,
    task_id: str,
) -> ReviewVerdict:
    # The original bug report leads the prompt (Section 6.4: check the fix
    # "actually addresses reported behavior, not just 'tests pass'"). Without
    # it the Reviewer can only check the diff against the *stated* root
    # cause, which is internally consistent even when the root cause has
    # nothing to do with what was reported.
    prompt = (
        f"ORIGINAL BUG REPORT (what the user actually reported):\n"
        f"{description}\n\n"
        f"Stated root cause: {report.root_cause}\n"
        f"Files the root cause identified: {report.affected_files}\n"
        f"Test suite currently passing: {test_report.all_passed}\n"
        f"Test details: {test_report.details}\n\n"
        "Fetch the diff with get_diff and review the change. Judge it "
        "against the ORIGINAL BUG REPORT above, not only against the "
        "stated root cause."
    )
    verdict = await _run_agent(agent, prompt)
    return _retag(verdict, task_id)


def _render_findings(verdict: ReviewVerdict) -> str:
    lines = [f"The reviewer rejected your change: {verdict.rejection_reason or ''}"]
    for f in verdict.findings:
        location = f" ({f.file}:{f.line})" if f.file else ""
        lines.append(f"- [{f.severity}]{location} {f.description}")
    return "\n".join(lines)


async def _run_tests_ground_truth(ctx) -> dict:
    """The authoritative test result: the orchestrator's own run_tests
    call, parsed from pytest's junit XML. Never a model's claim."""
    result = await invoke_tool("run_tests", {}, ctx, agent_name="orchestrator")
    if not result.success:
        return {
            "all_passed": False,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "failures": [],
            "error": result.message,
        }
    return result.output


async def _get_diff(ctx) -> str:
    result = await invoke_tool("get_diff", {}, ctx, agent_name="orchestrator")
    return result.output if result.success else ""


# ---------------------------------------------------------------------
# Full lifecycle entry point
# ---------------------------------------------------------------------


async def run_fix(
    session: AsyncSession,
    task: Task,
    *,
    description: str,
    repo_path: Path,
    model,
    scratch_root: Path | None = None,
    mode: str = "operator",
    emit: Callable[[str], None] = _noop,
) -> ChainResult:
    """Materialize the repo, stand up one container for the task, run the
    chain, then tear the container down.

    Section 9.1: one container per *task*, shared by every agent — the
    branch Coder creates has to still be there when Tester and Reviewer
    look at it.
    """
    scratch_root = Path(scratch_root or sandbox_tools.SCRATCH_DIR)
    scratch_dir = (scratch_root / str(task.id)).resolve()

    await asyncio.to_thread(git_repo.materialize, Path(repo_path), scratch_dir)

    manager = await asyncio.to_thread(SandboxManager)
    task_id = str(task.id)
    sandbox = await asyncio.to_thread(manager.create, task_id, scratch_dir)
    try:
        await asyncio.to_thread(git_repo.init_baseline, sandbox)
        await asyncio.to_thread(
            git_repo.create_branch, sandbox, f"amop/fix-{uuid.UUID(task_id).hex[:8]}"
        )

        ctx = ToolContext(
            agent_name="chain",
            scratch_dir=scratch_dir,
            mode=mode,
            sandbox=sandbox,
        )
        agents = ChainAgents.build(model, ctx, task_id)
        emit(f"Sandbox container: {sandbox.short_id}")
        emit(f"Workspace: {scratch_dir}")
        return await run_chain(
            session, task, description=description, ctx=ctx, agents=agents, emit=emit
        )
    finally:
        await asyncio.to_thread(manager.destroy, task_id)
