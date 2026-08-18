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

from amop.agents import tester as tester_mod
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
from amop.codebase_intel.indexer import index_repo
from amop.database.models import Task
from amop.memory import store as memory_store
from amop.orchestrator.state_machine import TERMINAL_STATES, TRANSITIONS, TaskState
from amop.orchestrator.task import transition
from amop.safety import scope_guard
from amop.safety.engine import resolve_within_scratch
from amop.sandbox import repo as git_repo
from amop.sandbox import tools as sandbox_tools  # noqa: F401 -- registers the sandboxed tools
from amop.sandbox.manager import SandboxManager
from amop.tools import github as github_tools  # noqa: F401 -- registers create_pull_request/get_ci_status
from amop.tools.registry import ToolContext, ToolResult, get_tool, invoke_tool

# Section 4.2's counters: "test failure AND retry_count < max_fix_iterations
# (default 4)" and "Reviewer rejects with actionable feedback AND
# review_cycles < 2".
MAX_FIX_ITERATIONS = 4
MAX_REVIEW_CYCLES = 2

# Not a spec number -- a bugfix budget. A coding attempt that makes zero
# mutating tool calls (no write_file/patch_file) produced no new diff at
# all; retrying it costs nothing like a real test/review failure does, so
# it gets its own small budget instead of eating into MAX_FIX_ITERATIONS or
# MAX_REVIEW_CYCLES. Still bounded, so a coder that never edits anything
# can't loop forever.
MAX_NOOP_ATTEMPTS = 2


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


_MIN_COUNTEREXAMPLE_CHARS = 20


def _has_concrete_counterexample(verdict: ReviewVerdict) -> bool:
    """Mechanical, not semantic: does `counterexample` look like an actual
    input/output pair rather than empty or a restated doubt? This does
    NOT execute or verify the counterexample -- that would mean running
    arbitrary model-supplied claims through the sandbox, a different and
    much larger feature. It only checks for the bare minimum shape a real
    one would have (some length, at least one digit -- a genuine
    input/output pair has numbers in it; a vague sentence usually
    doesn't). Deliberately cheap and gameable in principle, same as any
    format check; it exists to filter out silence and one-line
    restatements, not to adjudicate correctness.
    """
    text = (verdict.counterexample or "").strip()
    return len(text) >= _MIN_COUNTEREXAMPLE_CHARS and any(ch.isdigit() for ch in text)


def _override_ungrounded_rejection(
    verdict: ReviewVerdict,
    changed_files: list[str],
    report: RootCauseReport,
    tests_passed: bool,
) -> ReviewVerdict:
    """The one mechanical check in this module that can turn a rejection
    into an approval, not just the reverse -- added after live runs
    showed the Reviewer rejecting genuinely correct, fully in-scope,
    test-passing diffs on stylistic grounds ("a different formula than I
    expected") with no identified defect (docs-internal/ROADMAP.md's
    Known Limitations: 0/9 approvals across 3 runs prior to this).

    Deliberately narrow, and deliberately checked in this order:
      1. Only fires on a rejection the MODEL made itself -- never on one
         `enforce_review_checks` just produced a line above. A scope
         violation or a symptom mismatch is a fact this orchestrator
         verified about the repo; a missing counterexample doesn't get to
         undo that.
      2. Both preconditions for even considering an override are hard,
         ground-truth facts, not judgment calls: the orchestrator's own
         pytest run passed, and `out_of_scope_files` says the diff never
         left the files the root cause named. Nothing here is the
         Reviewer's word taken on faith.
      3. The escape hatch stays entirely in the Reviewer's hands: name one
         concrete input/output pair where the diff misbehaves, and the
         rejection stands untouched. It is a strictness requirement, not
         a bypass -- it can only make rejection *harder* to hand-wave, not
         easier.
    """
    if verdict.approved:
        return verdict
    if not tests_passed:
        return verdict
    if not report.affected_files:
        return verdict
    if out_of_scope_files(changed_files, report.affected_files):
        return verdict
    if _has_concrete_counterexample(verdict):
        return verdict

    return verdict.model_copy(
        update={
            "approved": True,
            "rejection_reason": (
                "MECHANICALLY OVERRIDDEN (chain._override_ungrounded_rejection): "
                "all tests pass and the diff is confined to the root cause's "
                "affected_files, but the rejection named no concrete "
                f"input/output counterexample. Original verdict: approved=False"
                + (f" -- {verdict.rejection_reason}" if verdict.rejection_reason else "")
            ),
        }
    )


def enforce_review_checks(
    verdict: ReviewVerdict,
    changed_files: list[str],
    report: RootCauseReport,
    tests_passed: bool = True,
) -> ReviewVerdict:
    """Apply the mechanical checks that sit on top of the Reviewer's
    judgment. An approval is a model opinion; these are not.

    Almost everything here only ever downgrades an approval -- a stricter
    Reviewer is always respected. The one exception is
    `_override_ungrounded_rejection`, applied last and only to a
    rejection that wasn't itself produced by a downgrade below (see its
    own docstring for why that ordering matters).

    `tests_passed` defaults to True to keep every existing call site
    (this function predates the override and was widely called with just
    3 positional args) behaving exactly as before; run_chain always
    passes the real ground-truth value explicitly.
    """
    if not verdict.approved:
        return _override_ungrounded_rejection(verdict, changed_files, report, tests_passed)

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
    # Milestone 6: the real GitHub PR URL, set once create_pull_request
    # succeeds. None when no PR was opened (blocked, failed, or the task
    # never reached PR_CREATION).
    pr_url: str | None = None
    container_id: str | None = None
    error: str | None = None
    stages: list[str] = field(default_factory=list)
    # Milestone 5: every tool call made by every agent across the whole
    # run, in order, each tagged with which agent made it -- {"agent":,
    # "name":, "args":, "success":, "error_code":, "message":}. Without
    # this there was no way to check "did search_code actually get used"
    # from outside the chain, which the milestone's own verification bar
    # explicitly asks to be shown (found while writing that check, not
    # anticipated up front).
    tool_calls: list[dict] = field(default_factory=list)


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
        # Milestone 9: a Watcher-created task is already put into TRIAGING
        # by orchestrator/watch.py's triage_anomaly() before run_chain()
        # is ever called (it has to be, to legally reach
        # MERGED_INTO_EXISTING/CANCELLED, both only reachable FROM
        # TRIAGING -- see state_machine.py's TRANSITIONS). There is no
        # (TRIAGING, TRIAGING) self-transition, so calling go(TRIAGING)
        # unconditionally here would raise IllegalTransitionError for
        # every Watcher-sourced task. A CLI-driven `amop fix` task is
        # still CREATED at this point exactly as before, so this is
        # additive, not a behavior change for the existing path.
        if TaskState(task.state) is TaskState.CREATED:
            await go(TaskState.TRIAGING, actor="system")
            stage("TRIAGING: bug report accepted, treating as novel")
        else:
            stage(
                f"TRIAGING: already triaged by caller (state={task.state}) -- "
                "dedup/severity checked upstream"
            )
        await go(TaskState.INVESTIGATING, actor="system")

        # -- INVESTIGATING -------------------------------------------
        # Section 10.3: retrieved once here, at task-context-construction
        # time, and reused for the whole task -- not re-queried inside the
        # agent's reasoning loop.
        relevant_memory, memory_hits = await _retrieve_relevant_memory(ctx, description)
        if memory_hits:
            stage(
                f"INVESTIGATING: injected {memory_hits} related past "
                "incident(s) from memory as evidence"
            )
        stage("INVESTIGATING: investigator examining the repo")
        report = await _run_agent(
            agents.investigator,
            _investigator_prompt(description, relevant_memory),
            result,
        )
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
        noop_attempts = 0
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
            code_report, full_diff = await _run_coder(
                agents.coder, ctx, report, findings_feedback, str(task.id), result
            )
            result.code_change_report = code_report
            stage(
                f"CODING: {code_report.status}, files changed: "
                f"{code_report.files_changed or '[]'}"
                + (" [no_op: no mutating tool call this attempt]" if code_report.no_op else "")
            )

            # Milestone 6, Section 29.1: diff-size cap, extending Section
            # 6.3.9's existing file-scope guard with a hard line-of-code
            # ceiling. Checked after-the-fact here (reusing the diff
            # _run_coder already computed) rather than proactively inside
            # CoderAgent's loop -- a smaller, lower-risk change, still a
            # real code-enforced gate: an oversized edit never reaches
            # TESTING, let alone PR_CREATION. Only checked on a genuine,
            # trusted new diff (status == "success") -- a no_op/failed
            # attempt is handled by its own retry path below regardless of
            # how large a stale earlier-attempt diff happens to be.
            if code_report.status == "success" and scope_guard.over_cap(full_diff):
                loc = scope_guard.changed_line_count(full_diff)
                code_report = code_report.model_copy(
                    update={
                        "status": "needs_decomposition",
                        "failure_diagnostic": (
                            f"diff is {loc} changed lines, exceeding the "
                            f"{scope_guard.MAX_LOC_PER_TASK}-line cap (Section 29.1)"
                        ),
                    }
                )
                result.code_change_report = code_report
                stage(
                    f"CODING: diff exceeds size cap ({loc} > "
                    f"{scope_guard.MAX_LOC_PER_TASK}) -- needs_decomposition"
                )
                await go(
                    TaskState.NEEDS_HUMAN_INPUT,
                    actor=f"agent:{agents.coder.name}",
                    trigger=(
                        f"diff exceeds coder.max_loc_per_task cap "
                        f"({loc} > {scope_guard.MAX_LOC_PER_TASK})"
                    ),
                )
                stage(
                    "NEEDS_HUMAN_INPUT: diff too large for one task -- "
                    "decompose into smaller changes"
                )
                return result

            # -- TESTING ---------------------------------------------
            await go(TaskState.TESTING, actor=f"agent:{agents.coder.name}")
            stage("TESTING: running the repo's own test suite")
            test_report = await _run_tester(
                agents.tester, ctx, str(task.id), result, description
            )
            result.test_report = test_report
            stage(f"TESTING: all_passed={test_report.all_passed} — {test_report.details[:160]}")

            # Bugfix: a no-op coding attempt (zero mutating tool calls)
            # must never reach Reviewer with a stale diff, and getting
            # lucky on a leftover passing suite doesn't change that.
            # Handled before route_after_testing so it can never fall
            # through to REVIEWING regardless of test outcome. Uses its
            # own small budget (MAX_NOOP_ATTEMPTS) instead of consuming
            # a real fix_iterations/review_cycles slot -- this isn't the
            # test-failure or review-rejection retry path, it's "the
            # coder didn't actually do anything, try again."
            if code_report.no_op:
                noop_attempts += 1
                stage(
                    f"CODING: discarding no-op attempt ({noop_attempts}/"
                    f"{MAX_NOOP_ATTEMPTS}) instead of sending a stale diff "
                    "to review"
                )
                if noop_attempts >= MAX_NOOP_ATTEMPTS:
                    await go(
                        TaskState.FAILED,
                        actor=f"agent:{agents.coder.name}",
                        trigger=(
                            f"coder made no mutating tool calls across "
                            f"{MAX_NOOP_ATTEMPTS} attempts"
                        ),
                    )
                    stage(
                        "FAILED: coder made no file changes after repeated "
                        "attempts (no_op budget exhausted)"
                    )
                    return result
                coding_actor = f"agent:{agents.coder.name}"
                coding_trigger = (
                    f"coder attempt made no mutating tool calls (no_op retry "
                    f"{noop_attempts} < {MAX_NOOP_ATTEMPTS}, does not consume "
                    "the fix-iteration or review-cycle budget)"
                )
                findings_feedback = (
                    "Your last turn ended without calling write_file or "
                    "patch_file -- no file was actually changed, so there was "
                    "nothing new to test or review. You must make a concrete "
                    "edit (write_file or patch_file) before giving your final "
                    "answer."
                )
                continue

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
                agents.reviewer, ctx, report, test_report, description, str(task.id), result
            )

            # Mechanical checks on top of the model's verdict. Reviewing
            # only happens once tests have already passed (route_after_
            # testing gates on it above), so tests_passed is always True
            # here in practice -- passed explicitly rather than relying
            # on the default, since that default exists for other/older
            # call sites, not this one.
            checked = enforce_review_checks(
                verdict, code_report.files_changed, report, tests_passed=test_report.all_passed
            )
            if checked.approved != verdict.approved:
                direction = "approved -> rejected" if verdict.approved else "rejected -> approved"
                stage(f"REVIEWING: mechanical override ({direction}) — {checked.rejection_reason}")
            verdict = checked
            result.review_verdict = verdict
            # Visibility into the mechanical-override input, not just its
            # output: without this, "why wasn't this rejection overridden?"
            # is unanswerable from the log alone. Only relevant on a
            # rejection -- an approval has nothing to override.
            counterexample_note = ""
            if not verdict.approved:
                counterexample_note = (
                    f" [counterexample: {verdict.counterexample}]"
                    if verdict.counterexample
                    else " [no counterexample given]"
                )
            stage(
                f"REVIEWING: approved={verdict.approved}"
                + (f" — {verdict.rejection_reason}" if verdict.rejection_reason else "")
                + counterexample_note
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

        # -- PR_CREATION -----------------------------------------------
        # Milestone 6: real GitHub PR creation, replacing the Milestone
        # 4/5 simulated print. On success the task goes to
        # WAITING_FOR_APPROVAL, never MERGED -- auto-merge is explicitly
        # not built this milestone; a human must review and merge the
        # real PR on GitHub themselves.
        await go(TaskState.PR_CREATION, actor=f"agent:{agents.reviewer.name}")
        result.diff = await _get_diff(ctx)
        stage("PR_CREATION: opening a real pull request on GitHub")

        pr_result = await _create_pull_request(
            ctx, report, code_report, result.diff, str(task.id)
        )
        if not pr_result.success:
            stage(
                f"PR_CREATION: failed to open PR -- "
                f"{pr_result.error_code}: {pr_result.message}"
            )
            await go(
                TaskState.NEEDS_HUMAN_INPUT,
                actor="system",
                trigger=f"create_pull_request failed: {pr_result.error_code}",
            )
            result.error = pr_result.message
            stage(f"NEEDS_HUMAN_INPUT: PR creation blocked/failed ({pr_result.error_code})")
            return result

        result.pr_url = pr_result.output.get("url")
        stage(f"PR_CREATION: opened {result.pr_url}")
        await go(
            TaskState.WAITING_FOR_APPROVAL,
            actor="system",
            trigger=(
                "create_pull_request succeeded (auto-merge not implemented "
                "this milestone -- human-gated)"
            ),
        )
        stage("WAITING_FOR_APPROVAL: a human must review and merge the real PR on GitHub")
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


def _record_tool_calls(chain_result: ChainResult, agent_name: str, agent_result) -> None:
    """Append `agent_result.tool_calls`, each tagged with which agent
    made it, onto the chain-wide log. This is what lets a caller (the
    CLI, a test) check "was search_code actually used" from outside the
    chain -- AgentResult.tool_calls is otherwise discarded by every
    _run_* helper below once it's extracted the field each one needs."""
    chain_result.tool_calls.extend(
        {"agent": agent_name, **call} for call in agent_result.tool_calls
    )


async def _run_agent(agent, prompt: str, chain_result: ChainResult):
    """Run an agent and return its validated handoff, or raise
    _ChainFailure. Section 4.7: a handoff that fails validation is an
    agent failure, never a silent pass-through."""
    result = await agent.run(prompt)
    _record_tool_calls(chain_result, agent.name, result)
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


def _investigator_prompt(description: str, relevant_memory: str = "") -> str:
    memory_block = f"\n\n{relevant_memory}" if relevant_memory else ""
    return (
        f"Bug report: {description}{memory_block}\n\n"
        "The repository is checked out at /workspace. Investigate and "
        "report the root cause."
    )


async def _retrieve_relevant_memory(
    ctx: ToolContext, description: str
) -> tuple[str, int]:
    """Section 10.3's retrieval, run ONCE per task at context-construction
    time -- explicitly not per agent-loop iteration, "to keep retrieval
    cost and prompt-token cost bounded per task rather than per
    tool-call".

    Returns (rendered_block, match_count). The count is returned rather
    than recovered by scanning the rendered text, so the operator-facing
    stage line can't drift from what was actually injected.

    Best-effort for the same reason the write side is: memory is an
    enhancement to the investigation, so a retrieval failure must degrade
    to "no memory" rather than fail a task that could otherwise proceed.
    """
    if ctx.db_session is None or ctx.repo_path is None:
        return "", 0
    try:
        matches = await memory_store.search_memory(
            ctx.db_session,
            description,
            repo_path=ctx.repo_path,
            top_k=memory_store.FEW_SHOT_TOP_K,
            min_similarity=memory_store.RELEVANCE_FLOOR,
        )
    except Exception:  # noqa: BLE001 -- see docstring: never fatal
        return "", 0
    return memory_store.render_relevant_memory(matches), len(matches)


async def _run_coder(
    agent, ctx, report: RootCauseReport, feedback: str, task_id: str, chain_result: ChainResult
) -> tuple[CodeChangeReport, str]:
    # Milestone 10: when the Investigator already named affected files,
    # tell the Coder to read them directly rather than re-discover them
    # via search_code -- the real-repo diagnosis found the Coder
    # re-searching from scratch, missing the file (a large-chunk
    # embedding gap, see indexer.py's _prepare_for_embedding), and giving
    # up despite already having a confident, specific answer in hand.
    if report.affected_files:
        files_line = (
            "Affected files (from the investigation -- read these "
            "directly with read_file first; only use search_code if you "
            f"need more context beyond them): {report.affected_files}\n\n"
        )
    else:
        files_line = ""

    prompt = (
        f"Root cause: {report.root_cause}\n"
        f"Suggested fix plan: {report.suggested_fix_plan}\n\n"
        f"{files_line}"
        "Apply the smallest change that fixes this root cause. Fix the "
        "source code, not the tests. The repository is at /workspace."
    )
    if feedback:
        prompt += f"\n\nFeedback you must address:\n{feedback}"

    agent_result = await agent.run(prompt)
    _record_tool_calls(chain_result, agent.name, agent_result)

    # A failed Coder is deliberately NOT a chain failure. Section 6.3.3:
    # "Coder hands off a CodeChangeReport with status: 'failed' and its
    # best diagnostic -- the orchestrator, not the Coder, decides whether
    # that's a task failure (Section 4.2: TESTING -> FAILED)." So the run
    # continues into TESTING, where the real suite gets the final word;
    # the retry budget handles it from there.
    diagnostic = None if agent_result.success else f"coder agent failed: {agent_result.error}"

    # Bugfix: whether THIS attempt actually did anything, independent of
    # git state. changed_files() below diffs against the baseline, so a
    # committed edit from an *earlier* attempt still shows up even when
    # this attempt's turn made no write_file/patch_file call at all --
    # observed live, where a third coding attempt called only search_code
    # + read_file, made no edit, and was still handed to Reviewer as if it
    # were a fresh diff. Checked from the tool-call log itself, not from
    # git, so it can't be fooled by a stale commit sitting on the branch.
    no_op = not any(
        call["success"] and (spec := get_tool(call["name"])) is not None and spec.mutating
        for call in agent_result.tool_calls
    )
    if no_op and diagnostic is None:
        diagnostic = (
            "coder made no mutating tool calls this attempt (no write_file/"
            "patch_file) -- nothing new was produced to test or review"
        )

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

    code_report = CodeChangeReport(
        task_id=task_id,
        status="success" if changed and agent_result.success and not no_op else "failed",
        branch=branch,
        commit_sha=sha,
        files_changed=changed,
        diff_summary=full_diff[:2000],
        iterations_used=agent_result.iterations_used,
        failure_diagnostic=diagnostic,
        no_op=no_op,
    )
    return code_report, full_diff


async def _run_tester(
    agent, ctx, task_id: str, chain_result: ChainResult, description: str = ""
) -> TestReport:
    """Run the Tester, then overwrite its all_passed with ground truth.

    The model contributes interpretation; pytest contributes the verdict.
    A Tester that claims success while the suite is red cannot move the
    task forward, because the value the router reads never came from it.

    Milestone 6, Section 29.1: the flaky-test double-check. Any test that
    failed against the fix branch and ISN'T carve-out-protected (not named
    in the original bug report, not one Tester just wrote) gets re-run
    against the base branch; if it fails there too, it's excluded from
    `all_passed`/`details` as environmental noise rather than real signal.
    This is glue code around agents/tester.py's pure classify_failures --
    all the checkout/re-run/restore I/O lives here, outside the model's
    control, matching how ground_truth itself is computed.
    """
    ground_truth = await _run_tests_ground_truth(ctx)

    prompt = (
        "A fix has just been applied to the repository at /workspace. "
        "Run the test suite and summarize the current state of it."
    )
    agent_result = await agent.run(prompt)
    _record_tool_calls(chain_result, agent.name, agent_result)

    details = ""
    handoff_new_tests: list[str] = []
    regression_confirmed = False
    if agent_result.success and agent_result.handoff is not None:
        details = agent_result.handoff.details
        handoff_new_tests = agent_result.handoff.new_tests_added
        regression_confirmed = agent_result.handoff.regression_confirmed
    else:
        # A failed Tester agent is not a failed task: the authoritative
        # result already exists. Record the degradation instead of
        # discarding a perfectly good pytest run.
        details = f"(tester agent unavailable: {agent_result.error})"

    fix_branch_failing = [f["test_name"] for f in ground_truth["failures"]]
    all_passed = ground_truth["all_passed"]

    if fix_branch_failing:
        # Build the re-run candidates: fix-branch failures minus anything
        # carve-out-protected, which is never even eligible to be checked
        # against base -- see tester_mod.is_carveout_protected's docstring
        # for why (a fresh regression test failing pre-fix, or the exact
        # bug report's own named test failing on both branches, is the
        # expected/correct signal, not noise).
        candidates: dict[str, str] = {}
        for f in ground_truth["failures"]:
            name = f["test_name"]
            if tester_mod.is_carveout_protected(name, description, handoff_new_tests):
                continue
            nodeid = f"{f['file']}::{name}" if f.get("file") else name
            candidates[name] = nodeid

        base_by_nodeid = await _rerun_failing_tests_against_base(
            ctx, list(candidates.values())
        )
        base_by_name = {
            name: base_by_nodeid.get(nodeid, False) for name, nodeid in candidates.items()
        }

        excluded, effective = tester_mod.classify_failures(
            fix_branch_failing, base_by_name, description, handoff_new_tests
        )
        if excluded:
            details = (
                f"{details} | excluded as environmental noise "
                f"(fails on base branch too): {excluded}"
            )
        if effective:
            details = f"{details} | failing: {', '.join(effective)}"
        # Ground truth said failed, but every failure that's still real
        # signal after the carve-out-aware double-check is empty -- the
        # suite is effectively green for this milestone's purposes.
        all_passed = all_passed or not effective

    return TestReport(
        task_id=task_id,
        all_passed=all_passed,
        details=details.strip(),
        new_tests_added=handoff_new_tests,
        regression_confirmed=regression_confirmed,
    )


async def _rerun_failing_tests_against_base(ctx, nodeids: list[str]) -> dict[str, bool]:
    """Re-run each of `nodeids` against sandbox.repo.BASE_BRANCH (the
    pre-fix commit) and report whether it failed there too. Restores the
    original branch (and any stashed changes) in a `finally`, regardless
    of outcome -- this must never leave the sandbox checked out somewhere
    other than where the rest of the chain expects it.

    An inconclusive re-run (tool error, timeout) simply leaves that
    nodeid out of the returned dict; classify_failures' fail-safe default
    treats an absent entry as "not conclusively flaky", so a test never
    gets silently excluded on an inconclusive result.
    """
    if not nodeids or ctx.sandbox is None:
        return {}

    sandbox = ctx.sandbox
    original_branch = await asyncio.to_thread(git_repo.current_branch, sandbox)
    stashed = await asyncio.to_thread(git_repo.stash_if_dirty, sandbox)
    results: dict[str, bool] = {}
    try:
        await asyncio.to_thread(git_repo.checkout, sandbox, git_repo.BASE_BRANCH)
        for nodeid in nodeids:
            outcome = await invoke_tool(
                "run_tests", {"path": nodeid}, ctx, agent_name="orchestrator"
            )
            results[nodeid] = bool(
                outcome.success and outcome.output and outcome.output.get("failed", 0) > 0
            )
    finally:
        await asyncio.to_thread(git_repo.checkout, sandbox, original_branch)
        if stashed:
            await asyncio.to_thread(git_repo.pop_stash, sandbox)
    return results


async def _run_reviewer(
    agent,
    ctx,
    report: RootCauseReport,
    test_report: TestReport,
    description: str,
    task_id: str,
    chain_result: ChainResult,
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
    verdict = await _run_agent(agent, prompt, chain_result)
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


async def _create_pull_request(
    ctx,
    report: RootCauseReport,
    code_report: CodeChangeReport,
    diff: str,
    task_id: str,
) -> ToolResult:
    """Orchestrator-initiated, not model-initiated -- same pattern as
    _get_diff/_run_tests_ground_truth. Title/body shape mirrors the old
    Milestone 4/5 simulated-PR block (cli/main.py) that this milestone
    replaces with a real API call."""
    args = {
        "title": f"fix: {report.root_cause[:72]}",
        "body": (
            f"Opened automatically by AMOP for task {task_id}.\n\n"
            f"## Root cause\n{report.root_cause}\n\n"
            f"## Fix plan\n{report.suggested_fix_plan}\n\n"
            f"## Files changed\n{code_report.files_changed}\n\n"
            f"## Diff\n```diff\n{diff}\n```\n"
        ),
        "head": code_report.branch,
        "base": git_repo.BASE_BRANCH,
    }
    return await invoke_tool("create_pull_request", args, ctx, agent_name="orchestrator")


# ---------------------------------------------------------------------
# Milestone 14 / Section 10.2: write-on-resolution incident memory
# ---------------------------------------------------------------------

# Section 10.2 says "every resolved task (terminal state RESOLVED or
# FAILED)". Taken literally against THIS state machine, that hook would
# almost never fire: nothing in the codebase ever transitions to RESOLVED
# (its only legal predecessor is MERGED, and nothing reaches that either
# -- auto-merge isn't built). Since Milestone 6 a successful run ends at
# WAITING_FOR_APPROVAL, which isn't even in TERMINAL_STATES. So a literal
# reading would record failures only, and few-shot retrieval would learn
# exclusively from things that went wrong.
#
# What the spec is actually asking for is "the task is over, record what
# happened", so that's what this is: every terminal state, plus
# WAITING_FOR_APPROVAL as the de-facto success terminal today. When the
# MERGED -> RESOLVED path does get built, RESOLVED is already in the set
# and nothing here needs to change.
MEMORY_WRITE_STATES = frozenset(TERMINAL_STATES | {TaskState.WAITING_FOR_APPROVAL})


async def _record_incident_memory(
    session: AsyncSession,
    result: ChainResult,
    repo_path: str,
    emit: Callable[[str], None],
) -> None:
    """Section 10.2's write-on-resolution, called once per completed
    chain run.

    Deliberately best-effort: a memory is a record of work already done,
    so failing to write one must never turn a finished task into a failed
    one. The exception is caught, surfaced to the operator, and dropped.
    """
    if result.final_state not in MEMORY_WRITE_STATES:
        return
    try:
        item = await memory_store.write_incident_memory(
            session, result, repo_path=repo_path
        )
    except Exception as exc:  # noqa: BLE001 -- see docstring: never fatal
        emit(f"memory: failed to record incident memory ({exc})")
        return
    if item is not None:
        emit(f"memory: recorded incident memory {item.id}")


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

    resolved_repo_path = str(Path(repo_path).resolve())
    await asyncio.to_thread(git_repo.materialize, Path(repo_path), scratch_dir)

    manager = await asyncio.to_thread(SandboxManager)
    task_id = str(task.id)
    sandbox = await asyncio.to_thread(manager.create, task_id, scratch_dir)
    try:
        await asyncio.to_thread(git_repo.init_baseline, sandbox)
        await asyncio.to_thread(
            git_repo.create_branch, sandbox, f"amop/fix-{uuid.UUID(task_id).hex[:8]}"
        )

        # Section 7.1: onboarding-time indexing, an orchestrator-level
        # step (not an agent tool call) run once before the chain starts.
        # "Index fresh each time" (CLAUDE.md's authorized simplification):
        # index_repo() deletes and re-populates every run rather than
        # incrementally updating, keyed on resolved_repo_path so repeated
        # runs against the same source repo don't accumulate stale rows.
        chunk_count = await index_repo(session, resolved_repo_path, scratch_dir)
        emit(f"Indexed {chunk_count} code chunks from {resolved_repo_path}")

        ctx = ToolContext(
            agent_name="chain",
            scratch_dir=scratch_dir,
            mode=mode,
            sandbox=sandbox,
            repo_path=resolved_repo_path,
            db_session=session,
        )
        agents = ChainAgents.build(model, ctx, task_id)
        emit(f"Sandbox container: {sandbox.short_id}")
        emit(f"Workspace: {scratch_dir}")
        result = await run_chain(
            session, task, description=description, ctx=ctx, agents=agents, emit=emit
        )
        await _record_incident_memory(session, result, resolved_repo_path, emit)
        return result
    finally:
        await asyncio.to_thread(manager.destroy, task_id)
