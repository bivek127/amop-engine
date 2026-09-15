"""Crash Recovery Reconciliation -- spec Section 4.6.2, Milestone 29 Stage 2.

Micro-commits (4.6.1, sandbox/repo.py) make filesystem work durable. They
do not make recovery automatic -- durable git state and durable database
state are two separate stores with no shared transaction, so a task
stranded mid-CODING/TESTING by a crashed orchestrator needs this
reconciliation step BEFORE it can safely re-enter the state machine.

RECONCILE's job, spec's own words: "1. Re-clone/fetch the working branch
... from the remote if it was ever pushed ... 2. Compare: DB's last
recorded commit_sha vs. actual branch HEAD ... 3. Verify no external
effect is orphaned (case E) ... 4. Only after 1-3 succeed does the task
re-enter its state machine."

Real-architecture departure from the spec's literal step 1, approved
explicitly (not assumed): nothing in this codebase is ever pushed to a
remote before PR_CREATION (github.py's push_to_remote is only called
from create_pull_request), so for a task stranded in CODING/TESTING
there is never a remote copy of its branch to fetch. What DOES usually
survive a crashed orchestrator is the host-mounted scratch directory
itself -- SandboxManager.destroy() only removes the container, never the
host bind-mount (sandbox/manager.py) -- so that surviving directory
stands in for "the remote" as the source of real git truth here. Only
when it, too, is gone does this degrade to the spec's literal case C.

Case A vs. case D, also approved explicitly: the spec tells these apart
by diffing against the remote ("git succeeded, DB didn't record" vs. "a
human pushed to the branch"). With no remote to diff against for these
tasks, commit authorship is the substitute signal -- every AMOP-written
commit anywhere in this codebase carries one of exactly two identities
(amop-bot, or the sandbox image's baked-in "AMOP Agent" -- see
sandbox/repo.py's branch_advanced_by_amop_only). A commit by neither
identity, sitting on the branch RECONCILE is about to adopt, means
something other than AMOP wrote it.

Case B is checked narrowly, per spec's own words for it ("should be
impossible given 4.6.1's ordering, but defended anyway"): only when the
DB's recorded commit_sha isn't even a real commit object in the repo.
Every other kind of non-ancestor divergence is folded into case D --
nothing in AMOP's own commit flow (micro_commit, commit_all,
squash_wip_commits) ever removes or rewrites a commit, so a branch whose
HEAD no longer contains what the DB expects, by any other means, is
external interference by construction, not a defect to silently repair.

Bounded resume, not general resumability (approved explicitly): every
resumable outcome here (EQUAL, case A, case B, case C) funnels through
the SAME single re-entry point -- a RootCauseReport handed to
run_chain(resume_report=...), which re-enters at PLANNING_FIX via the
(CODING|TESTING, PLANNING_FIX) edges state_machine.py adds for exactly
this (Milestone 29). A Coder re-examining already-correct work (case A)
simply finds it correct and moves on; that's a safe, small cost, not a
new code path to build and maintain.

Spec 4.6.2's case B remediation also says "TRUNCATE agent_messages for
the current agent run back to the start of the current state". No such
table exists in this codebase's real schema (checked: database/models.py
has no agent_messages) -- the bounded resume design above makes this
moot rather than unaddressed: run_chain(resume_report=...) never
rehydrates a prior CODING-loop conversation, it starts a brand-new Coder
loop at fix_iterations=0. There is no stale agent memory to truncate
because none is ever replayed on resume.

A real, separate gap this surfaces, not fixed here: any task that
predates this milestone's task_context['root_cause_report'] persistence
(chain.py's Stage 2 write, INVESTIGATING) has no recoverable root cause
at all. RECONCILE cannot invent one -- reconciling case A/B/C for such a
task routes to NEEDS_HUMAN_INPUT, reason root_cause_not_recorded,
exactly like no_working_repo_recorded below. This is not a synthetic
edge case: it is the real shape of at least one genuinely-stranded task
already in the dev database (see Stage 2 evidence).
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from amop.agents.handoffs import RootCauseReport
from amop.database.models import PullRequest, Task
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import transition_with_retry
from amop.sandbox import repo as git_repo
from amop.sandbox import tools as sandbox_tools
from amop.sandbox.manager import SandboxManager

_ACTOR = "system:reconcile"


def _noop(_: str) -> None:
    pass


@dataclass
class ReconcileResult:
    """RECONCILE's verdict for one stranded task.

    `resumable`: whether run_chain(resume_report=...) should be called
    at all. False means this function already transitioned the task to
    NEEDS_HUMAN_INPUT itself (spec: RECONCILE runs "before re-entering
    any state's entry action" -- routing away from the state machine IS
    the entry action for that outcome, so it's done here, not left for a
    caller to remember to do).

    `report`: the RootCauseReport to hand to run_chain(resume_report=...)
    when resumable is True. Never None in that case.

    `case`: "equal" | "A" | "B" | "C" | "D" | "F" | a precheck name
    (no_working_repo_recorded, root_cause_not_recorded) -- for evidence
    and test assertions, not used by any caller's control flow.

    `needs_fresh_checkout`: True only for case C. The surviving-scratch-
    dir cases (equal/A/B) can hand their EXISTING host_scratch_dir
    straight to a new sandbox; case C has no scratch dir left at all, so
    resuming means re-materializing the source repo from scratch (same
    as a brand-new run_fix) before run_chain(resume_report=...) can even
    start.

    `reason`: the NEEDS_HUMAN_INPUT trigger string when resumable is
    False; empty otherwise.

    `detail`: one human-readable line for CLI/emit output.
    """

    resumable: bool
    report: RootCauseReport | None
    case: str
    needs_fresh_checkout: bool
    reason: str
    detail: str


async def _to_human_input(
    session, task: Task, reason: str, detail: str
) -> ReconcileResult:
    await transition_with_retry(
        session, task, TaskState.NEEDS_HUMAN_INPUT, trigger=reason, actor=_ACTOR
    )
    return ReconcileResult(
        resumable=False,
        report=None,
        case=reason,
        needs_fresh_checkout=False,
        reason=reason,
        detail=detail,
    )


def _load_report(task_context: dict) -> RootCauseReport | None:
    raw = task_context.get("root_cause_report")
    if not raw:
        return None
    return RootCauseReport.model_validate(raw)


async def _adopt_head(
    session, task: Task, task_context: dict, branch: str, head: str
) -> None:
    """Case EQUAL/A: git is the more expensive state to reconstruct, so
    it wins (spec's own words) -- write it into the DB as the new
    recorded state."""
    task.task_context = {**task_context, "branch": branch, "commit_sha": head}
    session.add(task)
    await session.commit()


async def _check_orphaned_pr(
    session, task: Task, repo_path: str, branch: str, emit: Callable[[str], None]
) -> None:
    """Spec step 3, case E: adopt an already-open PR with no local
    pull_requests row, rather than leaving it invisible to
    GET /pull-requests or risking a duplicate later (create_pull_request
    would otherwise open a second PR for the same head branch). Best-
    effort -- a missing token or a transient GitHub error must not block
    an otherwise-resumable task on a side check the spec itself frames
    as orphan *prevention*, not a precondition for resuming. In practice
    this rarely fires for a CODING/TESTING-stranded task (nothing is
    pushed before PR_CREATION -- see module docstring); checked anyway,
    per spec step 3's own "only after 1-3 succeed" ordering.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        emit("RECONCILE: no GITHUB_TOKEN set -- skipping case E orphan-PR check")
        return
    try:
        from github import Auth, Github

        from amop.tools.github import find_open_pr_for_branch

        client = Github(auth=Auth.Token(token))
        found = await find_open_pr_for_branch(client, branch, git_repo.BASE_BRANCH)
    except Exception as exc:  # noqa: BLE001 -- best-effort, see docstring
        emit(f"RECONCILE: case E check failed, continuing anyway ({exc})")
        return
    if found is None:
        return
    emit(f"RECONCILE: adopting orphaned PR {found['url']} (case E)")
    try:
        session.add(
            PullRequest(
                task_id=task.id,
                repo_path=repo_path,
                url=found["url"],
                number=found.get("number"),
                status="open",
            )
        )
        await session.commit()
    except Exception as exc:  # noqa: BLE001 -- same best-effort reasoning
        emit(f"RECONCILE: warning -- could not record adopted PR row ({exc})")


async def reconcile(
    session,
    task: Task,
    *,
    emit: Callable[[str], None] = _noop,
) -> ReconcileResult:
    """Run spec 4.6.2's RECONCILE algorithm for one task stranded in
    CODING or TESTING. See this module's docstring for the real-
    architecture departures from the spec's literal remote-fetch
    wording, all explicitly approved rather than assumed.

    Never raises for an ordinary reconciliation outcome -- every case,
    including the two "cannot reconcile at all" prechecks, resolves to a
    ReconcileResult; the caller decides what to do with it (a CLI resume
    command calling run_chain(resume_report=...) when resumable, doing
    nothing further otherwise -- this function already performed the
    NEEDS_HUMAN_INPUT transition itself in that case).
    """
    task_context = task.task_context or {}
    repo_path = task_context.get("repo")

    if not repo_path:
        detail = (
            "NEEDS_HUMAN_INPUT: no working repo recorded on this task at all -- "
            "nothing for reconciliation to compare against (likely a "
            "pre-Milestone-4 scaffold task, not a real crashed chain run)"
        )
        emit(detail)
        return await _to_human_input(
            session, task, "no_working_repo_recorded", detail
        )

    scratch_dir = Path(sandbox_tools.SCRATCH_DIR) / str(task.id)
    if not scratch_dir.is_dir():
        # Case C, spec's own words: "Task is reset to the last state
        # whose outputs are reconstructible from the DB alone (typically
        # PLANNING_FIX)". If that output (the RootCauseReport) was never
        # persisted either, there is nothing left to reconstruct from --
        # this is the root_cause_not_recorded precheck, not a
        # contradiction of case C.
        report = _load_report(task_context)
        if report is None:
            detail = (
                "NEEDS_HUMAN_INPUT: branch/container are gone (case C) and no "
                "root cause was ever recorded for this task -- nothing to "
                "reconstruct PLANNING_FIX's output from"
            )
            emit(detail)
            return await _to_human_input(
                session, task, "root_cause_not_recorded", detail
            )
        detail = (
            "case C: container and scratch dir are both gone -- accepted data "
            "loss per spec 4.6.2 (never pushed before PR_CREATION); resuming "
            "with a clean checkout from the last recorded root cause"
        )
        emit(f"RECONCILE: {detail}")
        return ReconcileResult(
            resumable=True,
            report=report,
            case="C",
            needs_fresh_checkout=True,
            reason="",
            detail=detail,
        )

    manager = SandboxManager()
    sandbox = await asyncio.to_thread(manager.create, str(task.id), scratch_dir)
    try:
        if not await asyncio.to_thread(git_repo.git_fsck, sandbox):
            detail = (
                "NEEDS_HUMAN_INPUT: git fsck reports corruption on resume "
                "(case F) -- AMOP does not attempt automated repair"
            )
            emit(detail)
            return await _to_human_input(
                session, task, "repo_integrity_failure", detail
            )

        head = await asyncio.to_thread(git_repo.current_sha, sandbox)
        branch = await asyncio.to_thread(git_repo.current_branch, sandbox)
        db_sha = task_context.get("commit_sha")

        case: str
        if db_sha is None or db_sha == head:
            # db_sha is None: DB "doesn't know" -- spec's own case-A table
            # wording. db_sha == head: EQUAL, states already agree.
            # Either way head is truth and nothing needs rolling back.
            case = "A" if db_sha is None else "equal"
        elif not await asyncio.to_thread(git_repo.commit_exists, sandbox, db_sha):
            # Case B, narrowly: the DB's recorded commit isn't even a
            # real object in this repo.
            case = "B"
        elif await asyncio.to_thread(git_repo.is_ancestor, sandbox, db_sha, head):
            # HEAD is strictly ahead of db_sha on the same line of
            # history -- case A, UNLESS something other than AMOP wrote
            # the extra commits, which makes it case D instead.
            if await asyncio.to_thread(
                git_repo.branch_advanced_by_amop_only, sandbox, db_sha
            ):
                case = "A"
            else:
                detail = (
                    f"NEEDS_HUMAN_INPUT: HEAD ({head[:12]}) is ahead of the "
                    f"recorded commit ({db_sha[:12]}) but not every commit in "
                    "between was authored by AMOP -- branch_modified_externally "
                    "(case D)"
                )
                emit(detail)
                return await _to_human_input(
                    session, task, "branch_modified_externally", detail
                )
        else:
            # db_sha is a real commit, but HEAD does not descend from it
            # at all -- a true divergence. Nothing in AMOP's own commit
            # flow removes or rewrites history, so this is external
            # interference by construction (case D).
            detail = (
                f"NEEDS_HUMAN_INPUT: HEAD ({head[:12]}) does not contain the "
                f"recorded commit ({db_sha[:12]}) and neither is an ancestor "
                "of the other -- branch_modified_externally (case D)"
            )
            emit(detail)
            return await _to_human_input(
                session, task, "branch_modified_externally", detail
            )

        report = _load_report(task_context)
        if report is None:
            detail = (
                f"NEEDS_HUMAN_INPUT: case {case} is otherwise resumable, but no "
                "root cause was ever recorded for this task (predates "
                "Milestone 29's task_context persistence)"
            )
            emit(detail)
            return await _to_human_input(
                session, task, "root_cause_not_recorded", detail
            )

        if case in ("A", "B"):
            await _adopt_head(session, task, task_context, branch, head)

        await _check_orphaned_pr(session, task, repo_path, branch, emit)

        detail = f"case {case}: resuming from recorded root cause, HEAD {head[:12]}"
        emit(f"RECONCILE: {detail}")
        return ReconcileResult(
            resumable=True,
            report=report,
            case=case,
            needs_fresh_checkout=False,
            reason="",
            detail=detail,
        )
    finally:
        await asyncio.to_thread(manager.destroy, str(task.id))
