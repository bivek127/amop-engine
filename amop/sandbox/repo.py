"""Git operations for the task's repo, executed inside the sandbox
container (Milestone 4).

Every git command here runs in the container, never on the host: the
host side only ever copies plain files into the per-task scratch dir
before the container is created. That keeps Milestone 3's boundary
intact -- AMOP does not run git against anything on the host filesystem,
and a repo the agents mangle is a repo inside a disposable container.

Spec Section 9.1's rationale for one container per task applies directly
here: the branch Coder creates has to still be there when Tester and
Reviewer look, without re-cloning.

Milestone 29 / spec 4.6.1, 8.4, 12.2.1: micro-commits and hook-bypass.
Every AMOP-initiated commit here runs with --no-verify, and
core.hooksPath is pointed at an empty directory for the whole task --
a repo's own .git/hooks/pre-commit would otherwise fire on every
micro-commit, which is both a performance problem and a code-execution
vector from a repo AMOP was only asked to read and patch. This is set
once, in init_baseline(), before any commit (baseline or micro) can
happen.

Identity note, found while building this: spec 8.4 requires
`amop-bot <amop-bot@localhost>` as commit author/committer -- but the
sandbox image (Dockerfile) bakes in `AMOP Agent <agent@amop.local>`
globally, and every existing commit (baseline, Coder's final commit)
has used that identity since Milestone 6, not amop-bot. That drift
predates this milestone and is out of Stage 1's scope to silently fix
project-wide (it would touch the image and every prior commit's
behavior, not just micro-commits). micro_commit() below applies the
spec-correct amop-bot identity via `git -c user.name=... -c
user.email=...`, scoped to that one invocation only -- so new
micro-commits are spec-compliant without changing what identity
existing baseline/final commits use. Flagged here and in the milestone
findings rather than fixed silently.
"""

import shutil
import subprocess
from pathlib import Path

from amop.sandbox.manager import Sandbox

# spec 8.4's required identity for AMOP-authored commits. Applied via
# `git -c` overrides on individual commit invocations (micro_commit,
# squash_wip_commits) rather than the sandbox's global git config --
# see this module's docstring for why the global identity is left
# alone for now.
BOT_NAME = "amop-bot"
BOT_EMAIL = "amop-bot@localhost"

# Where core.hooksPath points inside the sandbox -- a directory that is
# guaranteed to exist and be empty, so no hook from the target repo (or
# anywhere else) can ever fire on an AMOP-initiated commit.
_EMPTY_HOOKS_DIR = "/tmp/amop-no-hooks"

# spec 4.6.1: the mechanically-identifiable prefix on every micro-commit
# message, used both to construct the message and, at squash time, to
# find/verify no such commit survives onto a branch that reaches PR
# creation.
WIP_MARKER = "[AMOP][wip]"

WORKSPACE = "/workspace"
BASE_BRANCH = "main"


class GitError(RuntimeError):
    """A git command inside the container exited non-zero."""


def materialize(source_repo: Path, host_scratch_dir: Path) -> None:
    """Copy a fixture/source repo into the per-task scratch dir that will
    be bind-mounted at /workspace.

    Host-side file copy only -- no git, no execution. The source tree is
    never mutated: every run starts from a byte-identical pristine copy,
    which is what makes the end-to-end test repeatable.
    """
    source_repo = Path(source_repo).resolve()
    if not source_repo.is_dir():
        raise FileNotFoundError(f"repo not found: {source_repo}")

    host_scratch_dir = Path(host_scratch_dir)
    host_scratch_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_repo,
        host_scratch_dir,
        dirs_exist_ok=True,
        # __pycache__ from a host-side pytest run would otherwise land in
        # the container and show up as untracked noise in git status.
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def _git(sandbox: Sandbox, args: str, timeout: float = 60.0, check: bool = True) -> str:
    result = sandbox.exec_run(f"cd {WORKSPACE} && git {args}", timeout=timeout)
    if check and result.exit_code != 0:
        raise GitError(
            f"git {args} failed (exit {result.exit_code}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


def init_baseline(sandbox: Sandbox) -> str:
    """`git init` + commit the pristine tree as the baseline on `main`, OR,
    if the materialized source already has real git history (a real
    clone, not a bare fixture directory), reuse that history instead.

    The fixture-repo case is what makes a fixture a real git repo in the
    first place -- it happens per run, inside the container, rather than
    being checked into AMOP's own repo (a nested .git would be stored by
    the parent repo as a gitlink, mode 160000, and the fixture's files
    would never actually be committed; see the fixture's README).

    Milestone 6: when the source repo materialize() copied already has a
    `.git` (a real clone -- e.g. of a real PR-target repo), synthesizing
    a fresh, disconnected one-commit history on top of it would make
    every branch Coder creates share NO common ancestor with the real
    remote's base branch -- and GitHub's create_pull_request API hard-
    rejects that ("branch has no history in common with main"), no
    matter how correct the diff is. So: reuse real history when it's
    there instead of stomping a synthetic one on top of it. This changes
    nothing for the fixture-repo case (no fixture under tests/fixtures/
    has a .git), only for a real clone.

    Milestone 29: also disarms the target repo's own git hooks for the
    rest of this task, before any commit (including this baseline one)
    can happen -- see module docstring for why (spec 4.6.1's hook-
    execution risk, closing the surface Milestone 19 named but never
    itself needed to touch).
    """
    already_a_repo = (
        sandbox.exec_run(
            f"cd {WORKSPACE} && git rev-parse --is-inside-work-tree", timeout=10
        ).exit_code
        == 0
    )
    if already_a_repo:
        _git(sandbox, f"checkout {BASE_BRANCH}")
    else:
        _git(sandbox, f"init -b {BASE_BRANCH}")

    sandbox.exec_run(f"mkdir -p {_EMPTY_HOOKS_DIR}")
    _git(sandbox, f"config core.hooksPath {_EMPTY_HOOKS_DIR}")

    if has_changes(sandbox):
        _git(sandbox, "add -A")
        _git(sandbox, "commit --no-verify -m 'baseline: fixture repo as seeded'")
    return current_sha(sandbox)


def create_branch(sandbox: Sandbox, branch: str) -> None:
    _git(sandbox, f"checkout -b {branch}")


def current_sha(sandbox: Sandbox) -> str:
    return _git(sandbox, "rev-parse HEAD").strip()


def current_branch(sandbox: Sandbox) -> str:
    return _git(sandbox, "rev-parse --abbrev-ref HEAD").strip()


def has_changes(sandbox: Sandbox) -> bool:
    """True if the working tree differs from HEAD (staged or not)."""
    return bool(_git(sandbox, "status --porcelain").strip())


def commit_all(sandbox: Sandbox, message: str) -> str | None:
    """Stage everything and commit. Returns the new sha, or None if there
    was nothing to commit -- an agent that changed no files is a real
    outcome the orchestrator has to handle, not an error to swallow.

    Milestone 29: --no-verify like every other AMOP-initiated commit
    (module docstring). Identity is deliberately NOT overridden here --
    this is the existing final-commit path, and changing its author
    identity is out of Stage 1's scope; see the identity note in the
    module docstring."""
    if not has_changes(sandbox):
        return None
    _git(sandbox, "add -A")
    # -m via a quoted string: message is orchestrator-controlled, never
    # model-controlled, so there's no injection surface here.
    _git(sandbox, f"commit --no-verify -m {_shell_quote(message)}")
    return current_sha(sandbox)


def micro_commit(sandbox: Sandbox, task_id: str, tool_name: str, path: str) -> str | None:
    """Spec 4.6.1 / Design Decision D-15: commit one successful edit to
    the working branch immediately, distinct from and prior to the
    Coder's own deliberate commit_all() at handoff. This is what makes
    a crash between a successful patch_file/write_file and the end-of-
    turn commit durable -- resume re-clones the working branch (4.6),
    which now has the edit, instead of silently losing it while
    agent_messages/agent_actions still shows the edit as applied.

    Returns the new sha, or None if the edit produced no net diff (a
    patch that changed nothing real) -- mirrors commit_all()'s own
    skip-on-empty-diff behavior.

    Author/committer is amop-bot <amop-bot@localhost> (spec 8.4),
    applied via `git -c` on this one invocation only -- see the module
    docstring's identity note for why the sandbox's global git identity
    is left alone.

    Raises GitError if the commit itself fails (e.g. a stale index
    lock) -- deliberately NOT swallowed here. Spec 4.6.1: "the tool
    result reports success-with-warning rather than failing the edit,"
    which is a decision only the caller (write_file/patch_file, which
    knows the edit itself already succeeded) can make. Swallowing the
    error in this low-level function would make that policy invisible
    and untestable at the point that actually needs it.
    """
    if not has_changes(sandbox):
        return None
    message = f"{WIP_MARKER}[{task_id}] {tool_name}: {path}"
    _git(sandbox, "add -A")
    _git(
        sandbox,
        f"-c user.name={BOT_NAME} -c user.email={BOT_EMAIL} "
        f"commit --no-verify -m {_shell_quote(message)}",
    )
    return current_sha(sandbox)


def squash_wip_commits(sandbox: Sandbox, message: str) -> str | None:
    """Collapse every commit since the branch's merge-base with
    BASE_BRANCH into one, right before the branch is ever pushed.

    Spec 4.6.1: micro-commits "are squashed/rebased away before PR
    creation... so they never appear in the final diff history" --
    "Mandatory squash-before-PR: a task reaching PR_CREATION with any
    unsquashed [wip] commit still present is a bug, not a style
    preference."

    Uses the merge-base with BASE_BRANCH rather than a remembered
    baseline sha: create_branch() always branches off BASE_BRANCH, so
    the merge-base IS the baseline commit, and this stays correct
    regardless of how many micro-commits or intermediate commit_all()
    calls happened in between (a task can cycle through CODING more
    than once across Reviewer rejections -- Milestones 4/24 -- so the
    real commit count since baseline is not always 1).

    Returns the new (squashed) sha, or None if there was nothing to
    squash (branch HEAD already equals the merge-base -- no commits
    were made at all, e.g. a no-op Coder attempt).
    """
    base_sha = _git(sandbox, f"merge-base {BASE_BRANCH} HEAD").strip()
    if current_sha(sandbox) == base_sha:
        return None
    _git(sandbox, f"reset --soft {base_sha}")
    _git(
        sandbox,
        f"-c user.name={BOT_NAME} -c user.email={BOT_EMAIL} "
        f"commit --no-verify -m {_shell_quote(message)}",
    )
    return current_sha(sandbox)


def has_wip_commits(sandbox: Sandbox, base: str = BASE_BRANCH) -> bool:
    """True if any commit since `base` still carries the micro-commit
    marker -- the acceptance-test hook for "no unsquashed [wip] commit
    survives into PR_CREATION" (spec 4.6.1, Section 27)."""
    log = _git(sandbox, f"log {base}..HEAD --format=%s")
    return any(WIP_MARKER in line for line in log.splitlines())


def diff_against_baseline(sandbox: Sandbox, base: str = BASE_BRANCH) -> str:
    """Full diff of the working branch vs the baseline branch, including
    uncommitted working-tree changes."""
    committed = _git(sandbox, f"diff {base} HEAD")
    uncommitted = _git(sandbox, "diff HEAD")
    return committed + uncommitted


def changed_files(sandbox: Sandbox, base: str = BASE_BRANCH) -> list[str]:
    """Paths changed vs the baseline, from git itself.

    Section 6.3.9: the Reviewer "is not allowed to trust Coder's
    self-report alone" -- so the orchestrator reads the changed file list
    out of the repo rather than believing what the model claimed it
    edited.

    Milestone 14 bugfix: `git diff` reports only files git already
    tracks, so a brand-new file an agent created was invisible here --
    committed edits and modifications to existing files showed up, but
    `write_file` to a path that didn't exist yet did not. Everything
    built on this function inherited the blind spot: the scope guard
    (6.3.9) couldn't see an out-of-scope NEW file, no-op detection
    counted a file-creating turn as having changed nothing, and Section
    6.7's max_files_for_auto_fix cap could be walked straight past by
    creating rather than editing. Found by a dependency-update test
    whose agent created four files and was still reported as touching
    one. `ls-files --others --exclude-standard` closes it, honoring
    .gitignore so container-generated junk still doesn't count (the
    Milestone 6 finding about throwaway repos needing their own
    .gitignore continues to apply).
    """
    committed = _git(sandbox, f"diff --name-only {base} HEAD")
    uncommitted = _git(sandbox, "diff --name-only HEAD")
    untracked = _git(sandbox, "ls-files --others --exclude-standard")
    names = {
        line.strip() for line in (committed + uncommitted + untracked).splitlines()
    }
    return sorted(n for n in names if n)


def revert_to_baseline(sandbox: Sandbox, base: str = BASE_BRANCH) -> None:
    """Throw away everything an agent did on this branch.

    Spec 6.6 describes the Optimizer's revert as `git checkout --
    <files>`, which only undoes *uncommitted* work. That isn't enough
    here: agents in this codebase reach a committed state (commit_all in
    chain.py's _run_coder), so a checkout-only revert would leave the
    change sitting on the branch while reporting it reverted -- the
    exact "the tool said success but the repo says otherwise" failure
    Milestone 12 found the hard way. `reset --hard` plus `clean -fd`
    covers committed, staged, unstaged, and newly-created files alike.

    Used by both Section 6.6's below-threshold optimization revert and
    Section 6.7's over-budget dependency migration abort.
    """
    _git(sandbox, f"reset --hard {base}")
    _git(sandbox, "clean -fd")


def _shell_quote(value: str) -> str:
    escaped = value.replace("'", "'\\''")
    return f"'{escaped}'"


# ---------------------------------------------------------------------
# Milestone 6: base-branch checkout/restore, for the flaky-test
# double-check's re-run-against-base cycle (Section 29.1). Matches
# Section 6.4's own documented mechanism for its "red before green"
# regression-test check -- "a throwaway git stash/checkout inside the
# sandbox" -- reused here for the same reason: never destroy a Coder
# attempt's uncommitted state just to look at how a test behaves on main.
# ---------------------------------------------------------------------


def stash_if_dirty(sandbox: Sandbox) -> bool:
    """Stash working-tree changes (including untracked files) before a
    throwaway checkout. Returns True iff something was actually stashed,
    so the caller knows whether to pop it back afterward."""
    if not has_changes(sandbox):
        return False
    _git(sandbox, "stash push --include-untracked")
    return True


def checkout(sandbox: Sandbox, ref: str) -> None:
    _git(sandbox, f"checkout {ref}")


def pop_stash(sandbox: Sandbox) -> None:
    _git(sandbox, "stash pop")


# ---------------------------------------------------------------------
# Milestone 6: the one deliberate exception to this module's "every git
# command runs in the container, never on the host" invariant (see module
# docstring above). The sandbox container runs network_mode="none"
# (sandbox/manager.py) -- it has no route to github.com at all -- so the
# fix branch cannot be pushed from inside it. host_scratch_dir is the
# exact same on-disk directory the container bind-mounts at /workspace
# (sandbox/manager.py's Manager.create), so a host-side git process sees
# precisely the branch/history the container wrote, with no re-clone
# needed. Every OTHER function in this module stays container-only.
# ---------------------------------------------------------------------


def push_to_remote(
    host_scratch_dir: Path, remote_url: str, branch: str, timeout: float = 60.0
) -> None:
    """Push `branch` (as-is, same name on both ends) to `remote_url` from
    the host.

    `remote_url` should carry auth embedded as
    https://x-access-token:<token>@github.com/<owner>/<repo>.git rather
    than being registered via `git remote add` -- that way the token is
    never written to .git/config on disk. Callers MUST mask the token out
    of any error text raised here before it reaches a ToolResult.message
    or a log line: git's own stderr sometimes echoes the remote URL
    verbatim on failure.
    """
    result = subprocess.run(
        ["git", "push", remote_url, f"{branch}:{branch}"],
        cwd=host_scratch_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise GitError(
            f"git push failed (exit {result.returncode}): {result.stderr.strip()}"
        )


def fetch_remote_branch_sha(
    host_scratch_dir: Path, remote_url: str, branch: str, timeout: float = 30.0
) -> str | None:
    """Milestone 29 / spec 4.6.2: what's actually on the remote for
    `branch`, right now -- from the host, mirroring push_to_remote's
    reasoning exactly (the sandbox runs network_mode="none", so this
    can't happen from inside the container).

    Uses `git ls-remote`, not a real clone/fetch: RECONCILE only needs
    to know whether the branch exists remotely and what sha it's at,
    not its content -- a full fetch would be slower and would pull
    objects nothing here needs.

    Returns None if the branch doesn't exist on the remote at all
    (spec 4.6.2's "otherwise the branch exists only in the destroyed
    container" case), never raises for that -- a missing branch is an
    expected, common outcome (this project never pushes before
    PR_CREATION, so any task reconciled before reaching it will
    legitimately have no remote branch), not an error.

    Same token-masking obligation as push_to_remote: callers must not
    let `remote_url` reach a ToolResult/log line unmasked.
    """
    result = subprocess.run(
        ["git", "ls-remote", remote_url, branch],
        cwd=host_scratch_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise GitError(
            f"git ls-remote failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    line = result.stdout.strip()
    if not line:
        return None
    # `ls-remote` output is "<sha>\t<ref>" per matching ref; only one ref
    # should ever match an exact branch name.
    return line.split()[0]


def git_fsck(sandbox: Sandbox) -> bool:
    """Milestone 29 / spec 4.6.2 case F: `git fsck` on resume. True if
    the repo passes integrity checking; False if it reports any
    corruption. AMOP does not attempt automated repair of a corrupted
    repository (spec's own words) -- this function only detects, the
    caller (reconcile.py) decides what to do with a False result
    (NEEDS_HUMAN_INPUT, reason repo_integrity_failure)."""
    result = sandbox.exec_run(f"cd {WORKSPACE} && git fsck --no-dangling")
    return result.exit_code == 0


# ---------------------------------------------------------------------
# Milestone 29 / spec 4.6.2 case A vs. case D: when a stranded task's
# real HEAD is ahead of the sha its DB row last recorded, RECONCILE
# (orchestrator/reconcile.py) has to tell "more of AMOP's own work
# landed before the crash" (case A -- adopt it) apart from "a human
# touched this branch" (case D -- never auto-resolve). The spec
# distinguishes these by diffing against the remote; nothing is ever
# pushed before PR_CREATION in this codebase (github.py's
# push_to_remote is only called from create_pull_request), so for a
# task stranded in CODING/TESTING there is no remote copy to diff
# against -- only the surviving host-mounted scratch dir. Commit
# authorship is the substitute signal: every AMOP-initiated commit
# anywhere in this codebase carries one of exactly two identities
# (micro_commit's amop-bot, or the sandbox image's baked-in "AMOP
# Agent" -- Dockerfile, used by commit_all/init_baseline), so a commit
# author outside that set is, by construction, not something AMOP
# wrote.
_KNOWN_AMOP_IDENTITIES = frozenset({BOT_EMAIL, "agent@amop.local"})


def commit_exists(sandbox: Sandbox, sha: str) -> bool:
    """True if `sha` names a real commit object in this repo. Used for
    case B's detection: a DB-recorded commit_sha that isn't even a
    commit in the repo is unambiguously "the branch is missing the
    edit", spec 4.6.2's own words for case B."""
    result = sandbox.exec_run(f"cd {WORKSPACE} && git cat-file -e {sha}^{{commit}}")
    return result.exit_code == 0


def is_ancestor(sandbox: Sandbox, maybe_ancestor: str, descendant: str) -> bool:
    """True if `maybe_ancestor` is on `descendant`'s own history (reachable
    by walking descendant's first-parent-or-merge ancestry) -- including
    the case where they're the same commit. False for both a genuine
    non-ancestor and an unknown/missing revision (`git merge-base
    --is-ancestor` exits non-zero, non-1 for the latter; RECONCILE only
    needs the yes/no answer, and a missing revision is exactly the
    "doesn't reach it" case B/D callers already treat non-ancestry as)."""
    result = sandbox.exec_run(
        f"cd {WORKSPACE} && git merge-base --is-ancestor {maybe_ancestor} {descendant}"
    )
    return result.exit_code == 0


def branch_advanced_by_amop_only(sandbox: Sandbox, since_sha: str) -> bool:
    """True if every commit strictly after `since_sha` on the current
    branch (`since_sha..HEAD`) was authored under a known AMOP identity
    -- see the module note above this function. Vacuously True if there
    are no such commits (HEAD == since_sha or since_sha is not even an
    ancestor of HEAD; callers only call this once ahead-ness is already
    established)."""
    log = _git(sandbox, f"log {since_sha}..HEAD --format=%ae")
    authors = {line.strip() for line in log.splitlines() if line.strip()}
    return authors <= _KNOWN_AMOP_IDENTITIES
