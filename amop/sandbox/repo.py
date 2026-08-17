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
"""

import shutil
import subprocess
from pathlib import Path

from amop.sandbox.manager import Sandbox

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

    if has_changes(sandbox):
        _git(sandbox, "add -A")
        _git(sandbox, "commit -m 'baseline: fixture repo as seeded'")
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
    outcome the orchestrator has to handle, not an error to swallow."""
    if not has_changes(sandbox):
        return None
    _git(sandbox, "add -A")
    # -m via a quoted string: message is orchestrator-controlled, never
    # model-controlled, so there's no injection surface here.
    _git(sandbox, f"commit -m {_shell_quote(message)}")
    return current_sha(sandbox)


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
    """
    committed = _git(sandbox, f"diff --name-only {base} HEAD")
    uncommitted = _git(sandbox, "diff --name-only HEAD")
    names = {line.strip() for line in (committed + uncommitted).splitlines()}
    return sorted(n for n in names if n)


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
