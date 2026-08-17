"""GitHub Tools — spec Section 8.5, the project's first tools with a real
external side effect. This milestone (CLAUDE.md's "Real External Side
Effects") makes AMOP touch a real GitHub repo for the first time.

`SANDBOX_REPO` has no tool-argument path to override it -- an agent can
never choose the target repo, matching the milestone's explicit safety
requirement: "NEVER point any tool at any other GitHub repo, especially
not this project's own repo (amop-engine), without the human explicitly
saying so." It IS operator-configurable via the `AMOP_GITHUB_REPO` env var
(Milestone 7: real-repo validation runs need a different scratch target
than `amop-sandbox`, and leaving this hardcoded would mean a successful
run on a different repo's clone tries to push into `amop-sandbox` by
mistake -- a real wrong-destination risk, not a hypothetical one). Default
is unchanged from Milestone 6 unless a human explicitly sets the env var;
no agent-controlled input reaches this value either way. `GITHUB_TOKEN` is
read from the environment only (per Section 12.5: never hardcoded, never
passed as a tool argument, never printed in full).

create_pull_request runs the secret scan (safety/secret_scan.py) and
diff-size cap (safety/scope_guard.py) as hard gates immediately before the
GitHub API is ever touched -- Section 12.5.1's attack table: "Pre-PR
secret scan is a hard gate blocking create_pull_request" enforced at the
"Tool Registry middleware" layer. No path through this function bypasses
either gate; the checks run before the branch is even pushed.

Branch delivery: the sandbox container has no network access at all
(network_mode="none", sandbox/manager.py) so the fix branch cannot be
pushed from inside it. This function pushes it host-side via
sandbox.repo.push_to_remote() -- see that function's docstring for why
that's the one deliberate exception to "git only runs in the container".

Idempotency (spec Section 8.5.1, v3.0): the idempotency key for
create_pull_request is the `head` branch name (amop/<task_id>, unique per
task by construction). Before creating, this queries for an existing open
PR on that head first; if one exists, it's adopted (created: False)
instead of creating a duplicate -- so a retried/re-run task never opens a
second PR for the same fix.

Client: PyGithub, not raw REST calls -- cleaner idempotency-query
ergonomics (repo.get_pulls(state="open", head=...)) than hand-rolling
pagination/error shapes for two endpoints, matching this codebase's
existing preference for thin client libraries over raw HTTP where one
exists (docker, sqlalchemy). This module is amop.tools.github, so `import
github` below resolves to PyGithub, not a self-import -- worth knowing if
this file is ever renamed or moved to a package's top level.
"""

import asyncio
import os

from github import Auth, Github
from github.GithubException import GithubException

from amop.safety import scope_guard, secret_scan
from amop.sandbox import repo as git_repo
from amop.tools.registry import ToolContext, ToolResult, tool

SANDBOX_REPO = os.environ.get("AMOP_GITHUB_REPO", "bivek127/amop-sandbox")
_SANDBOX_OWNER = SANDBOX_REPO.split("/")[0]


def _mask_token(token: str) -> str:
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}****...{token[-4:]}"


def _mask_token_in_text(text: str, token: str) -> str:
    return text.replace(token, _mask_token(token)) if token else text


def _client(token: str) -> Github:
    return Github(auth=Auth.Token(token))


@tool(
    name="create_pull_request",
    description=(
        "Open a pull request on the AMOP sandbox repo (github.com/"
        f"{SANDBOX_REPO}), or adopt an already-open PR for the same head "
        "branch. Runs a secret scan and a diff-size check as hard gates "
        "before touching the GitHub API -- a hit on either blocks PR "
        "creation entirely, with no retry path around them."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "body": {"type": "string"},
            "head": {"type": "string"},
            "base": {"type": "string"},
        },
        "required": ["title", "body", "head", "base"],
    },
    mutating=True,
    timeout_seconds=60,
)
async def create_pull_request(
    title: str, body: str, head: str, base: str, ctx: ToolContext
) -> ToolResult:
    if ctx.sandbox is None:
        return ToolResult(
            success=False,
            error_code="SANDBOX_UNAVAILABLE",
            message="no sandbox session for this task",
        )

    try:
        diff_text = await asyncio.to_thread(
            git_repo.diff_against_baseline, ctx.sandbox, base
        )
    except Exception as exc:
        return ToolResult(success=False, error_code="DIFF_ERROR", message=str(exc))

    hits = secret_scan.scan_for_secrets(diff_text)
    if hits:
        reasons = "; ".join(f"{h.pattern_name}: {h.reason}" for h in hits)
        return ToolResult(
            success=False,
            error_code="SECRET_DETECTED",
            message=f"diff blocked -- possible secret(s) found: {reasons}",
        )

    if scope_guard.over_cap(diff_text):
        loc = scope_guard.changed_line_count(diff_text)
        return ToolResult(
            success=False,
            error_code="DIFF_TOO_LARGE",
            message=(
                f"{loc} changed lines exceeds the "
                f"{scope_guard.MAX_LOC_PER_TASK}-line cap"
            ),
        )

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return ToolResult(
            success=False,
            error_code="GITHUB_AUTH_MISSING",
            message="GITHUB_TOKEN is not set",
        )

    remote_url = f"https://x-access-token:{token}@github.com/{SANDBOX_REPO}.git"
    try:
        await asyncio.to_thread(
            git_repo.push_to_remote, ctx.scratch_dir, remote_url, head
        )
    except git_repo.GitError as exc:
        return ToolResult(
            success=False,
            error_code="GITHUB_PUSH_ERROR",
            message=_mask_token_in_text(str(exc), token),
        )

    client = _client(token)
    try:
        gh_repo = await asyncio.to_thread(client.get_repo, SANDBOX_REPO)

        existing = await asyncio.to_thread(
            lambda: list(
                gh_repo.get_pulls(
                    state="open", head=f"{_SANDBOX_OWNER}:{head}", base=base
                )
            )
        )
        if existing:
            pr = existing[0]
            return ToolResult(
                success=True,
                output={"url": pr.html_url, "number": pr.number, "created": False},
            )

        pr = await asyncio.to_thread(
            gh_repo.create_pull, base=base, head=head, title=title, body=body
        )
        return ToolResult(
            success=True,
            output={"url": pr.html_url, "number": pr.number, "created": True},
        )
    except GithubException as exc:
        return ToolResult(
            success=False,
            error_code="GITHUB_API_ERROR",
            message=_mask_token_in_text(str(exc), token),
        )


@tool(
    name="get_ci_status",
    description="Best-effort combined CI status for an open pull request's head commit.",
    parameters={
        "type": "object",
        "properties": {"pr_number": {"type": "number"}},
        "required": ["pr_number"],
    },
    mutating=False,
    timeout_seconds=15,
)
async def get_ci_status(pr_number: int, ctx: ToolContext) -> ToolResult:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return ToolResult(
            success=False,
            error_code="GITHUB_AUTH_MISSING",
            message="GITHUB_TOKEN is not set",
        )

    client = _client(token)
    try:
        gh_repo = await asyncio.to_thread(client.get_repo, SANDBOX_REPO)
        pr = await asyncio.to_thread(gh_repo.get_pull, int(pr_number))
        combined = await asyncio.to_thread(
            lambda: gh_repo.get_commit(pr.head.sha).get_combined_status()
        )
        return ToolResult(
            success=True,
            output={"state": combined.state, "total_count": combined.total_count},
        )
    except GithubException as exc:
        return ToolResult(
            success=False,
            error_code="GITHUB_API_ERROR",
            message=_mask_token_in_text(str(exc), token),
        )
