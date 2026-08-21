"""The Safety Engine — spec Section 12.2, the gate every tool call passes
through. "LLMs decide what to do; software decides what's allowed" (the
project's core rule) becomes real here: evaluate() is pure, has no LLM
call in it, and is fully unit-testable in isolation (Section 19.4).

Deviation from Section 12.2's literal pseudocode, and why:
    Spec's evaluate() short-circuits every non-mutating tool straight to
    ALLOW, with no path check at all. That's safe in the full system
    because Section 9.3's sandbox container mount is what actually bounds
    read access -- reads aren't unrestricted there, they're restricted by
    a different layer that doesn't exist yet here. Milestone 2 has no
    sandbox, and its own instructions require tools to be "hard-restricted
    to a scratch working directory so nothing can touch the real project
    or system files" -- unqualified, covering reads too. Taken literally,
    spec's pseudocode would let read_file("/etc/passwd") succeed today.
    So here, path_restricted() runs first, before the mutating check, for
    any tool carrying a path argument -- everything else matches Section
    12.2's structure and ordering exactly.

Milestone 19: protected_path_match() (Section 12.2.1) is wired in below
-- previously deferred, now built (safety/protected_paths.py). Prompt-
injection input tagging (12.2.2) is a separate, complementary layer;
see safety/untrusted_input.py.
"""

from dataclasses import dataclass
from pathlib import Path

from amop.safety.blacklist import is_blacklisted
from amop.safety.permissions import resolve_mode_from
from amop.tools.registry import ToolContext, ToolSpec

# Tools that mutate the merge state and require operator+ mode. No such
# tool is registered yet (no GitHub tools this milestone) -- kept for
# structural fidelity with Section 12.2's pseudocode, currently inert.
_OPERATOR_ONLY_TOOLS = frozenset({"merge_pull_request"})


@dataclass
class Decision:
    allow: bool
    reason: str | None = None

    @classmethod
    def allow_(cls) -> "Decision":
        return cls(allow=True)

    @classmethod
    def deny(cls, reason: str) -> "Decision":
        return cls(allow=False, reason=reason)


def resolve_mode(agent: str, ctx: ToolContext) -> str:
    """Section 12.1's mode resolution, D-8's full three-level chain:
    global default < per-repo override < per-agent-per-repo override.

    Milestone 20: previously this returned ctx.mode unconditionally (D-8
    was deferred). It still does exactly that whenever a repo has no
    overrides -- which is every unregistered repo and every repo
    registered without a policy -- so no existing call site changes
    behavior. Stays pure and synchronous: the overrides were already
    loaded onto ctx by whoever built it, precisely so this function (and
    therefore evaluate()) keeps Section 12.2's "pure and unit-testable in
    isolation" property. See safety/permissions.py.
    """
    return resolve_mode_from(agent, ctx.mode, ctx.permission_overrides)


CONTAINER_MOUNT = "/workspace"


def strip_container_mount(path: str) -> str:
    """Translate a container-absolute path into a scratch-relative one.

    Milestone 4: agents are told (truthfully) that the repo is checked
    out at /workspace, so they refer to files as "/workspace/foo.py".
    Host-side, that same file lives under the task's scratch directory.
    Without this translation the Safety Engine reads "/workspace/foo.py"
    as a real absolute *host* path, finds it outside the scratch dir, and
    denies the call -- so a model doing exactly what it was told would be
    locked out of every file. Caught by a live chain run, where the Coder
    reported back "I'm unable to read or modify the file ... due to
    permissions restrictions."

    Only an exact "/workspace" or a "/workspace/" prefix is translated,
    so a lookalike like "/workspaceevil/x" stays absolute and is still
    denied. Traversal is unaffected: "/workspace/../etc/passwd" becomes
    "../etc/passwd", which the prefix check below still rejects.
    """
    if path == CONTAINER_MOUNT:
        return ""
    if path.startswith(CONTAINER_MOUNT + "/"):
        return path[len(CONTAINER_MOUNT) + 1 :]
    return path


def resolve_within_scratch(path: str, scratch_dir: Path) -> Path | None:
    """Canonicalize `path` (relative or absolute) against `scratch_dir`
    and return the resolved absolute path if it stays inside the
    scratch directory, else None.

    Container-absolute paths (/workspace/...) are translated to
    scratch-relative first — see strip_container_mount().

    Section 12.5.1's requirements, both handled by Path.resolve():
      - path traversal (../../etc/passwd): joining then resolving walks
        the ".." components out, and the resulting path fails the
        prefix check below.
      - symlink escape: Path.resolve() follows symlinks for path
        components that exist on disk (realpath semantics), so a
        symlink inside the scratch dir pointing outside it is judged by
        its target, not its name.
      - an outright absolute path (e.g. "/etc/passwd") as `path`:
        Path.__truediv__ with an absolute right-hand side discards the
        left side entirely, so it resolves to itself -- still caught by
        the prefix check.
    String-level ".." filtering alone is not relied upon anywhere here.
    """
    scratch_dir = scratch_dir.resolve()
    candidate = (scratch_dir / strip_container_mount(path)).resolve()
    if candidate == scratch_dir or candidate.is_relative_to(scratch_dir):
        return candidate
    return None


def path_restricted(tool: ToolSpec, args: dict, ctx: ToolContext) -> bool:
    """True if this call's path argument resolves outside ctx.scratch_dir.
    Tools with no "path" argument aren't restricted by this check."""
    path_arg = args.get("path")
    if path_arg is None:
        return False
    return resolve_within_scratch(path_arg, ctx.scratch_dir) is None


def blacklist_match(tool: ToolSpec, args: dict) -> bool:
    """Section 12.4. Blacklist patterns are shell-command patterns; they
    apply to a conventional "command" argument (what a future run_shell
    tool would use), not to arbitrary tool arguments -- so this is a
    no-op for today's filesystem tools and ready the moment a
    shell-executing tool is registered."""
    command = args.get("command")
    if not command:
        return False
    return is_blacklisted(command) is not None


def evaluate(agent: str, tool: ToolSpec, args: dict, ctx: ToolContext) -> Decision:
    mode = resolve_mode(agent, ctx)  # 12.1 / D-8 precedence (Milestone 20)

    # See module docstring: this check is moved ahead of the mutating
    # short-circuit as a deliberate, documented deviation from 12.2's
    # literal pseudocode, to compensate for the sandbox mount boundary
    # (9.3) that doesn't exist in this milestone.
    if path_restricted(tool, args, ctx):
        return Decision.deny("path_not_permitted")

    if not tool.mutating:
        return Decision.allow_()  # read tools always allowed at any mode >= observer
    if mode == "observer":
        return Decision.deny("mode_forbids_mutation")
    if tool.name in _OPERATOR_ONLY_TOOLS and mode not in ("operator", "autonomous"):
        return Decision.deny("merge_requires_operator")
    if blacklist_match(tool, args):
        return Decision.deny("blacklisted_command")

    # Local import: protected_paths.py imports resolve_within_scratch
    # FROM this module, so a top-level import here would be circular.
    # Deferred import breaks the cycle without restructuring either
    # module -- proportionate for one small, additive check.
    from amop.safety.protected_paths import protected_path_match

    if protected_path_match(tool, args, ctx):
        return Decision.deny("protected_infrastructure_path")

    return Decision.allow_()
