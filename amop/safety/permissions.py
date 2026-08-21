"""Permission mode resolution — spec Section 12.1, Design Decision D-8.

D-8: "mode is set at three levels with most-specific-wins precedence:
global default < per-repo override < per-agent-per-repo override. This
is what lets 'DependencyUpdater is autonomous, everything else on this
repo is suggestor' coexist with 'this whole other repo is observer-only'
in one config file."

**Why this module is pure, with no database access at all.** Section
12.2 states as a property worth having that `evaluate()` "is pure and
unit-testable in isolation from any LLM call -- the entire safety
posture of the system can be regression-tested without spending a single
model token." `resolve_mode()` is called from inside `evaluate()` and is
synchronous; the overrides themselves live in Postgres
(`repositories.permission_overrides`). Reaching into the database from
here would either make the whole gate async or hide blocking I/O inside
it, and would make every Safety Engine test need a live database.

So the overrides are loaded ONCE, at context-construction time, by
whoever builds the ToolContext for a run (see `load_permission_overrides`
below), and travel on `ctx.permission_overrides` as a plain dict. That
is the same discipline `ctx.mode` already used ("already resolved onto
ctx" -- safety/engine.py) and the same once-per-task pattern Milestone 14
used for memory retrieval. The result is that the precedence rules here
are a pure function of their arguments, and the gate stays exactly as
testable as Section 12.2 wants it to be.

Overrides JSONB shape, matching Section 12.1's own worked example:

    {"default": "observer", "agents": {"dependency_updater": "autonomous"}}

Both keys are optional. An absent/empty dict means "no override", which
resolves to the global mode -- i.e. every repo registered before this
milestone, and every repo registered without explicit overrides, behaves
exactly as it did before.
"""

from typing import Any

VALID_MODES = frozenset({"observer", "suggestor", "operator", "autonomous"})


def resolve_mode_from(
    agent: str, global_mode: str, overrides: dict[str, Any] | None
) -> str:
    """D-8's precedence chain, as a pure function.

    global default < per-repo override < per-agent-per-repo override.

    An override naming a mode this system doesn't have is IGNORED rather
    than honored or raised on. Deliberate: this value comes from a JSONB
    column a human edits by hand (or a future config file), and the two
    alternatives are both worse than falling back. Raising would let a
    typo in one repo's config break unrelated tool calls at evaluation
    time -- inside the safety gate, which is the last place that should
    fail open OR fail loudly-but-unsafely. Honoring it would mean an
    unrecognized string silently becoming... something. Falling back to
    the next-least-specific level is the only option that stays
    predictable, and it fails toward the global default rather than
    toward more permission.
    """
    mode = global_mode
    if not overrides:
        return mode

    repo_default = overrides.get("default")
    if isinstance(repo_default, str) and repo_default in VALID_MODES:
        mode = repo_default

    agents = overrides.get("agents")
    if isinstance(agents, dict):
        agent_mode = agents.get(agent)
        if isinstance(agent_mode, str) and agent_mode in VALID_MODES:
            mode = agent_mode

    return mode


async def load_permission_overrides(session, repo_path: str | None) -> dict | None:
    """Fetch one repo's overrides, for the caller that builds a
    ToolContext -- NOT called from inside the Safety Engine.

    Returns None when the repo isn't registered, which is the common case
    for anything run straight off a path (`amop fix --repo ...` against
    an unregistered directory) and resolves to the global mode.
    Registration is what opts a repo into per-repo policy; not being
    registered is not an error.
    """
    # Imported here rather than at module scope to keep this module free
    # of database imports for every caller that only wants the pure
    # resolver above.
    from sqlalchemy import select

    from amop.database.models import Repository

    if session is None or not repo_path:
        return None
    normalized = _normalize(repo_path)
    row = (
        await session.execute(
            select(Repository).where(Repository.repo_path == normalized)
        )
    ).scalar_one_or_none()
    return row.permission_overrides if row is not None else None


def normalize_repo_identity(repo_path: str) -> str:
    """The single canonical form of a repo's working identity.

    Delegates to Milestone 16's `concurrency.normalize_repo` rather than
    re-deriving `Path(...).resolve()` here, deliberately: that function
    already defines what "the same repo" means for advisory-lock keys and
    concurrency slots. If permission lookup used its own normalization,
    two definitions of repo identity could drift apart and a repo could
    end up locked under one identity while its permissions resolved under
    another. One definition, reused.

    This must be applied on WRITE as well as read -- registering a repo
    stores the normalized form. Found the hard way: on macOS `/tmp` is a
    symlink to `/private/tmp`, so a row stored raw as "/tmp/x" never
    matched a lookup for the resolved "/private/tmp/x", and every
    per-repo override silently resolved to "no overrides" -- failing
    toward the global mode, quietly, with nothing raising.
    """
    from amop.orchestrator.concurrency import normalize_repo

    return normalize_repo(repo_path)


def _normalize(repo_path: str) -> str:
    return normalize_repo_identity(repo_path)
