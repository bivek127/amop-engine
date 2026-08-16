"""Tool Registration & Contract — spec Section 8.1/8.2.

Invocation pipeline (Section 8.1, every tool call, no exceptions):
    Agent emits tool_use -> Registry validates JSON-schema args ->
    Safety Engine evaluates permission (12.2) -> [if mutating] blacklist/
    allowlist check (12.4) -> Sandbox executes (9) -> Registry normalizes
    result -> Audit log write (12.6) -> result returned to agent

No sandbox this milestone (that's Milestone 3), so tools execute
in-process directly. Audit logging this milestone is the returned
ToolResult + the caller's tool_calls log (agents/coder.py, cli/main.py),
not a persisted agent_actions row -- Section 12.6 isn't in this
milestone's scope.

A rejection at any gate short-circuits: the tool body never runs, and the
caller gets a structured ToolResult(success=False, error_code=...,
message=...) -- never a silent no-op.
"""

import asyncio
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ToolContext:
    """Passed to every tool call and to Safety Engine evaluation.
    Section 12.1's full per-repo/per-agent mode precedence is deferred --
    `mode` here is this milestone's single global mode.

    `sandbox` (Milestone 3): the task's live sandbox.manager.Sandbox, set
    by the caller (agents/coder.py) once its container is up. Optional
    and defaulted to None so Milestone 2's ToolContext(...) call sites
    and tests keep working unchanged; sandboxed tools (sandbox/tools.py)
    require it to be set.

    `repo_path` / `db_session` (Milestone 5): codebase_intel/search.py's
    search_code tool needs a stable repo identity to filter code_chunks
    by (repo_path -- the resolved SOURCE repo path, not scratch_dir; see
    database/models.py's CodeChunk docstring) and a live DB session for
    its semantic half. db_session is the SAME AsyncSession the caller
    (orchestrator/chain.py's run_fix) already holds, reused rather than
    opening a second engine/connection pool -- safe because the chain is
    strictly sequential, never concurrent DB access within one task run.
    Both default to None so every existing ToolContext(...) call site
    keeps working unchanged; search_code fails cleanly, not silently,
    when either is unset (see its own docstring)."""

    agent_name: str
    scratch_dir: Path
    mode: str
    sandbox: Any = None
    repo_path: str | None = None
    db_session: Any = None


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error_code: str | None = None
    message: str | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema, Section 8.2
    mutating: bool
    timeout_seconds: float
    func: Callable[..., Awaitable[ToolResult]]


_REGISTRY: dict[str, ToolSpec] = {}


def tool(
    name: str,
    description: str,
    parameters: dict,
    mutating: bool,
    timeout_seconds: float = 10.0,
):
    """Decorator per Section 8.2's metadata contract. Registers the
    decorated async function under `name`."""

    def decorator(func: Callable[..., Awaitable[ToolResult]]):
        _REGISTRY[name] = ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            mutating=mutating,
            timeout_seconds=timeout_seconds,
            func=func,
        )
        return func

    return decorator


def get_tool(name: str) -> ToolSpec | None:
    return _REGISTRY.get(name)


def all_tools() -> list[ToolSpec]:
    return list(_REGISTRY.values())


def _validate_args(spec: ToolSpec, args: dict) -> str | None:
    """Minimal JSON-schema validation: required keys present, and present
    args roughly type-match the schema. Not a full JSON-schema validator
    (stdlib-only, per this milestone's scope) -- good enough for the
    simple object/string-property schemas our tools declare."""
    schema = spec.parameters or {}
    required = schema.get("required", [])
    for key in required:
        if key not in args:
            return f"missing required argument: {key}"

    properties = schema.get("properties", {})
    type_map = {"string": str, "object": dict, "boolean": bool, "number": (int, float)}
    for key, value in args.items():
        prop_schema = properties.get(key)
        if not prop_schema:
            continue
        expected = type_map.get(prop_schema.get("type"))
        if expected and not isinstance(value, expected):
            return f"argument {key!r} must be of type {prop_schema['type']}"
    return None


async def invoke_tool(
    name: str,
    args: dict,
    ctx: ToolContext,
    agent_name: str,
    allowed_tools: Collection[str] | None = None,
) -> ToolResult:
    """The full pipeline for one tool call: agent tool allowlist ->
    validate args -> Safety Engine evaluate -> execute -> normalize ->
    return. Imports safety.engine.evaluate lazily to avoid a
    module-import cycle (safety.engine imports ToolSpec/ToolContext from
    this module).

    `allowed_tools` (Milestone 4) is Section 5.1's per-agent tool subset.
    It's checked first: a tool this agent may not call is refused before
    its arguments are even inspected, since nothing about the call could
    make it permissible. None means "no per-agent restriction" (the
    Milestone 2/3 call sites, unchanged).
    """
    from amop.safety.engine import evaluate

    spec = get_tool(name)
    if spec is None:
        return ToolResult(
            success=False, error_code="UNKNOWN_TOOL", message=f"No such tool: {name}"
        )

    if allowed_tools is not None and name not in allowed_tools:
        return ToolResult(
            success=False,
            error_code="TOOL_NOT_PERMITTED",
            message=(
                f"agent {agent_name!r} may not call {name!r} "
                f"(permitted: {sorted(allowed_tools)})"
            ),
        )

    validation_error = _validate_args(spec, args)
    if validation_error:
        return ToolResult(
            success=False, error_code="INVALID_ARGS", message=validation_error
        )

    decision = evaluate(agent_name, spec, args, ctx)
    if not decision.allow:
        return ToolResult(
            success=False, error_code="DENIED", message=decision.reason
        )

    try:
        result = await asyncio.wait_for(
            spec.func(**args, ctx=ctx), timeout=spec.timeout_seconds
        )
    except TimeoutError:
        return ToolResult(
            success=False,
            error_code="TIMEOUT",
            message=f"{name} exceeded {spec.timeout_seconds}s",
        )
    except Exception as exc:  # tool body raised -- normalize, never propagate
        return ToolResult(success=False, error_code="TOOL_ERROR", message=str(exc))

    return result
