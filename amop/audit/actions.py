"""Writing `agent_actions` rows — spec Section 12.6, Milestone 23.

Every tool call that reaches a permission decision writes one row here,
allowed or denied. Section 12.6's own framing: this is "the primary
artifact for the adversarial testing suite (Section 19.4) and for
post-incident review of what an agent actually attempted, not just what
it was told to do."

Two design points that are load-bearing rather than incidental:

**Its own transaction, not the caller's.** A DENY must survive even when
the surrounding task transaction later rolls back -- that is precisely
the case you most want recorded. So this opens a separate session from
the same engine (via Milestone 20's `concurrency.engine_for`) and
commits independently. Sharing the caller's session would tie the
durability of a refusal to the success of the very work that was
refused, and would also mean this function could commit a caller's
half-finished, unrelated pending changes.

**Never fatal.** An audit-write failure must not turn a working tool
call into a failed one. Failures are logged loudly and swallowed. That
is a deliberate, and uncomfortable, tradeoff: it means a determined
attacker who can break the audit write gets a tool call with no audit
row. The alternative -- failing the tool call -- makes audit-log
availability a denial-of-service surface against the whole system, which
is worse. Logged here so the choice is visible rather than implicit.
"""

import logging
from datetime import UTC, datetime

from amop.audit.chain import append_chained
from amop.database.models import AgentAction
from amop.safety.secret_scan import redact_arguments, redact_secrets

logger = logging.getLogger("amop.audit.actions")

# `ToolResult.output` can be an entire file's contents (read_file on a
# real source file) -- an unbounded audit column would balloon the
# database and slow every verification pass, which walks every row. The
# truncation is recorded IN the stored value so a reader can tell the
# difference between "the tool returned this" and "the tool returned
# this much of something longer".
MAX_RESULT_CHARS = 2000

# Marks an agent_actions row as a MODEL CALL rather than a tool call
# (spec 11.3: "every complete() call logs {input_tokens, output_tokens,
# estimated_cost_usd} to agent_actions"). One table now carries two kinds
# of row, so every consumer has to say which it means:
#
#   tool calls   -> tool_name != MODEL_CALL_TOOL_NAME   (Section 12.6's
#                   audit record, and Section 20.1's tool-efficiency
#                   metric)
#   model calls  -> tool_name == MODEL_CALL_TOOL_NAME   (Section 20.1's
#                   tokens/cost-per-task metric)
#
# A sentinel in tool_name rather than a new discriminator column: the
# column already exists, it is already part of the hashed chain content,
# and it keeps the distinction visible in `amop audit`'s plain listing
# instead of hidden behind a flag.
MODEL_CALL_TOOL_NAME = "model.complete"


def summarize_result(result) -> str | None:
    """A compact, redacted, bounded rendering of a ToolResult."""
    if result is None:
        return None
    if result.success:
        body = "" if result.output is None else str(result.output)
        rendered = f"OK: {body}" if body else "OK"
    else:
        rendered = f"{result.error_code}: {result.message or ''}".strip()

    rendered = redact_secrets(rendered)
    if len(rendered) > MAX_RESULT_CHARS:
        omitted = len(rendered) - MAX_RESULT_CHARS
        rendered = f"{rendered[:MAX_RESULT_CHARS]}...[truncated {omitted} chars]"
    return rendered


async def record_action(
    ctx,
    *,
    agent_name: str,
    tool_name: str,
    arguments: dict | None,
    decision: str,
    decision_reason: str | None,
    result=None,
    latency_ms: int | None = None,
) -> None:
    """Write one chained `agent_actions` row. Never raises."""
    session = getattr(ctx, "db_session", None)
    if session is None:
        # Plenty of call sites (Milestone 2/3-era tests, tool invocations
        # outside a task) legitimately have no session. Not an error, and
        # deliberately not a warning either -- it would fire constantly
        # in the test suite and train everyone to ignore the log.
        return

    try:
        from amop.orchestrator.concurrency import engine_for

        engine = engine_for(session)
        # A separate session, so this commit can never carry the caller's
        # unrelated pending changes -- see the module docstring.
        from sqlalchemy.ext.asyncio import async_sessionmaker

        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as audit_session:
            row = AgentAction(
                task_id=getattr(ctx, "task_id", None),
                agent_name=agent_name,
                tool_name=tool_name,
                arguments=redact_arguments(arguments),
                result=summarize_result(result),
                decision=decision,
                decision_reason=decision_reason,
                latency_ms=latency_ms,
                timestamp=datetime.now(UTC),
            )
            await append_chained(audit_session, row)
            await audit_session.commit()
    except Exception:  # noqa: BLE001 -- see module docstring
        logger.exception(
            "audit: failed to record %s decision for %s.%s -- the tool call "
            "itself is unaffected, but this action has NO audit row",
            decision, agent_name, tool_name,
        )


async def record_model_call(
    ctx,
    *,
    agent_name: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    latency_ms: int | None = None,
) -> None:
    """Write one chained `agent_actions` row for a model completion.

    Spec 11.3 puts these in the same table as tool calls, which is what
    lets Section 20's cost-per-task metric aggregate them alongside
    everything else a task did. `tool_name` is MODEL_CALL_TOOL_NAME so
    the two row kinds stay separable -- see that constant for the
    contract every consumer follows.

    `decision` is deliberately NULL: no Safety Engine decision was made
    here. A model call is not gated the way a tool call is, and writing
    a fake "ALLOW" would put rows in the audit trail claiming a
    permission check that never ran.

    Never raises, for the same reason record_action never does: losing
    an audit row must not take down the run that produced it. The
    failure is logged loudly instead.
    """
    session = getattr(ctx, "db_session", None)
    if session is None:
        return

    try:
        from sqlalchemy.ext.asyncio import async_sessionmaker

        from amop.orchestrator.concurrency import engine_for

        engine = engine_for(session)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as audit_session:
            row = AgentAction(
                task_id=getattr(ctx, "task_id", None),
                agent_name=agent_name,
                tool_name=MODEL_CALL_TOOL_NAME,
                # The model that answered -- which, with Milestone 27's
                # fallback in play, may not be the one configured. 11.2
                # calls that "intentional visibility, not a bug to hide",
                # so it is recorded per call rather than assumed per run.
                arguments={"model": model},
                result=None,
                decision=None,
                decision_reason=None,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                timestamp=datetime.now(UTC),
            )
            await append_chained(audit_session, row)
            await audit_session.commit()
    except Exception:  # noqa: BLE001 -- see module docstring
        logger.exception(
            "audit: failed to record a model call for %s -- the completion "
            "itself is unaffected, but its token usage has NO audit row",
            agent_name,
        )
