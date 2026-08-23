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
