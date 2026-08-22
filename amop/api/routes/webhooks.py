"""GitHub webhook ingestion — spec Sections 3.3 / 8.5, Milestone 21.

Watcher (Milestone 9) only polls; this pushes the moment an issue opens.
Reuses Milestone 9's entire pipeline unchanged (`_build_watcher_prompt`,
`WatcherAgent`, `find_existing_task_for_issue`, `check_anomaly_rate`,
`triage_anomaly`) — only how the candidate issue is obtained differs
(this payload, not a `list_open_issues` call).

**Auth is NOT `Depends(require_operator_token)`.** GitHub doesn't hold
this system's shared API token; it holds a webhook secret configured on
its own side, verified here via HMAC-SHA256 over the raw request body
(`api/webhook_auth.py`). This is its own, separate, equally-strict gate
-- not an oversight that this route skips the usual dependency.

**Idempotency is Postgres-only, not Redis Streams** (spec 3.3's original
design). CLAUDE.md's own scope decision, reaffirmed rather than
relitigated: this project has deliberately never built Redis
(ADR-04/ADR-06). `processed_events` (spec 4.3.2/D-17's own table) gives
the same at-least-once-delivery-made-safe guarantee for a single-process
receiver without a queue in front of it. See ROADMAP.md for the
associated tradeoff, stated honestly there.

**Ack fast, process in the background.** Everything up through the
`processed_events` commit happens synchronously and returns 200
immediately; classification (a real Ollama call) and `triage_anomaly()`
run afterward via `BackgroundTasks`. Real Ollama latency observed
repeatedly in this project's own live runs (well over a minute) would
otherwise risk GitHub's own webhook delivery timeout. The accepted
tradeoff, stated plainly rather than absorbed silently: the dedup insert
and the resulting task creation are no longer one atomic transaction, so
a crash in the background job after ack (rare) could leave a delivery
marked processed with no task created -- narrower than spec's literal
single-transaction guarantee, same spirit as this milestone's own
Redis-skip.
"""

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from amop.api.deps import get_session, get_session_factory
from amop.api.errors import api_error
from amop.api.webhook_auth import (
    DELIVERY_HEADER,
    EVENT_HEADER,
    SIGNATURE_HEADER,
    verify_signature,
)
from amop.database.models import ProcessedEvent, Repository
from amop.models.ollama import DEFAULT_MODEL, OllamaProvider

logger = logging.getLogger("amop.api.webhooks")

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


async def _resolve_webhook_secret(
    session: AsyncSession, repo_full_name: str | None
) -> tuple[str | None, Repository | None]:
    """Section 12.1's own global-default/per-entity-override shape
    (D-8), reused here for webhook secrets rather than inventing a
    second precedence pattern: a registered repo's own
    `webhook_secret` if set, else the global `GITHUB_WEBHOOK_SECRET`
    env var (spec's `webhook_secret_env`). Never a default of "no
    secret required" -- see the route body for what happens when
    neither is configured.

    Looking the repo up by a field read from the UNVERIFIED payload is
    safe: this is a read with no side effect, used only to decide which
    secret to *try*. If an attacker lies about which repo this is, the
    HMAC check that follows still fails unless they also know that
    repo's real secret -- there is no way to choose a secret that makes
    a forged signature verify.
    """
    repo_row = None
    if repo_full_name:
        repo_row = (
            await session.execute(select(Repository).where(Repository.url == repo_full_name))
        ).scalar_one_or_none()
    secret = (repo_row.webhook_secret if repo_row else None) or os.environ.get(
        "GITHUB_WEBHOOK_SECRET"
    )
    return secret, repo_row


@router.post("/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> dict:
    # Read the exact bytes GitHub signed BEFORE anything treats them as
    # data. See webhook_auth.py's module docstring for why this can't be
    # a Pydantic body parameter.
    raw_body = await request.body()

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise api_error(400, "MALFORMED_PAYLOAD", "request body is not valid JSON")

    repo_full_name = (payload.get("repository") or {}).get("full_name")
    secret, repo_row = await _resolve_webhook_secret(session, repo_full_name)

    signature = request.headers.get(SIGNATURE_HEADER)
    if not verify_signature(raw_body, signature, secret or ""):
        # Deliberately the same message whether the secret was missing,
        # the header was missing, or the signature was simply wrong --
        # distinguishing them in the response would tell a probing
        # attacker which repos are even registered.
        raise api_error(401, "INVALID_SIGNATURE", "missing or invalid webhook signature")

    # Everything below this line only runs once the payload is verified
    # authentic. No exceptions.

    delivery_id = request.headers.get(DELIVERY_HEADER)
    if not delivery_id:
        raise api_error(400, "MISSING_DELIVERY_ID", f"missing {DELIVERY_HEADER} header")

    event = request.headers.get(EVENT_HEADER, "")
    if event == "ping":
        # GitHub sends this automatically the moment a webhook is
        # configured, to verify the endpoint is reachable -- must not
        # error, and there is nothing to process.
        return {"status": "pong"}

    if event != "issues" or payload.get("action") != "opened":
        # Subscribing to "issues" fires for every action (edited,
        # labeled, closed, ...). Only "opened" is in scope, matching
        # Watcher's own existing GitHub-issues-only scope (Milestone 9).
        return {"status": "ignored", "event": event, "action": payload.get("action")}

    if repo_row is None or not repo_row.repo_path:
        # Signature verified (so this really is GitHub, for a repo whose
        # secret we knew), but we have no local checkout registered to
        # run the chain against. Acked, not processed -- distinct from
        # "ignored" above, since this IS in-scope and simply can't
        # proceed; visible in the response for whoever's watching
        # deliveries on GitHub's side.
        return {"status": "accepted", "processed": False, "reason": "no local repo_path registered"}

    # Idempotency (spec 4.3.2/D-17, Postgres-only per this milestone's
    # scope decision): insert first: a unique-violation means this
    # exact delivery was already handled, ack without reprocessing.
    session.add(ProcessedEvent(event_key=delivery_id))
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return {"status": "accepted", "processed": False, "reason": "duplicate delivery"}

    issue = payload.get("issue") or {}
    issue_dict = {
        "number": issue.get("number"),
        "title": issue.get("title") or "",
        "body": issue.get("body") or "",
    }
    provider = OllamaProvider(model=DEFAULT_MODEL)
    background_tasks.add_task(
        _process_issue_opened, repo_full_name, repo_row.repo_path, issue_dict, provider
    )
    return {"status": "accepted", "processed": True}


async def _process_issue_opened(
    repo_url: str, local_repo_path: str, issue: dict, provider
) -> None:
    """The reused pipeline, run after the HTTP response has already
    gone back to GitHub. Every real decision here is Milestone 9's own
    function, unchanged -- only the candidate issue's origin differs
    from the polling path (this payload, not `list_open_issues`)."""
    # Local imports: avoids a webhooks.py <-> cli.main import cycle
    # (cli/main.py imports plenty that eventually reaches this package),
    # and this function only runs once per background job, not per
    # request, so the import cost is negligible.
    from amop.agents.watcher import WatcherAgent
    from amop.cli.main import _build_watcher_prompt
    from amop.orchestrator.watch import find_existing_task_for_issue, triage_anomaly
    from amop.safety import circuit_breakers
    from amop.sandbox import tools as sandbox_tools
    from amop.tools.registry import ToolContext

    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            # Dedup layer 1a, same as polling: cheap, exact match, before
            # this issue ever reaches a prompt.
            existing = await find_existing_task_for_issue(session, repo_url, issue["number"])
            if existing is not None:
                logger.info(
                    "webhook: issue #%s on %s already has task %s, skipping",
                    issue["number"], repo_url, existing.id,
                )
                return

            rate_check = await circuit_breakers.check_anomaly_rate(session, repo_url)
            if not rate_check.allow:
                logger.warning(
                    "webhook: anomaly_rate_breaker tripped for %s: %s",
                    repo_url, rate_check.reason,
                )
                return

            ctx = ToolContext(
                agent_name="watcher",
                scratch_dir=sandbox_tools.SCRATCH_DIR,
                mode="observer",
                db_session=session,
            )
            prompt = _build_watcher_prompt([issue])
            watcher = WatcherAgent(provider, ctx)
            agent_result = await watcher.run(prompt)
            if not agent_result.success or agent_result.handoff is None:
                logger.warning("webhook: watcher classification failed: %s", agent_result.error)
                return

            for alert in agent_result.handoff.alerts:
                if alert.issue_index != 1:  # a length-1 batch; only index 1 is valid
                    continue
                stamped = alert.model_copy(
                    update={"repo": repo_url, "github_issue_number": issue["number"]}
                )
                task = await triage_anomaly(
                    session, stamped, local_path=Path(local_repo_path), model=provider
                )
                if task is not None:
                    logger.info(
                        "webhook: created task %s for issue #%s on %s (state=%s)",
                        task.id, issue["number"], repo_url, task.state,
                    )
    except Exception:
        # Found live, building this milestone: an unhandled exception
        # here (e.g. run_fix() failing to materialize a bad local
        # checkout) propagates into Starlette's own background-task
        # machinery, which runs INSIDE Response.__call__ -- after the
        # response body has gone out over the wire, but still able to
        # produce noisy, confusing server-side errors that look like the
        # webhook delivery itself failed when it didn't. The CLI's own
        # polling loop (`_watch`) already catches exactly this class of
        # per-cycle error one level up and continues to the next cycle;
        # this is that same discipline applied here, since a webhook
        # event has no next cycle to fall back to -- this IS the one
        # attempt, so it must fail loudly in the log and cleanly
        # everywhere else, never silently and never noisily.
        logger.exception(
            "webhook: background processing failed for issue #%s on %s",
            issue.get("number"), repo_url,
        )
