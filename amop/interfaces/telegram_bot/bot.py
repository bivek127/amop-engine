"""Telegram Bot — spec Section 16.1, scoped down to CLAUDE.md's own list
of five commands: `/status`, `/ask`, `/report`, `/approve`, `/reject`.
Not built (explicitly out per CLAUDE.md's own scope): `/set_mode`,
`/pause`, `/resume`.

An API CLIENT, not a privileged path (Section 3.1's Command Layer):
every command here calls the Stage 1 REST API over `httpx`, holding the
shared `AMOP_API_TOKEN` itself, exactly like any other caller -- this
process never imports `amop.database`/`amop.orchestrator` to touch
Postgres directly.

The ONE thing this module decides on its own, never delegated to the
API: Section 15.3's separate allowlist gate, `allowed_user_ids` --
"is this Telegram user allowed to use the bot at all" is a
Telegram-specific identity question the API's own token auth has no
opinion on (the API answers "is this HTTP request authenticated", which
the bot itself always satisfies once it holds the one shared token).
"""

import logging
import os

import httpx
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

logger = logging.getLogger("amop.telegram_bot")

API_BASE_URL = os.environ.get("AMOP_API_BASE_URL", "http://127.0.0.1:8000")
API_TOKEN = os.environ.get("AMOP_API_TOKEN", "")
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

DEFAULT_POLL_INTERVAL_SECONDS = 30.0
POLL_INTERVAL_SECONDS = float(
    os.environ.get("AMOP_TELEGRAM_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL_SECONDS))
)

# Section 16.1's "Proactive push" row, scoped to the two states that are
# both (a) genuinely actionable by a human and (b) already visible
# through an existing GET /tasks?state= filter -- no new API surface
# needed for this. "New critical anomaly" and "PR opened" (the other
# two spec'd triggers) would need either a dedicated incidents feed or
# the pull_requests table polled the same way; left for a later pass,
# named here rather than silently dropped.
_NOTIFY_STATES = {
    "WAITING_FOR_APPROVAL": "\U0001f514 Task `{id}` is waiting for your approval.",
    "FAILED": "❌ Task `{id}` failed.",
}

# Section 16.1's "NL intent classification (routed to the nearest
# task_type; ambiguous requests get a clarifying reply, not a guessed
# task)" -- implemented as a keyword heuristic, not an LLM call. A
# deliberate simplification: CLAUDE.md's own text says proactive
# notifications for this milestone don't "need to be architecturally
# perfect", and the same reasoning extends to this smaller, cheaper
# piece of NL handling.
_DEPENDENCY_KEYWORDS = (
    "update",
    "dependency",
    "dependencies",
    "cve",
    "vulnerable",
    "vulnerability",
    "advisory",
    "outdated",
    "upgrade",
    "bump",
)
_OPTIMIZATION_KEYWORDS = (
    "slow",
    "optimize",
    "optimise",
    "optimization",
    "performance",
    "latency",
    "faster",
    "speed up",
    "bottleneck",
)


def classify_intent(text: str) -> str | None:
    """Returns a task_type, or None when the request is ambiguous
    enough that guessing would be worse than asking -- the spec's own
    "ambiguous requests get a clarifying reply, not a guessed task".

    Two concrete triggers for "ambiguous" under this heuristic: BOTH
    keyword sets matched (a genuine signal conflict -- something can't
    be a dependency bump and a performance fix at once), or the message
    is too short to carry any signal at all. Anything else with no
    keyword match at all falls through to `bug_fix` -- not a guess so
    much as this project's own default/most-common task type, the same
    one every other interface defaults to.
    """
    lowered = text.lower()
    is_dependency = any(k in lowered for k in _DEPENDENCY_KEYWORDS)
    is_optimization = any(k in lowered for k in _OPTIMIZATION_KEYWORDS)

    if is_dependency and is_optimization:
        return None
    if len(text.split()) < 3 and not (is_dependency or is_optimization):
        return None
    if is_dependency:
        return "dependency_update"
    if is_optimization:
        return "optimization"
    return "bug_fix"


def allowed_user_ids() -> set[int]:
    raw = os.environ.get("AMOP_TELEGRAM_ALLOWED_USER_IDS", "")
    return {int(x) for x in raw.split(",") if x.strip()}


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_TOKEN}"}


async def _api_get(path: str, params: dict | None = None) -> httpx.Response:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=15.0) as client:
        return await client.get(path, params=params)


async def _api_post(path: str, json: dict | None = None) -> httpx.Response:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=60.0) as client:
        return await client.post(path, json=json, headers=_auth_headers())


def _is_allowed(update: Update) -> bool:
    user = update.effective_user
    return user is not None and user.id in allowed_user_ids()


async def _reject_unauthorized(update: Update) -> None:
    if update.message is not None:
        await update.message.reply_text("Not authorized to use this bot.")


# ---------------------------------------------------------------------
# /status formatting -- grouped by what the human actually needs to act
# on, not chronological order (feedback from a real phone-sized read of
# the flat list this used to be). "Terminal" here is a display grouping,
# deliberately looser than the state machine's own `TERMINAL_STATES`:
# MERGED isn't state-machine-terminal (it can still become RESOLVED) but
# reads as "done" to a human skimming this list, so it's grouped there.
# ---------------------------------------------------------------------

_APPROVAL_STATE = "WAITING_FOR_APPROVAL"
_FAILED_STATE = "FAILED"
_DONE_STATE_NAMES = {
    "MERGED",
    "RESOLVED",
    "CANCELLED",
    "MERGED_INTO_EXISTING",
    "NEEDS_HUMAN_INPUT",
}

_STATE_EMOJI = {
    "WAITING_FOR_APPROVAL": "⏳",
    "FAILED": "❌",
    "CREATED": "🆕",
    "TRIAGING": "🔍",
    "INVESTIGATING": "🔎",
    "PLANNING_FIX": "📝",
    "CODING": "💻",
    "TESTING": "🧪",
    "REVIEWING": "👀",
    "PR_CREATION": "📤",
    "MERGED": "✅",
    "RESOLVED": "✅",
    "CANCELLED": "🚫",
    "MERGED_INTO_EXISTING": "🔗",
    "NEEDS_HUMAN_INPUT": "🤔",
}

_STATUS_GROUP_CAP = 10
_DESCRIPTION_MAX_LEN = 80

# Telegram MarkdownV2's reserved characters -- every one must be
# backslash-escaped in text that isn't itself meant as formatting, or
# the whole message fails to send (not just renders oddly). Task ids
# are UUID hex/hyphens only and never need this; task_type and
# free-form descriptions do, since either can contain any of these.
_MARKDOWN_V2_SPECIAL = "_*[]()~`>#+-=|{}.!\\"


def _escape_markdown_v2(text: str) -> str:
    return "".join(f"\\{ch}" if ch in _MARKDOWN_V2_SPECIAL else ch for ch in text)


def _task_description(task: dict) -> str | None:
    ctx = task.get("task_context") or {}
    desc = ctx.get("prompt")
    if not desc:
        return None
    desc = " ".join(desc.split())  # collapse embedded newlines/whitespace to one line
    if len(desc) > _DESCRIPTION_MAX_LEN:
        desc = desc[:_DESCRIPTION_MAX_LEN].rstrip() + "…"
    return desc


def _format_status_group(
    emoji: str,
    title: str,
    tasks: list[dict],
    *,
    show_description: bool,
    show_state: bool,
) -> str | None:
    if not tasks:
        return None
    lines = [f"{emoji} *{title}* \\({len(tasks)}\\)"]
    shown = tasks[:_STATUS_GROUP_CAP]
    for t in shown:
        line_emoji = _STATE_EMOJI.get(t["state"], "•")
        header = f"{line_emoji} `{t['id'][:8]}` {_escape_markdown_v2(t['task_type'])}"
        if show_state:
            header += f" — {_escape_markdown_v2(t['state'])}"
        lines.append(header)
        if show_description:
            desc = _task_description(t)
            lines.append(f"     {_escape_markdown_v2(desc)}" if desc else "     _no description_")
    remaining = len(tasks) - len(shown)
    if remaining > 0:
        lines.append(_escape_markdown_v2(f"...and {remaining} more"))
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return await _reject_unauthorized(update)
    r = await _api_get("/tasks", params={"limit": 100})
    if r.status_code != 200:
        await update.message.reply_text(f"Could not fetch tasks ({r.status_code}).")
        return
    tasks = r.json()
    if not tasks:
        await update.message.reply_text("No tasks yet.")
        return

    waiting = [t for t in tasks if t["state"] == _APPROVAL_STATE]
    failed = [t for t in tasks if t["state"] == _FAILED_STATE]
    done = [t for t in tasks if t["state"] in _DONE_STATE_NAMES]
    in_progress = [
        t
        for t in tasks
        if t["state"] not in _DONE_STATE_NAMES
        and t["state"] not in (_APPROVAL_STATE, _FAILED_STATE)
    ]

    blocks = [
        _format_status_group(
            "⏳", "Waiting for your approval", waiting, show_description=True, show_state=False
        ),
        _format_status_group("❌", "Failed", failed, show_description=False, show_state=False),
        _format_status_group(
            "🔄", "In progress", in_progress, show_description=False, show_state=True
        ),
        _format_status_group("✅", "Done", done, show_description=False, show_state=True),
    ]
    text = "\n\n".join(block for block in blocks if block)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return await _reject_unauthorized(update)
    message = " ".join(context.args) if context.args else ""
    if not message.strip():
        await update.message.reply_text("Usage: /ask <message>")
        return

    task_type = classify_intent(message)
    if task_type is None:
        await update.message.reply_text(
            "Not sure what kind of task this is -- a bug to fix, a "
            "dependency to update, or something to speed up? Try "
            "rephrasing with a bit more detail."
        )
        return

    r = await _api_post("/tasks", json={"task_type": task_type, "description": message})
    if r.status_code != 201:
        await update.message.reply_text(f"Could not create the task ({r.status_code}).")
        return
    task = r.json()
    await update.message.reply_text(
        f"Created a {task_type} task ({task['id'][:8]}). It still needs "
        "to be run (this creates the task record; execution is a "
        "separate step today)."
    )


async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return await _reject_unauthorized(update)
    await update.message.reply_text("Generating report...")
    r = await _api_post("/reports", json={})
    if r.status_code != 200:
        await update.message.reply_text(f"Could not generate the report ({r.status_code}).")
        return
    report = r.json()
    lines = [
        f"Report: {report['period_start']} .. {report['period_end']}",
        f"  resolved: {report['tasks_resolved']}  "
        f"PRs opened: {report['prs_opened']}  merged: {report['prs_merged']}  "
        f"deps updated: {report['dependencies_updated']}",
        "",
        "Top issues:",
    ]
    lines += [f"  - {issue}" for issue in report["top_issues"]] or ["  (none)"]
    await update.message.reply_text("\n".join(lines))


async def _approve_or_reject(
    update: Update, context: ContextTypes.DEFAULT_TYPE, action: str
) -> None:
    if not _is_allowed(update):
        return await _reject_unauthorized(update)
    if not context.args:
        await update.message.reply_text(f"Usage: /{action} <task_id>")
        return
    task_id = context.args[0]
    r = await _api_post(f"/tasks/{task_id}/{action}")
    if r.status_code == 200:
        await update.message.reply_text(f"Task {task_id[:8]} -> {r.json()['state']}.")
    elif r.status_code == 404:
        await update.message.reply_text(f"No task with id {task_id}.")
    elif r.status_code == 409:
        await update.message.reply_text(
            f"Can't {action} task {task_id[:8]} from its current state."
        )
    else:
        await update.message.reply_text(f"Request failed ({r.status_code}).")


async def approve_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _approve_or_reject(update, context, "approve")


async def reject_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _approve_or_reject(update, context, "reject")


# ---------------------------------------------------------------------
# Proactive notifications -- CLAUDE.md's "simple poll-and-notify loop is
# fine" allowance, taken literally: no new API surface, just periodic
# GET /tasks?state= calls diffed against what was seen last tick.
# ---------------------------------------------------------------------


async def _notify_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_data = context.application.bot_data
    seen = bot_data.setdefault("seen_by_state", {state: set() for state in _NOTIFY_STATES})
    primed = bot_data.get("primed", False)

    for state, template in _NOTIFY_STATES.items():
        r = await _api_get("/tasks", params={"state": state, "limit": 50})
        if r.status_code != 200:
            continue
        current_ids = {t["id"] for t in r.json()}
        if primed:
            for new_id in current_ids - seen[state]:
                text = template.format(id=new_id[:8])
                for user_id in allowed_user_ids():
                    try:
                        await context.bot.send_message(chat_id=user_id, text=text)
                    except Exception:
                        logger.exception("failed to notify user %s", user_id)
        seen[state] = current_ids

    # Only after every state's been polled once -- a fresh bot process
    # must never treat everything already in these states at startup as
    # "new", or the first tick would spam every historical task.
    bot_data["primed"] = True


# ---------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------

def build_application() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set -- see .env")
    if not API_TOKEN:
        raise RuntimeError("AMOP_API_TOKEN is not set -- see .env")

    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("report", report_command))
    application.add_handler(CommandHandler("approve", approve_command))
    application.add_handler(CommandHandler("reject", reject_command))

    if application.job_queue is not None:
        application.job_queue.run_repeating(
            _notify_job, interval=POLL_INTERVAL_SECONDS, first=POLL_INTERVAL_SECONDS
        )
    return application


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if not allowed_user_ids():
        logger.warning(
            "AMOP_TELEGRAM_ALLOWED_USER_IDS is empty -- every command will be "
            "rejected until it's set in .env"
        )
    application = build_application()
    application.run_polling()


if __name__ == "__main__":
    main()
