import asyncio
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import click
from sqlalchemy import select

from amop.agents.coder import CoderAgent
from amop.agents.handoffs import AnomalyAlert
from amop.agents.watcher import WatcherAgent
from amop.database.models import MemoryItem
from amop.database.session import init_db, make_engine, make_session_factory
from amop.memory import store as memory_store
from amop.models.ollama import DEFAULT_MODEL, OllamaProvider
from amop.orchestrator.chain import run_fix
from amop.orchestrator.deps import DEFAULT_PERMISSION_MODE, run_dependency_update
from amop.orchestrator.optimize import MIN_IMPROVEMENT_PCT, run_optimization
from amop.orchestrator.reporting import run_report
from amop.orchestrator.state_machine import IllegalTransitionError, TaskState
from amop.orchestrator.task import create_task, get_task, get_transitions, transition
from amop.orchestrator.watch import find_existing_task_for_issue, triage_anomaly
from amop.safety import circuit_breakers
from amop.sandbox import tools as sandbox_tools  # noqa: F401 -- registers the sandboxed tools
from amop.tools import github as github_tools  # noqa: F401 -- registers list_open_issues
from amop.tools.registry import ToolContext, invoke_tool

# Placeholder actor for the CLI's demo path through TRIAGING/INVESTIGATING/
# PLANNING_FIX/CODING — these are scaffolding transitions the CLI drives
# directly, not real Investigator/Coder-planner decisions (those arrive in
# later milestones). Kept distinct from "system" in the audit trail.
SCAFFOLD_ACTOR = "system:scaffold"


@click.group()
def app() -> None:
    """AMOP CLI."""


@app.command()
@click.option("--prompt", required=True, help="Prompt to send to the agent.")
@click.option(
    "--model", default=DEFAULT_MODEL, show_default=True, help="Ollama model to use."
)
def run(prompt: str, model: str) -> None:
    """Run a single prompt through the CoderAgent, persisted as a Task."""
    asyncio.run(_run(prompt, model))


async def _run(prompt: str, model: str) -> None:
    engine = make_engine()
    session_factory = make_session_factory(engine)
    await init_db(engine)

    async with session_factory() as session:
        task = await create_task(
            session, task_type="bug_fix", task_context={"prompt": prompt}
        )

        # Only Coder exists so far (Milestone 0). There is no real
        # Investigator/PLANNING_FIX reasoning yet, so the CLI drives these
        # states itself as scaffolding — the only legal path into CODING
        # per Section 4.2's table runs through all of them.
        try:
            for to_state in (
                TaskState.TRIAGING,
                TaskState.INVESTIGATING,
                TaskState.PLANNING_FIX,
                TaskState.CODING,
            ):
                task = await transition(session, task, to_state, actor=SCAFFOLD_ACTOR)
        except IllegalTransitionError as exc:
            click.echo(f"Internal state machine error: {exc}", err=True)
            sys.exit(1)

        provider = OllamaProvider(model=model)
        agent = CoderAgent(model=provider, task_id=str(task.id))
        result = await agent.run(prompt)

        task.task_context = {
            **(task.task_context or {}),
            "output": result.output,
            "error": result.error,
            "tool_calls": result.tool_calls,
        }
        session.add(task)
        await session.commit()

    if result.success:
        click.echo(result.output)
    else:
        click.echo(f"Task failed: {result.error}", err=True)

    # Milestone 3 item 8: which container this run actually executed
    # inside, not decoration -- absent only if sandbox creation itself
    # failed (see result.error in that case).
    click.echo()
    click.echo(f"Sandbox container: {agent.last_container_id or '(none)'}")

    # Visible proof the Safety Engine is actually in the loop (Milestone 2
    # item 6) -- every tool call the agent made, and whether it was
    # allowed or denied.
    click.echo()
    if result.tool_calls:
        click.echo("Tool calls:")
        for tc in result.tool_calls:
            # A tool can also fail for a non-permission reason (NOT_FOUND,
            # TIMEOUT, ...) -- only error_code == "DENIED" means the
            # Safety Engine actually blocked it.
            if tc["error_code"] == "DENIED":
                status_word = "DENIED"
            elif tc["success"]:
                status_word = "ALLOWED"
            else:
                status_word = f"ALLOWED (failed: {tc['error_code']})"
            click.echo(
                f"  {tc['name']}({tc['args']}) -> {status_word}"
                + (f"  [{tc['message']}]" if tc["message"] else "")
            )
    else:
        click.echo("Tool calls: (none)")

    if not result.success:
        sys.exit(1)


@app.command()
@click.option(
    "--repo",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to the repository to fix.",
)
@click.option("--description", required=True, help="Plain-text description of the bug.")
@click.option(
    "--model", default=DEFAULT_MODEL, show_default=True, help="Ollama model to use."
)
@click.option(
    "--mode",
    default="operator",
    show_default=True,
    help="Permission mode for the run (Section 12.1).",
)
def fix(repo: str, description: str, model: str, mode: str) -> None:
    """Run the full agent chain against a repo to fix a described bug."""
    asyncio.run(_fix(repo, description, model, mode))


async def _fix(repo: str, description: str, model: str, mode: str) -> None:
    engine = make_engine()
    session_factory = make_session_factory(engine)
    await init_db(engine)

    async with session_factory() as session:
        task = await create_task(
            session,
            task_type="bug_fix",
            task_context={"prompt": description, "repo": repo},
        )
        click.echo(f"Task {task.id}")
        click.echo(f"Bug: {description}")
        click.echo()

        provider = OllamaProvider(model=model)
        result = await run_fix(
            session,
            task,
            description=description,
            repo_path=Path(repo),
            model=provider,
            mode=mode,
            emit=lambda message: click.echo(f"  {message}"),
        )

        # Milestone 15: `diff` joins the fields `_fix` already persisted
        # -- GET /tasks/{id}/diff needs somewhere real to read from, and
        # ChainResult.diff was previously computed then discarded the
        # moment this function returned. Same shape now written by
        # `_optimize`/`_update_deps` below, so the API can read a
        # consistent set of task_context keys regardless of task_type.
        task.task_context = {
            **(task.task_context or {}),
            "final_state": result.final_state.value,
            "error": result.error,
            "stages": result.stages,
            "tool_calls": result.tool_calls,
            "diff": result.diff,
        }
        session.add(task)
        await session.commit()

    click.echo()
    if result.root_cause_report:
        report = result.root_cause_report
        click.echo("Root cause report:")
        click.echo(f"  cause:      {report.root_cause}")
        click.echo(f"  confidence: {report.confidence}")
        click.echo(f"  affected:   {report.affected_files}")

    if result.code_change_report:
        change = result.code_change_report
        click.echo()
        click.echo("Code change report (from git, not self-reported):")
        click.echo(f"  branch:  {change.branch}")
        click.echo(f"  commit:  {change.commit_sha}")
        click.echo(f"  files:   {change.files_changed}")

    if result.pr_url:
        # result.pr_url is GitHub's own html_url -- never contains
        # GITHUB_TOKEN, safe to print as-is. Any future diagnostic output
        # that touches the token itself must go through
        # amop.tools.github._mask_token first (Section 12.5: never print
        # a token in full).
        click.echo()
        click.echo("=" * 62)
        click.echo("PULL REQUEST OPENED")
        click.echo("=" * 62)
        click.echo(f"  {result.pr_url}")
        click.echo("=" * 62)
    elif result.diff:
        # No PR opened -- either a blocked/failed create_pull_request call
        # (result.error explains why, printed below) or the chain never
        # reached PR_CREATION. Show the diff so there's still something to
        # inspect, clearly labeled as not a pull request.
        click.echo()
        click.echo(f"Diff (no PR opened -- final state {result.final_state.value}):")
        click.echo(result.diff)

    # Milestone 5's own verification bar: "show the human the tool calls
    # to prove search was actually used" -- this is what makes that
    # checkable at all, for `amop fix` and not just `amop run`.
    click.echo()
    if result.tool_calls:
        click.echo("Tool calls:")
        for tc in result.tool_calls:
            if tc["error_code"] == "DENIED":
                status_word = "DENIED"
            elif tc["success"]:
                status_word = "ALLOWED"
            else:
                status_word = f"ALLOWED (failed: {tc['error_code']})"
            click.echo(
                f"  [{tc['agent']}] {tc['name']}({tc['args']}) -> {status_word}"
                + (f"  [{tc['message']}]" if tc["message"] else "")
            )
    else:
        click.echo("Tool calls: (none)")

    click.echo()
    if result.error:
        click.echo(f"Error: {result.error}")
    click.echo(f"Final state: {result.final_state.value}")
    click.echo(f"Inspect the full history with:  amop status {task.id}")

    # Milestone 6: a successful run now legitimately ends at
    # WAITING_FOR_APPROVAL (PR opened, human must merge -- no auto-merge)
    # rather than RESOLVED, which nothing in this milestone's chain
    # reaches anymore. NEEDS_HUMAN_INPUT is "not broken, needs attention"
    # (a low-confidence investigation, an oversized diff, a blocked/failed
    # PR creation) -- distinct from a hard FAILED/CANCELLED, so it gets
    # its own exit code rather than being lumped in with real failure.
    if result.final_state in (TaskState.WAITING_FOR_APPROVAL, TaskState.RESOLVED):
        pass
    elif result.final_state is TaskState.NEEDS_HUMAN_INPUT:
        sys.exit(2)
    else:
        sys.exit(1)


@app.command()
@click.option(
    "--repo",
    required=True,
    help="GitHub repo to poll, as owner/repo (e.g. bivek127/amop-watcher-sandbox).",
)
@click.option(
    "--local-path",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help=(
        "Pre-existing local clone of --repo. A qualifying anomaly proceeds "
        "through run_fix() against this path, same as `amop fix`. Without "
        "it, Watcher-created tasks stop at triage/dedup this milestone."
    ),
)
@click.option(
    "--interval", default=30, show_default=True, help="Seconds between poll cycles."
)
@click.option(
    "--model", default=DEFAULT_MODEL, show_default=True, help="Ollama model to use."
)
def watch(repo: str, local_path: str | None, interval: int, model: str) -> None:
    """Poll a GitHub repo's open issues on an interval, dedupe against
    existing tasks, and create real bug_fix Tasks for qualifying
    anomalies (Section 6.1)."""
    asyncio.run(_watch(repo, Path(local_path) if local_path else None, interval, model))


def _build_watcher_prompt(candidates: list[dict]) -> str:
    lines = ["Open issues to classify:\n"]
    for i, issue in enumerate(candidates, start=1):
        lines.append(f"Issue {i} (#{issue['number']}): {issue['title']}")
        lines.append(issue["body"][:2000] or "(no body)")
        lines.append("")
    return "\n".join(lines)


async def _watch(repo: str, local_path: Path | None, interval: int, model: str) -> None:
    engine = make_engine()
    session_factory = make_session_factory(engine)
    await init_db(engine)
    provider = OllamaProvider(model=model)

    click.echo(f"Watching {repo} every {interval}s" + (f" (local: {local_path})" if local_path else " (detection/triage only -- no --local-path)"))
    click.echo("Ctrl+C to stop.\n")

    try:
        while True:
            try:
                await _poll_once(session_factory, repo, local_path, provider)
            except Exception as exc:
                # Section 6.1's own failure-handling rule: a data-source
                # read failure is logged and that source is skipped for
                # the cycle -- Watcher never fails the whole cycle (and
                # here, never kills the whole `watch` process) over one
                # bad poll.
                click.echo(f"  [poll cycle error, will retry next cycle] {exc}")
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nStopped.")
    finally:
        await engine.dispose()


async def _poll_once(session_factory, repo: str, local_path: Path | None, provider) -> None:
    async with session_factory() as session:
        ctx = ToolContext(
            agent_name="watcher",
            scratch_dir=sandbox_tools.SCRATCH_DIR,
            mode="observer",
            db_session=session,
        )

        result = await invoke_tool(
            "list_open_issues", {"repo": repo}, ctx, agent_name="orchestrator"
        )
        if not result.success:
            click.echo(f"  list_open_issues failed: {result.error_code}: {result.message}")
            return
        issues = result.output

        # Dedup layer 1a: cheap, exact match. Filters BEFORE the issue
        # ever reaches Watcher's prompt -- a filtered issue was never a
        # real candidate for this cycle.
        candidates = []
        for issue in issues:
            existing = await find_existing_task_for_issue(session, repo, issue["number"])
            if existing is None:
                candidates.append(issue)

        if not candidates:
            click.echo(f"  {len(issues)} open issue(s), 0 new candidates")
            return

        rate_check = await circuit_breakers.check_anomaly_rate(session, repo)
        if not rate_check.allow:
            click.echo(f"  META-ALERT (anomaly_rate_breaker): {rate_check.reason}")
            return

        click.echo(f"  {len(issues)} open issue(s), {len(candidates)} new candidate(s)")
        prompt = _build_watcher_prompt(candidates)
        watcher = WatcherAgent(provider, ctx)
        agent_result = await watcher.run(prompt)

        if not agent_result.success or agent_result.handoff is None:
            click.echo(f"  watcher classification failed: {agent_result.error}")
            return

        for alert in agent_result.handoff.alerts:
            if not (1 <= alert.issue_index <= len(candidates)):
                click.echo(f"  skipping alert with out-of-range issue_index={alert.issue_index}")
                continue
            issue = candidates[alert.issue_index - 1]
            stamped = alert.model_copy(
                update={"repo": repo, "github_issue_number": issue["number"]}
            )
            click.echo(
                f"  AnomalyAlert: #{issue['number']} \"{issue['title']}\" "
                f"severity={stamped.severity} confidence={stamped.confidence}"
            )
            task = await triage_anomaly(
                session,
                stamped,
                local_path=local_path,
                model=provider,
                emit=lambda message: click.echo(f"    {message}"),
            )
            if task is not None:
                click.echo(f"    -> task {task.id}, final state this cycle: {task.state}")


@app.command()
@click.argument("task_id")
def status(task_id: str) -> None:
    """Show a task's current state and full transition history."""
    try:
        parsed_id = uuid.UUID(task_id)
    except ValueError:
        click.echo(f"Not a valid task id: {task_id}", err=True)
        sys.exit(1)
    asyncio.run(_status(parsed_id))


async def _status(task_id: uuid.UUID) -> None:
    engine = make_engine()
    session_factory = make_session_factory(engine)
    await init_db(engine)

    async with session_factory() as session:
        task = await get_task(session, task_id)
        if task is None:
            click.echo(f"No task found with id {task_id}", err=True)
            sys.exit(1)

        transitions = await get_transitions(session, task_id)

    click.echo(f"Task {task.id}")
    click.echo(f"  type:  {task.task_type}")
    click.echo(f"  state: {task.state}")
    click.echo(f"  created_at: {task.created_at}")
    click.echo(f"  updated_at: {task.updated_at}")
    click.echo(f"  resolved_at: {task.resolved_at}")
    click.echo()
    click.echo("Transition history:")
    if not transitions:
        click.echo("  (none)")
    for t in transitions:
        click.echo(
            f"  [{t.timestamp}] {t.from_state or '(none)'} -> {t.to_state}"
            f"  trigger={t.trigger!r} actor={t.actor!r}"
        )


@app.command()
@click.option(
    "--repo",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to the repository to optimize.",
)
@click.option(
    "--entry-point",
    default="benchmark.py",
    show_default=True,
    help="Benchmark entry point (a script with a __main__ block).",
)
@click.option(
    "--model", default=DEFAULT_MODEL, show_default=True, help="Ollama model to use."
)
@click.option(
    "--min-improvement",
    default=MIN_IMPROVEMENT_PCT,
    show_default=True,
    help="Percent improvement required to keep the change (Section 6.6).",
)
def optimize(repo: str, entry_point: str, model: str, min_improvement: float) -> None:
    """Profile, optimize, and measure -- reverting a marginal change (6.6)."""
    asyncio.run(_optimize(repo, entry_point, model, min_improvement))


async def _optimize(
    repo: str, entry_point: str, model: str, min_improvement: float
) -> None:
    engine = make_engine()
    session_factory = make_session_factory(engine)
    await init_db(engine)

    async with session_factory() as session:
        task = await create_task(
            session,
            task_type="optimization",
            task_context={"repo": repo, "entry_point": entry_point},
        )
        click.echo(f"Task {task.id}")

        result = await run_optimization(
            session,
            task,
            repo_path=Path(repo),
            model=OllamaProvider(model=model),
            entry_point=entry_point,
            min_improvement_pct=min_improvement,
            emit=lambda message: click.echo(f"  {message}"),
        )

        # Milestone 15: same shape `_fix` persists, so GET /tasks/{id}
        # (diff, actions) works uniformly across task_type.
        task.task_context = {
            **(task.task_context or {}),
            "final_state": result.report.status,
            "error": result.report.diagnostic,
            "stages": result.stages,
            "tool_calls": result.tool_calls,
            "diff": result.diff,
        }
        session.add(task)
        await session.commit()

    await engine.dispose()

    report = result.report
    click.echo()
    click.echo("Optimization report (timings measured, not self-reported):")
    click.echo(f"  status         : {report.status}")
    click.echo(f"  baseline       : {report.baseline_ms:.1f} ms")
    click.echo(f"  optimized      : {report.optimized_ms:.1f} ms")
    click.echo(f"  improvement    : {report.improvement_pct:+.1f}%")
    click.echo(f"  technique      : {report.technique or '(none reported)'}")
    click.echo(f"  files changed  : {report.files_changed or '[]'}")
    click.echo(f"  reverted       : {report.reverted}")
    if report.diagnostic:
        click.echo(f"  diagnostic     : {report.diagnostic}")

    # Same visibility every other command gives: whether the agent
    # actually profiled before hypothesizing (6.6's hard requirement) is
    # only checkable if the tool calls are shown.
    click.echo()
    if result.tool_calls:
        click.echo("Tool calls:")
        for tc in result.tool_calls:
            status_word = "ALLOWED" if tc["success"] else f"FAILED: {tc['error_code']}"
            click.echo(f"  {tc['name']} -> {status_word}")
    else:
        click.echo("Tool calls: (none)")
    if result.diff:
        click.echo()
        click.echo("Diff:")
        click.echo(result.diff)

    if report.status != "improved":
        sys.exit(2)


@app.command("update-deps")
@click.option(
    "--repo",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to the repository whose dependencies should be updated.",
)
@click.option(
    "--manifest",
    default="requirements.txt",
    show_default=True,
    help="Manifest to check, relative to the repo root.",
)
@click.option(
    "--model", default=DEFAULT_MODEL, show_default=True, help="Ollama model to use."
)
@click.option(
    "--mode",
    default=DEFAULT_PERMISSION_MODE,
    show_default=True,
    help="Permission mode (Section 6.7 defaults this agent to autonomous).",
)
def update_deps(repo: str, manifest: str, model: str, mode: str) -> None:
    """Bump vulnerable dependencies and verify the suite (Section 6.7)."""
    asyncio.run(_update_deps(repo, manifest, model, mode))


async def _update_deps(repo: str, manifest: str, model: str, mode: str) -> None:
    engine = make_engine()
    session_factory = make_session_factory(engine)
    await init_db(engine)

    async with session_factory() as session:
        task = await create_task(
            session,
            task_type="dependency_update",
            task_context={"repo": repo, "manifest": manifest},
        )
        click.echo(f"Task {task.id}")

        result = await run_dependency_update(
            session,
            task,
            repo_path=Path(repo),
            model=OllamaProvider(model=model),
            manifest=manifest,
            mode=mode,
            emit=lambda message: click.echo(f"  {message}"),
        )

        # Milestone 15: same shape `_fix`/`_optimize` persist.
        task.task_context = {
            **(task.task_context or {}),
            "final_state": result.report.status,
            "error": result.report.diagnostic,
            "stages": result.stages,
            "tool_calls": result.tool_calls,
            "diff": result.diff,
        }
        session.add(task)
        await session.commit()

    await engine.dispose()

    report = result.report
    click.echo()
    # Same reasoning as Milestone 5's tool-call output for `amop fix`:
    # without this, a needs_manual_review verdict is unfalsifiable from
    # the outside -- you can see that it gave up but not what it tried.
    click.echo("Tool calls:")
    if not result.tool_calls:
        click.echo("  (none)")
    for call in result.tool_calls:
        status = "ALLOWED" if call["success"] else f"FAILED: {call['error_code']}"
        click.echo(f"  {call['name']} -> {status}")
        if not call["success"] and call.get("message"):
            click.echo(f"      {str(call['message'])[:200]}")
    click.echo()
    click.echo("Dependency update report (counts/verdict from git and pytest):")
    click.echo(f"  status       : {report.status}")
    click.echo(f"  package      : {report.package or '(none reported)'}")
    click.echo(f"  from -> to   : {report.from_version or '?'} -> {report.to_version or '?'}")
    click.echo(f"  cve ids      : {', '.join(report.cve_ids) or '(none reported)'}")
    click.echo(f"  tests passed : {report.tests_passed}")
    click.echo(f"  files changed: {report.files_changed or '[]'}")
    click.echo(f"  reverted     : {result.reverted}")
    if report.diagnostic:
        click.echo(f"  diagnostic   : {report.diagnostic}")
    if result.diff:
        click.echo()
        click.echo("Diff:")
        click.echo(result.diff)

    if report.status != "success":
        sys.exit(2)


@app.command()
@click.option(
    "--since",
    required=True,
    help="Start of the reporting window (YYYY-MM-DD, or an ISO timestamp).",
)
@click.option(
    "--until",
    default=None,
    help="End of the window (YYYY-MM-DD or ISO). Defaults to now.",
)
@click.option(
    "--model", default=DEFAULT_MODEL, show_default=True, help="Ollama model to use."
)
def report(since: str, until: str | None, model: str) -> None:
    """Summarize a window of activity (Section 6.8)."""
    try:
        start = _parse_window_bound(since)
        end = _parse_window_bound(until) if until else datetime.now(UTC)
    except ValueError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    if start > end:
        click.echo("--since must be before --until", err=True)
        sys.exit(1)
    asyncio.run(_report(start, end, model))


def _parse_window_bound(value: str) -> datetime:
    """Accept a plain date or a full ISO timestamp, always timezone-aware.

    A naive datetime compared against timezone-aware DB columns raises at
    query time rather than returning a wrong answer -- but it raises deep
    inside asyncpg, so it's normalized here where the error can name the
    actual problem.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"Not a valid date/time: {value!r} -- use YYYY-MM-DD or an ISO timestamp"
        ) from None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


async def _report(start: datetime, end: datetime, model: str) -> None:
    engine = make_engine()
    session_factory = make_session_factory(engine)
    await init_db(engine)

    try:
        async with session_factory() as session:
            provider = OllamaProvider(model=model)
            summary, window = await run_report(
                session, start=start, end=end, model=provider
            )
    finally:
        await engine.dispose()

    click.echo(f"Report for {summary.period_start} .. {summary.period_end}")
    click.echo()
    click.echo("Verified counts (from the database, not the model):")
    click.echo(f"  tasks reaching a terminal state : {summary.tasks_resolved}")
    click.echo(f"  pull requests opened            : {summary.prs_opened}")
    click.echo(f"  pull requests merged            : {summary.prs_merged}")
    click.echo(f"  dependency updates              : {summary.dependencies_updated}")
    click.echo(f"  (tasks created in window        : {window.tasks_created})")
    click.echo(f"  (of the terminal ones, failed   : {window.tasks_failed})")
    click.echo()
    click.echo("Top issues (Reporter's summary):")
    if not summary.top_issues:
        click.echo("  (none reported)")
    for issue in summary.top_issues:
        click.echo(f"  - {issue}")


@app.group()
def memory() -> None:
    """Inspect and curate long-term incident memory (Section 10)."""


@memory.command("list")
@click.option("--repo", default=None, help="Filter to one repo path.")
@click.option("--limit", default=20, show_default=True, help="Max rows to show.")
def memory_list(repo: str | None, limit: int) -> None:
    """List stored incident memories, newest first."""
    asyncio.run(_memory_list(repo, limit))


async def _memory_list(repo: str | None, limit: int) -> None:
    engine = make_engine()
    session_factory = make_session_factory(engine)
    await init_db(engine)

    async with session_factory() as session:
        stmt = select(MemoryItem).order_by(MemoryItem.created_at.desc()).limit(limit)
        if repo:
            stmt = stmt.where(MemoryItem.repo_path == repo)
        items = (await session.execute(stmt)).scalars().all()

    if not items:
        click.echo("(no memories recorded)")
        return
    for item in items:
        content = item.content or {}
        flag = " [DISPUTED]" if item.disputed else ""
        click.echo(f"{item.id}{flag}")
        click.echo(f"  recorded : {item.created_at}")
        click.echo(f"  repo     : {item.repo_path}")
        click.echo(f"  outcome  : {content.get('outcome')}")
        click.echo(f"  reported : {str(content.get('anomaly_signature') or '')[:100]}")
        click.echo(f"  diagnosed: {str(content.get('root_cause') or '(none)')[:100]}")
        click.echo()


@memory.command("dispute")
@click.argument("memory_id")
@click.option(
    "--undo",
    is_flag=True,
    help="Clear the disputed flag instead of setting it.",
)
def memory_dispute(memory_id: str, undo: bool) -> None:
    """Mark a memory as wrong (Section 10.4).

    Disputed memories are excluded from retrieval but never deleted --
    they stay in the table for audit.
    """
    try:
        parsed_id = uuid.UUID(memory_id)
    except ValueError:
        click.echo(f"Not a valid memory id: {memory_id}", err=True)
        sys.exit(1)
    asyncio.run(_memory_dispute(parsed_id, disputed=not undo))


async def _memory_dispute(memory_id: uuid.UUID, *, disputed: bool) -> None:
    engine = make_engine()
    session_factory = make_session_factory(engine)
    await init_db(engine)

    async with session_factory() as session:
        found = await memory_store.mark_disputed(session, memory_id, disputed=disputed)

    if not found:
        click.echo(f"No memory found with id {memory_id}", err=True)
        sys.exit(1)
    state = "disputed" if disputed else "no longer disputed"
    click.echo(f"Memory {memory_id} marked {state}.")
    click.echo(
        "Excluded from retrieval; the row is retained for audit (Section 10.4)."
        if disputed
        else "It will be eligible for retrieval again."
    )


@app.command("serve-api")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
def serve_api(host: str, port: int) -> None:
    """Run the internal REST API (Section 15.1's Command Layer).

    Blocks until interrupted, same shape as `amop watch`'s poll loop.
    The app is imported by string ("amop.api.app:app"), not by name,
    deliberately -- this module's own click group is ALSO named `app`;
    importing FastAPI's app object directly here would shadow it.
    """
    import uvicorn

    uvicorn.run("amop.api.app:app", host=host, port=port)


@app.command("serve-telegram")
def serve_telegram() -> None:
    """Run the Telegram bot (Section 16.1).

    An API client -- calls the running `amop serve-api` process over
    HTTP (AMOP_API_BASE_URL, default http://127.0.0.1:8000), so that
    must already be up. Blocks until interrupted.
    """
    from amop.interfaces.telegram_bot.bot import main as bot_main

    bot_main()


if __name__ == "__main__":
    app()
