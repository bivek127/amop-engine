import asyncio
import sys
import uuid
from pathlib import Path

import click

from amop.agents.coder import CoderAgent
from amop.database.session import init_db, make_engine, make_session_factory
from amop.models.ollama import DEFAULT_MODEL, OllamaProvider
from amop.orchestrator.chain import run_fix
from amop.orchestrator.state_machine import IllegalTransitionError, TaskState
from amop.orchestrator.task import create_task, get_task, get_transitions, transition

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

        task.task_context = {
            **(task.task_context or {}),
            "final_state": result.final_state.value,
            "error": result.error,
            "stages": result.stages,
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

    if result.diff:
        # PR creation is simulated this milestone -- this is the diff and
        # summary a PR description would carry, not a GitHub API call.
        click.echo()
        click.echo("=" * 62)
        click.echo(f"SIMULATED PULL REQUEST — {result.code_change_report.branch}")
        click.echo("=" * 62)
        if result.root_cause_report:
            click.echo(f"\n{result.root_cause_report.root_cause}\n")
        click.echo(result.diff)
        click.echo("=" * 62)

    click.echo()
    click.echo(f"Final state: {result.final_state.value}")
    click.echo(f"Inspect the full history with:  amop status {task.id}")

    if result.final_state is not TaskState.RESOLVED:
        sys.exit(1)


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


if __name__ == "__main__":
    app()
