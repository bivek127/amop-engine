import asyncio
import sys
import uuid

import click

from amop.agents.coder import CoderAgent
from amop.models.ollama import DEFAULT_MODEL, OllamaProvider
from amop.orchestrator.task import Task, TaskState, run_task


@click.group()
def app() -> None:
    """AMOP CLI."""


@app.command()
@click.option("--prompt", required=True, help="Prompt to send to the agent.")
@click.option(
    "--model", default=DEFAULT_MODEL, show_default=True, help="Ollama model to use."
)
def run(prompt: str, model: str) -> None:
    """Run a single prompt through the CoderAgent via a local Ollama model."""
    task = Task(id=str(uuid.uuid4()), prompt=prompt)
    provider = OllamaProvider(model=model)
    agent = CoderAgent(model=provider)

    result_task = asyncio.run(run_task(task, agent))

    if result_task.state == TaskState.DONE:
        click.echo(result_task.output)
    else:
        click.echo(f"Task failed: {result_task.error}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    app()
