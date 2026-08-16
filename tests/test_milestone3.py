"""Milestone 3 — Docker Sandbox. Section 9.1-9.3, 9.6, 9.7.

These tests require a real, running Docker daemon (`docker ps` must
succeed) -- there is no mock/fake sandbox path. The whole point of this
milestone is proving isolation actually holds, not that the code *would*
enforce it if the mechanism worked as assumed (see manager.py's exec_run
docstring and the Dockerfile's procps comment for a case where an
untested assumption -- that a bare `kill` binary exists -- was wrong and
silently no-op'd; these tests exist so that class of bug gets caught
here, not in production).

A fresh container is created per test (not shared) so a failing
assertion can't leave a later test looking at stale state; `sandbox_manager`
tears down anything left running via destroy_all() even on failure.
"""

import os
import time
import uuid

import docker
import pytest

from amop.agents.coder import CoderAgent  # also registers sandboxed read_file/write_file on import
from amop.models.base import BaseLLM, ModelResponse
from amop.sandbox.manager import PIDS_LIMIT, SandboxManager
from amop.tools.registry import ToolContext, invoke_tool

# No `pytestmark = pytest.mark.asyncio` needed -- pyproject.toml sets
# asyncio_mode = "auto", which already treats every `async def test_*`
# below as an asyncio test without a per-test or module-level marker.


@pytest.fixture
def sandbox_manager():
    manager = SandboxManager()
    yield manager
    manager.destroy_all()  # best-effort cleanup even if a test's own destroy() was skipped by a failure


def make_ctx(scratch_dir, sandbox, mode="suggestor"):
    return ToolContext(agent_name="test", scratch_dir=scratch_dir, mode=mode, sandbox=sandbox)


# ---------------------------------------------------------------------
# Core isolation proof (Done-When #1): a container write is genuinely
# separate from the host, not just written to a differently-named host
# folder. Two directions: the ONE intentional shared boundary
# (/workspace) really is shared, and everything outside it really isn't.
# ---------------------------------------------------------------------


def test_workspace_mount_is_genuinely_shared(tmp_path, sandbox_manager):
    """The one deliberate exception: /workspace is bind-mounted from the
    host scratch dir, so a write there IS expected to appear on the host
    -- this is the control case the isolation tests below are contrasted
    against, not itself a leak."""
    sandbox = sandbox_manager.create("t-shared-mount", tmp_path)
    try:
        sandbox.write_file("/workspace/proof.txt", "shared-mount-proof")
        host_file = tmp_path / "proof.txt"
        assert host_file.exists()
        assert host_file.read_text() == "shared-mount-proof"
    finally:
        sandbox_manager.destroy("t-shared-mount")


def test_container_write_outside_workspace_never_reaches_host(tmp_path, sandbox_manager):
    """The real isolation proof: write a uniquely-named file *outside*
    /workspace (the container's own /tmp), confirm the write really
    happened (readable from inside), then confirm the exact same path
    does NOT exist on the host -- the container's non-workspace
    filesystem is its own ephemeral layer, not a view onto the host's."""
    sandbox = sandbox_manager.create("t-write-escape", tmp_path)
    try:
        nonce = uuid.uuid4().hex
        container_path = f"/tmp/host_escape_{nonce}.txt"
        sandbox.exec_run(f"echo leaked > {container_path}")

        inside = sandbox.exec_run(f"cat {container_path}")
        assert inside.stdout.strip() == "leaked"  # the write is real

        assert not os.path.exists(container_path)  # same path, host side: absent
    finally:
        sandbox_manager.destroy("t-write-escape")


def test_container_cannot_read_a_real_host_secret(tmp_path, sandbox_manager):
    """Reverse direction: a file that indisputably exists on the host,
    at a path outside the scratch dir, must be unreadable from inside
    the container -- proving host secrets don't leak through even when
    the container asks for them by their real absolute path."""
    host_secret_dir = tmp_path.parent / f"host-secret-{tmp_path.name}"
    host_secret_dir.mkdir()
    secret_path = host_secret_dir / "secret.txt"
    secret_path.write_text("do-not-leak-me")
    assert secret_path.exists()  # sanity: it's real, on the host

    scratch = tmp_path / "scratch"
    sandbox = sandbox_manager.create("t-host-secret", scratch)
    try:
        result = sandbox.exec_run(f"cat {secret_path} 2>&1; echo EXIT:$?")
        assert "do-not-leak-me" not in result.stdout
        assert "EXIT:0" not in result.stdout  # cat did not succeed
    finally:
        sandbox_manager.destroy("t-host-secret")


def test_network_is_disabled(tmp_path, sandbox_manager):
    """Section 9.3's v1 simplification: network_mode=none. An
    agent-initiated outbound connection must fail at the network layer,
    not merely be discouraged by a prompt."""
    sandbox = sandbox_manager.create("t-network", tmp_path)
    try:
        result = sandbox.exec_run(
            "python3 -c \"import socket; "
            "socket.create_connection(('8.8.8.8', 53), timeout=3)\" "
            ">/tmp/out 2>&1; echo EXIT:$?; cat /tmp/out",
            timeout=10,
        )
        assert "EXIT:0" not in result.stdout
        assert "Network is unreachable" in result.stdout or "unreachable" in result.stdout.lower()
    finally:
        sandbox_manager.destroy("t-network")


# ---------------------------------------------------------------------
# Timeout handling (Section 9.6, 9.7.1)
# ---------------------------------------------------------------------


def test_timeout_kills_the_process_not_just_the_client_wait(tmp_path, sandbox_manager):
    sandbox = sandbox_manager.create("t-timeout", tmp_path)
    try:
        t0 = time.monotonic()
        result = sandbox.exec_run("sleep 30", timeout=2)
        elapsed = time.monotonic() - t0

        assert result.timed_out is True
        assert elapsed < 15  # well under the full 30s sleep -- bounded by timeout + grace, not the command's own duration

        # Server-side proof the process is actually gone, not just that
        # our client gave up waiting on it.
        check = sandbox._raw_exec(["pgrep", "-af", "sleep"])
        assert check.exit_code != 0, f"sleep still running: {check.stdout}"
    finally:
        sandbox_manager.destroy("t-timeout")


def test_command_within_timeout_completes_normally(tmp_path, sandbox_manager):
    sandbox = sandbox_manager.create("t-no-timeout", tmp_path)
    try:
        result = sandbox.exec_run("echo done", timeout=5)
        assert not result.timed_out
        assert result.exit_code == 0
        assert "done" in result.stdout
    finally:
        sandbox_manager.destroy("t-no-timeout")


# ---------------------------------------------------------------------
# Resource limits (Section 9.3): pids_limit actually caps process
# creation inside the container, and the host is unaffected.
# ---------------------------------------------------------------------


def test_pids_limit_caps_process_creation(tmp_path, sandbox_manager):
    host_procs_before = len(os.listdir("/proc")) if os.path.isdir("/proc") else None

    sandbox = sandbox_manager.create("t-pids-limit", tmp_path)
    try:
        result = sandbox.exec_run(
            "i=0; ok=0; fail=0; "
            f"while [ $i -lt $(({PIDS_LIMIT} * 3)) ]; do "
            "  if sh -c 'sleep 5 &' 2>/dev/null; then ok=$((ok+1)); else fail=$((fail+1)); fi; "
            "  i=$((i+1)); "
            "done; "
            "echo ok=$ok fail=$fail",
            timeout=60,
        )
        assert not result.timed_out
        out = result.stdout.strip()
        ok = int(out.split("ok=")[1].split()[0])
        fail = int(out.split("fail=")[1].split()[0])

        # The cap is real: creation attempts past it fail, well short of
        # the 3x-over-limit total we tried.
        assert ok <= PIDS_LIMIT
        assert fail > 0

        if host_procs_before is not None:
            host_procs_after = len(os.listdir("/proc"))
            # Host process table did not explode in sympathy with the
            # container's attempted fork storm.
            assert host_procs_after - host_procs_before < 50
    finally:
        sandbox_manager.destroy("t-pids-limit")


# ---------------------------------------------------------------------
# Cleanup (Section 9.7)
# ---------------------------------------------------------------------


def test_container_removed_from_docker_after_destroy(tmp_path, sandbox_manager):
    sandbox = sandbox_manager.create("t-cleanup", tmp_path)
    container_id = sandbox.id

    client = docker.from_env()
    assert client.containers.list(all=True, filters={"id": container_id})

    sandbox_manager.destroy("t-cleanup")

    assert client.containers.list(all=True, filters={"id": container_id}) == []


def test_expired_sandbox_is_destroyed_on_next_get(tmp_path, sandbox_manager):
    """Section 9.1's max-lifetime cap, exercised directly rather than by
    actually sleeping 600s: backdate the sandbox's own clock past the
    configured lifetime and confirm get() destroys it as a side effect,
    without needing task completion."""
    sandbox = sandbox_manager.create("t-lifetime", tmp_path)
    container_id = sandbox.id
    sandbox.created_at -= 100_000  # force is_expired() True regardless of MAX_LIFETIME_SECONDS

    assert sandbox_manager.get("t-lifetime") is None

    client = docker.from_env()
    assert client.containers.list(all=True, filters={"id": container_id}) == []


# ---------------------------------------------------------------------
# Tool-level: sandbox/tools.py's read_file/write_file through the full
# invoke_tool() pipeline (Safety Engine gate still runs first, unchanged)
# ---------------------------------------------------------------------


async def test_write_then_read_inside_scratch_dir_via_sandbox(tmp_path, sandbox_manager):
    sandbox = sandbox_manager.create("t-tool-io", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        write_result = await invoke_tool(
            "write_file", {"path": "note.txt", "content": "sandboxed write"}, ctx, agent_name="test"
        )
        assert write_result.success

        # Really landed in the container's /workspace, i.e. on the host
        # mount -- not silently written somewhere else.
        assert (tmp_path / "note.txt").read_text() == "sandboxed write"

        read_result = await invoke_tool("read_file", {"path": "note.txt"}, ctx, agent_name="test")
        assert read_result.success
        assert read_result.output == "sandboxed write"
    finally:
        sandbox_manager.destroy("t-tool-io")


async def test_write_outside_scratch_dir_denied_before_reaching_sandbox(tmp_path, sandbox_manager):
    """Safety Engine denial happens before invoke_tool ever calls into
    sandbox/tools.py's body -- proven here by denying with sandbox=None,
    i.e. even without a live container the call is rejected at the gate,
    same as Milestone 2."""
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    ctx = ToolContext(agent_name="test", scratch_dir=tmp_path, mode="suggestor", sandbox=None)

    result = await invoke_tool(
        "write_file", {"path": f"../{outside.name}/leaked.txt", "content": "leaked"}, ctx, agent_name="test"
    )

    assert not result.success
    assert result.error_code == "DENIED"
    assert not outside.exists()


# ---------------------------------------------------------------------
# End-to-end: CoderAgent's real tool loop, routed through a real
# container (Done-When #7: Milestone 2 behavior still works, now
# sandboxed)
# ---------------------------------------------------------------------


class ScriptedLLM(BaseLLM):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, messages, tools=None) -> ModelResponse:
        content = self._responses[self.calls]
        self.calls += 1
        return ModelResponse(content=content, model="scripted")

    async def embed(self, texts):
        raise NotImplementedError


async def test_agent_write_is_routed_through_a_real_container(tmp_path):
    import json

    ctx = ToolContext(agent_name="coder", scratch_dir=tmp_path, mode="suggestor")
    responses = [
        json.dumps(
            {"tool_call": {"name": "write_file", "arguments": {"path": "hello.py", "content": "print('hi')"}}}
        ),
        json.dumps({"final_answer": "wrote hello.py"}),
    ]
    agent = CoderAgent(model=ScriptedLLM(responses), ctx=ctx, task_id="t-agent-e2e")

    result = await agent.run("write hello.py")

    assert result.success
    assert result.tool_calls[0]["success"] is True
    assert (tmp_path / "hello.py").read_text() == "print('hi')"
    assert agent.last_container_id is not None

    # Container is torn down when run() returns (task-end destroy).
    client = docker.from_env()
    still_there = client.containers.list(filters={"label": f"amop.task_id=t-agent-e2e"})
    assert still_there == []
