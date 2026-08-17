"""Sandbox Manager — spec Section 9.1-9.3, 9.6, 9.7. Ephemeral Docker
container per task; Milestone 3 moves read_file/write_file execution off
the host filesystem and into this container. The Safety Engine
(Milestone 2, safety/engine.py) still gates every call BEFORE it reaches
here — this is a second, independent isolation layer, not a replacement
for the permission check.

v1 simplifications, both explicitly sanctioned by CLAUDE.md's Milestone 3
section over spec Section 9.1/9.3's own stated defaults — documented
here, not silently narrowed:
  - network_mode="none": spec 9.3 wants an egress allowlist (package
    registries + the repo's own git remote); a real allowlist proxy is
    Section 29.2, later work. Full network shutdown is simpler and
    sufficient while the only sandboxed tools are file I/O with no need
    to reach the network at all.
  - Resource defaults (1 CPU / 1024MB, pids_limit 128, 600s max
    lifetime) are the milestone doc's stated numbers, below spec
    9.1/9.3's own defaults (2 CPU / 2048MB, pids_limit 256, 3600s). Same
    config keys either way, so raising them later is a config change,
    not a code change.

Also out of scope this milestone (CLAUDE.md's "What NOT to Build"): the
egress allowlist proxy, per-stack images beyond Python (9.2), dependency
caching (9.5), and Section 9.7.1's periodic orphan-container sweep timer
— containers are still labeled `amop.task_id` so that sweep can be added
later without a data-model change, but no background timer runs yet.
"""

import io
import os
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import docker
import docker.errors
from docker.models.containers import Container

IMAGE_NAME = os.environ.get("AMOP_SANDBOX_IMAGE", "amop-sandbox:latest")
DOCKERFILE_DIR = Path(__file__).parent

CPU_LIMIT = float(os.environ.get("AMOP_SANDBOX_CPU_LIMIT", "1"))
MEMORY_LIMIT_MB = int(os.environ.get("AMOP_SANDBOX_MEMORY_LIMIT_MB", "1024"))
PIDS_LIMIT = int(os.environ.get("AMOP_SANDBOX_PIDS_LIMIT", "128"))
MAX_LIFETIME_SECONDS = int(os.environ.get("AMOP_SANDBOX_MAX_LIFETIME_SECONDS", "600"))

CONTAINER_UID = 1000
TASK_LABEL = "amop.task_id"

# Grace period between SIGTERM and SIGKILL on a timed-out exec (Section
# 9.6/9.7.1's "SIGTERM, 5s grace, then SIGKILL").
_KILL_GRACE_SECONDS = 5.0


class SandboxError(Exception):
    """Sandbox-layer failure (container create/exec plumbing), distinct
    from a tool's own logic error. Callers normalize this into a
    ToolResult, same as any other tool-body exception."""


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class Sandbox:
    """One task's ephemeral container. Constructed only by
    SandboxManager.create() — never directly."""

    def __init__(self, container: Container, task_id: str) -> None:
        self.container = container
        self.task_id = task_id
        self.created_at = time.monotonic()

    @property
    def id(self) -> str:
        return self.container.id

    @property
    def short_id(self) -> str:
        return self.container.short_id

    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > MAX_LIFETIME_SECONDS

    # -- file I/O -----------------------------------------------------
    # put_archive/get_archive are tar-stream APIs, not a shelled-out
    # command -- arbitrary file content never passes through a shell, so
    # there is no quoting to get wrong here.

    def write_file(self, container_path: str, content: str, mode: int = 0o644) -> None:
        container_path = Path(container_path)
        parent, name = str(container_path.parent), container_path.name
        self._raw_exec(["mkdir", "-p", parent])

        data = content.encode("utf-8")
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tar:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = int(time.time())
            info.uid = CONTAINER_UID
            info.gid = CONTAINER_UID
            info.mode = mode
            tar.addfile(info, io.BytesIO(data))
        tar_buf.seek(0)
        self.container.put_archive(parent, tar_buf.getvalue())

    def read_file(self, container_path: str) -> str:
        container_path = Path(container_path)
        try:
            stream, _stat = self.container.get_archive(str(container_path))
        except docker.errors.NotFound as exc:
            raise FileNotFoundError(str(container_path)) from exc

        tar_buf = io.BytesIO()
        for chunk in stream:
            tar_buf.write(chunk)
        tar_buf.seek(0)

        with tarfile.open(fileobj=tar_buf, mode="r") as tar:
            member = tar.getmember(container_path.name)
            extracted = tar.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(str(container_path))
            return extracted.read().decode("utf-8")

    def stat_size(self, container_path: str) -> int:
        """Byte count via `wc -c`, not a get_archive/tar-extract round
        trip -- Milestone 11: read_file needs to decide whether a file
        is too large to return whole *before* paying to fetch and
        decode all of it, since for a genuinely huge file that content
        would just get thrown away in favor of a scoped default anyway
        (confirmed live: sending the full content with a warning notice
        attached doesn't work -- the model's own context truncation
        drops the notice's instruction along with everything else before
        it ever gets a chance to act on it, so the fix has to be not
        sending the whole thing in the first place, not asking nicely)."""
        result = self._raw_exec(["wc", "-c", str(container_path)])
        if result.exit_code != 0:
            raise FileNotFoundError(str(container_path))
        # "  12345 /workspace/path\n" -- first whitespace-separated token.
        return int(result.stdout.split()[0])

    # -- command execution ---------------------------------------------

    def _raw_exec(self, argv: list[str]) -> ExecResult:
        """Run an internal plumbing command (mkdir, cat, kill, rm — all
        argv-list, never a shell string) as the non-root sandbox user, no
        timeout handling. Used for Sandbox's own bookkeeping, not for
        untrusted agent-supplied commands."""
        api = self.container.client.api
        exec_id = api.exec_create(
            self.container.id, argv, user=str(CONTAINER_UID), stdout=True, stderr=True
        )["Id"]
        out = api.exec_start(exec_id, demux=True)
        stdout_b, stderr_b = out if isinstance(out, tuple) else (out, b"")
        inspect = api.exec_inspect(exec_id)
        return ExecResult(
            exit_code=inspect.get("ExitCode") if inspect.get("ExitCode") is not None else -1,
            stdout=(stdout_b or b"").decode("utf-8", errors="replace"),
            stderr=(stderr_b or b"").decode("utf-8", errors="replace"),
        )

    def exec_run(self, command: str, timeout: float | None = None) -> ExecResult:
        """Run a shell `command` inside the container (Section 9.6/9.7.1).

        The command is written to a script file via put_archive (again:
        no shell-string interpolation of `command` itself) and started
        under `setsid -w` so it becomes the leader of its own new process
        group -- its pid IS its pgid. The script's first line records
        that pid to a file before running anything else, so a timeout can
        target the *entire* group (kill(-pgid, ...)), not just the direct
        child -- a background-spawning command would otherwise survive
        its own parent's death (Section 9.7.1's orphaned-process leak).

        On timeout: SIGTERM to the group, 5s grace, then SIGKILL --
        Section 9.6's contract exactly. `timed_out` is reported back as a
        normal ExecResult field, not raised, matching Section 9.6's "fed
        back to the agent as a normal (if unwelcome) observation, not a
        task-ending crash."
        """
        api = self.container.client.api
        marker = f"amop-exec-{uuid.uuid4().hex}"
        script_path = f"/tmp/{marker}.sh"
        pid_path = f"/tmp/{marker}.pid"

        script = f"#!/bin/sh\necho $$ > {pid_path}\n{command}\n"
        self.write_file(script_path, script, mode=0o755)

        exec_id = api.exec_create(
            self.container.id,
            ["setsid", "-w", "sh", script_path],
            user=str(CONTAINER_UID),
            stdout=True,
            stderr=True,
        )["Id"]

        holder: dict = {}

        def _run() -> None:
            holder["output"] = api.exec_start(exec_id, demux=True)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout)

        timed_out = False
        if thread.is_alive():
            timed_out = True
            pgid = self._read_pgid(pid_path)
            if pgid is not None:
                self._raw_exec(["kill", "-TERM", f"-{pgid}"])
                thread.join(_KILL_GRACE_SECONDS)
                if thread.is_alive():
                    self._raw_exec(["kill", "-KILL", f"-{pgid}"])
                    thread.join(2)

        inspect = api.exec_inspect(exec_id)
        exit_code = inspect.get("ExitCode")
        stdout_b, stderr_b = holder.get("output", (b"", b""))
        self._raw_exec(["rm", "-f", script_path, pid_path])

        return ExecResult(
            exit_code=exit_code if exit_code is not None else -1,
            stdout=(stdout_b or b"").decode("utf-8", errors="replace"),
            stderr=(stderr_b or b"").decode("utf-8", errors="replace"),
            timed_out=timed_out,
        )

    def _read_pgid(self, pid_path: str) -> int | None:
        result = self._raw_exec(["cat", pid_path])
        text = result.stdout.strip()
        return int(text) if text.isdigit() else None


class SandboxManager:
    """One Sandbox per task, created at task-session start and destroyed
    at session end (Section 9.1: Create -> ... -> Destroy), reused across
    every tool call within that task (not per call)."""

    def __init__(self) -> None:
        self._client = docker.from_env()
        self._sandboxes: dict[str, Sandbox] = {}
        self._ensure_image()

    def _ensure_image(self) -> None:
        try:
            self._client.images.get(IMAGE_NAME)
        except docker.errors.ImageNotFound:
            self._client.images.build(path=str(DOCKERFILE_DIR), tag=IMAGE_NAME, rm=True)

    def create(self, task_id: str, host_scratch_dir: Path) -> Sandbox:
        host_scratch_dir = Path(host_scratch_dir)
        host_scratch_dir.mkdir(parents=True, exist_ok=True)
        # Bind mounts on Docker Desktop don't reliably line up with the
        # container's fixed uid 1000 against the host's own uid -- open
        # permissions on this scratch-only directory side-step that
        # rather than trying to make numeric uids match across the VM
        # boundary.
        os.chmod(host_scratch_dir, 0o777)

        container = self._client.containers.run(
            IMAGE_NAME,
            command=["tail", "-f", "/dev/null"],
            detach=True,
            user=str(CONTAINER_UID),
            working_dir="/workspace",
            volumes={str(host_scratch_dir): {"bind": "/workspace", "mode": "rw"}},
            network_mode="none",
            nano_cpus=int(CPU_LIMIT * 1_000_000_000),
            mem_limit=f"{MEMORY_LIMIT_MB}m",
            pids_limit=PIDS_LIMIT,
            cap_drop=["ALL"],
            init=True,
            labels={TASK_LABEL: task_id},
        )
        sandbox = Sandbox(container, task_id)
        self._sandboxes[task_id] = sandbox
        return sandbox

    def get(self, task_id: str) -> Sandbox | None:
        """Returns the task's live sandbox, or None if there isn't one or
        it has outlived sandbox.max_lifetime_seconds (Section 9.1) -- in
        the latter case it's destroyed as a side effect of this check."""
        sandbox = self._sandboxes.get(task_id)
        if sandbox is None:
            return None
        if sandbox.is_expired():
            self.destroy(task_id)
            return None
        return sandbox

    def destroy(self, task_id: str) -> None:
        sandbox = self._sandboxes.pop(task_id, None)
        if sandbox is None:
            return
        try:
            sandbox.container.remove(force=True)
        except docker.errors.NotFound:
            pass

    def destroy_all(self) -> None:
        for task_id in list(self._sandboxes):
            self.destroy(task_id)
