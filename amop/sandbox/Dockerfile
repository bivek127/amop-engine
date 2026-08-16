# Base sandbox image — spec Section 9.2, trimmed to the single Python
# image this milestone needs (per-stack images beyond Python are later
# work, explicitly out of scope for Milestone 3).
FROM python:3.12-slim

# procps: python:3.12-slim has no standalone `kill`/`ps`/`pgrep` binaries
# (only sh's `kill` builtin, which exec-by-argv-list can't reach without
# a shell). sandbox/manager.py's exec_run() timeout path signals whole
# process groups from *outside* a shell context, so it needs the real
# binary -- this was verified against a running container, not assumed
# (a no-op `kill` here silently fails to kill anything and only looks
# like it worked because the exit code is swallowed).
#
# git (Milestone 4): Section 9.2's tool-belt. The chain branches, commits
# and diffs inside the container (sandbox/tools.py's get_diff, the
# orchestrator's baseline commit) -- all of which need the real binary.
RUN apt-get update && apt-get install -y --no-install-recommends procps git \
    && rm -rf /var/lib/apt/lists/*

# pytest (Milestone 4): baked into the image rather than installed per
# task, because containers run with network_mode="none" (Section 9.3) --
# there is no pip-install path from inside a live sandbox, so anything
# run_tests needs has to already be here.
RUN pip install --no-cache-dir pytest

# Non-root user inside the container (Section 9.3's isolation table).
# Fixed uid so host-side code (sandbox/manager.py's CONTAINER_UID) can
# chown file writes to match without querying the image at runtime.
RUN useradd --uid 1000 --create-home --shell /bin/sh amop

WORKDIR /workspace
RUN chown amop:amop /workspace

USER amop

# Git identity + defaults for the sandbox user (Milestone 4). A commit
# with no configured identity fails outright, and there is no interactive
# way to supply one inside a network-less container, so it's baked in.
# safe.directory: /workspace is a bind mount whose host-side ownership
# won't match uid 1000, which otherwise trips git's "dubious ownership"
# refusal before any command runs.
RUN git config --global user.email "agent@amop.local" \
    && git config --global user.name "AMOP Agent" \
    && git config --global init.defaultBranch main \
    && git config --global --add safe.directory /workspace

# Kept alive with a no-op process so the Sandbox Manager can `exec` into
# the same container repeatedly across a task's tool calls (Section
# 9.1: one container per task, not per call) instead of spinning up a
# fresh container per invocation.
CMD ["tail", "-f", "/dev/null"]
