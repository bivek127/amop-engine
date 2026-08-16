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
RUN apt-get update && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

# Non-root user inside the container (Section 9.3's isolation table).
# Fixed uid so host-side code (sandbox/manager.py's CONTAINER_UID) can
# chown file writes to match without querying the image at runtime.
RUN useradd --uid 1000 --create-home --shell /bin/sh amop

WORKDIR /workspace
RUN chown amop:amop /workspace

USER amop

# Kept alive with a no-op process so the Sandbox Manager can `exec` into
# the same container repeatedly across a task's tool calls (Section
# 9.1: one container per task, not per call) instead of spinning up a
# fresh container per invocation.
CMD ["tail", "-f", "/dev/null"]
