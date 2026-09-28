"""Log/metric-based anomaly detection -- spec Section 6.1's second half,
never built until now. agents/watcher.py's own module docstring scoped
Milestone 9 down to GitHub issues only ("no production app is being
monitored yet, so log/metric sources ... are out of scope"). This module
is that scope, closed.

Section 8.7 names two possible backends: "Section 3.4's Prometheus, or
file-tail/syslog per original Section 7.3". Prometheus is not installed
or run anywhere in this project. amop/tools/monitoring.py's tools are
file-tail based instead -- the spec's own sanctioned fallback for a
setup without a real metrics stack, not an invented workaround.

Anomaly detection method, per 6.1: "threshold rules (config-defined,
e.g. error_rate_pct > 5 for 5m) run first and are cheap" -- statistical
(z-score) and ML (Isolation Forest) detection are explicitly later
passes and explicitly out of scope here, matching
AnomalyAlert.detection_method's own comment that non-threshold detection
stays post-MVP.

Deliberately NOT an LLM call: "error_rate_pct > 5 for 5m" is a numeric
comparison, not a judgment call -- WatcherAgent's LLM classification
pass exists for genuinely ambiguous input (is this GitHub issue text a
real bug report?), which this isn't. Same "code decides what's
mechanically knowable, the model only judges what's actually ambiguous"
split as AnomalyAlert.repo/github_issue_number already being stamped by
code rather than trusted from Watcher's own output (see that class's
docstring).

No `incidents`/`monitoring_state` tables (spec 14.2) -- same "reuse what
exists where reasonable" call Milestone 9 already made for GitHub-
sourced alerts. Dedup here reuses watch.py's existing
find_existing_task_for_issue against a synthetic, stable
github_issue_number (THRESHOLD_INCIDENT_ISSUE_NUMBER, always negative so
it can never collide with a real GitHub issue number), which keeps an
ongoing incident from spawning a new task every poll cycle while one is
already open, without a new table.
"""

import os
from collections.abc import Callable

from amop.agents.handoffs import AnomalyAlert, AnomalyEvidence
from amop.orchestrator.watch import find_existing_task_for_issue
from amop.tools.registry import ToolContext, invoke_tool

# Which single watched repo (as passed to `amop watch --repo`/registered
# slug) this log source describes. None (the default) disables
# log-based detection entirely -- every pre-existing `amop watch`
# invocation is completely unaffected unless this is explicitly set,
# same "off by default" pattern as every other env-var-gated addition
# in this codebase. One log source, not a per-repo registry: this MVP
# has exactly one real running monitored app (the demo fixture); a
# multi-repo log registry is a natural but unbuilt future extension,
# same shape as AMOP_WATCHER_SEVERITY_FLOOR being global today.
LOG_SOURCE_REPO = os.environ.get("AMOP_LOG_SOURCE_REPO")

DEFAULT_ERROR_RATE_THRESHOLD_PCT = 5.0
DEFAULT_ERROR_RATE_WINDOW_MINUTES = 5

ERROR_RATE_THRESHOLD_PCT = float(
    os.environ.get(
        "AMOP_ERROR_RATE_THRESHOLD_PCT", str(DEFAULT_ERROR_RATE_THRESHOLD_PCT)
    )
)
ERROR_RATE_WINDOW_MINUTES = float(
    os.environ.get(
        "AMOP_ERROR_RATE_WINDOW_MINUTES", str(DEFAULT_ERROR_RATE_WINDOW_MINUTES)
    )
)

# A synthetic, stable "incident id" for this one threshold rule, per
# repo -- see module docstring. Never a real GitHub issue number (those
# are always >= 1), so it can never collide with a genuine GitHub-
# sourced task's dedup key on the same repo.
THRESHOLD_INCIDENT_ISSUE_NUMBER = -1


def _noop(_message: str) -> None:
    pass


async def check_log_threshold(
    session,
    repo: str,
    ctx: ToolContext,
    *,
    threshold_pct: float = ERROR_RATE_THRESHOLD_PCT,
    window_minutes: float = ERROR_RATE_WINDOW_MINUTES,
    emit: Callable[[str], None] = _noop,
) -> AnomalyAlert | None:
    """One poll cycle's worth of the threshold rule (6.1). Returns a
    fully-stamped AnomalyAlert ready for triage_anomaly(), or None if
    nothing qualifies this cycle -- not an error; "no breach" and
    "already tracked" are both legitimate, common outcomes, same as
    Watcher emitting zero alerts for a batch of GitHub issues.

    A get_error_rate tool failure (missing/unreadable log source) is
    reported to `emit` and treated as "nothing this cycle", never
    raised -- the caller (cli/main.py's _poll_once) wraps this whole
    call in its own try/except anyway (defense in depth, matching every
    other tool call site in this codebase), but the actual 6.1 failure-
    handling contract ("a data-source read failure is logged and that
    source is skipped for the cycle") is satisfied right here, the same
    place the read happens.
    """
    result = await invoke_tool(
        "get_error_rate",
        {"window_minutes": window_minutes},
        ctx,
        agent_name="orchestrator",
    )
    if not result.success:
        emit(f"logs: get_error_rate failed ({result.error_code}): {result.message}")
        return None

    rate = result.output["error_rate_pct"]
    total = result.output["total"]
    if total == 0:
        emit(f"logs: 0 lines in the last {window_minutes}m -- nothing to evaluate")
        return None
    if rate <= threshold_pct:
        emit(f"logs: error_rate_pct={rate:.2f} (threshold {threshold_pct:.2f}) -- OK")
        return None

    existing = await find_existing_task_for_issue(
        session, repo, THRESHOLD_INCIDENT_ISSUE_NUMBER
    )
    if existing is not None:
        emit(
            f"logs: error_rate_pct={rate:.2f} still above threshold, but already "
            f"tracked as task {existing.id} (state={existing.state})"
        )
        return None

    if rate >= threshold_pct * 3:
        severity = "critical"
    elif rate >= threshold_pct * 2:
        severity = "high"
    else:
        severity = "medium"

    return AnomalyAlert(
        source="logs",
        severity=severity,
        summary=(
            f"Error rate {rate:.1f}% over the last {window_minutes:g}m exceeds the "
            f"configured {threshold_pct:.1f}% threshold ({total} log lines, "
            f"{result.output['errors']} errors)"
        ),
        evidence=[
            AnomalyEvidence(
                type="metric_point",
                ref=(
                    f"error_rate_pct={rate:.2f} window_minutes={window_minutes:g} "
                    f"threshold_pct={threshold_pct:.2f}"
                ),
            )
        ],
        detection_method="threshold",
        confidence=1.0,
        issue_index=0,  # unused for this source -- see AnomalyAlert docstring
        repo=repo,
        github_issue_number=THRESHOLD_INCIDENT_ISSUE_NUMBER,
    )
