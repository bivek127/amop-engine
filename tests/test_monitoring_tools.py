"""Tests for the log/metric-based anomaly detection built here: spec
8.7's four monitoring tools (amop/tools/monitoring.py) and spec 6.1's
threshold-rule wiring into Watcher's detection cycle
(amop/orchestrator/monitoring.py).

Two tiers, same discipline as this project's other milestones:
  * Deterministic tool tests: real log files on disk (tmp_path), no DB,
    no LLM -- pure parsing/windowing/rate math.
  * check_log_threshold / triage_anomaly integration: real Postgres
    (same session fixture as test_milestone9.py), no LLM -- the
    threshold rule is deliberately not an LLM call (see
    orchestrator/monitoring.py's module docstring).

The real, live end-to-end demo (a real running tests/fixtures/
monitored_service/app.py, discovered by a real `amop watch` poll loop)
is a CLI demo, not a pytest tier -- same split test_milestone9.py's own
docstring already draws for the GitHub-issue path.

Requires: Postgres at TEST_DATABASE_URL for the integration tests below.
The tool-level tests need no external services.
"""

import os
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text

from amop.agents.handoffs import AnomalyAlert
from amop.database.session import init_db, make_engine, make_session_factory
from amop.orchestrator import monitoring as mon_mod
from amop.orchestrator import watch as watch_mod
from amop.orchestrator.task import create_task
from amop.tools import monitoring as tools
from amop.tools.registry import ToolContext

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://localhost/amop_test"
)


def _ctx(scratch_dir: Path) -> ToolContext:
    return ToolContext(agent_name="test", scratch_dir=scratch_dir, mode="observer")


def _write_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# amop/tools/monitoring.py -- pure file-tail parsing/windowing
# ---------------------------------------------------------------------


async def test_get_error_rate_computes_real_percentage(tmp_path, monkeypatch):
    log = tmp_path / "app.log"
    now = "2026-09-28T12:00:00.000Z"
    _write_log(
        log,
        [
            f"{now} INFO order 1 processed avg_item_price=5.00",
            f"{now} INFO order 2 processed avg_item_price=5.00",
            f"{now} ERROR order 3 failed: division by zero",
            f"{now} INFO order 4 processed avg_item_price=5.00",
        ],
    )
    monkeypatch.setenv("AMOP_LOG_SOURCE_PATH", str(log))
    # Freeze "now" by using a huge window so the fixed 2026 timestamps
    # above always fall inside it, regardless of when the test runs.
    result = await tools.get_error_rate(_ctx(tmp_path), window_minutes=10_000_000)
    assert result.success
    assert result.output["total"] == 4
    assert result.output["errors"] == 1
    assert result.output["error_rate_pct"] == pytest.approx(25.0)


async def test_get_error_rate_zero_lines_is_zero_not_error(tmp_path, monkeypatch):
    log = tmp_path / "app.log"
    _write_log(log, ["2000-01-01T00:00:00.000Z INFO ancient line"])
    monkeypatch.setenv("AMOP_LOG_SOURCE_PATH", str(log))
    result = await tools.get_error_rate(_ctx(tmp_path), window_minutes=1)
    assert result.success
    assert result.output["total"] == 0
    assert result.output["error_rate_pct"] == 0.0


async def test_get_error_rate_unavailable_when_source_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("AMOP_LOG_SOURCE_PATH", raising=False)
    result = await tools.get_error_rate(_ctx(tmp_path))
    assert not result.success
    assert result.error_code == "LOG_SOURCE_UNAVAILABLE"
    # This must never be readable as "no errors" -- see module docstring.
    assert "NOT a statement" in result.message


async def test_get_error_rate_unavailable_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("AMOP_LOG_SOURCE_PATH", str(tmp_path / "does_not_exist.log"))
    result = await tools.get_error_rate(_ctx(tmp_path))
    assert not result.success
    assert result.error_code == "LOG_SOURCE_UNAVAILABLE"


async def test_search_logs_matches_case_insensitively(tmp_path, monkeypatch):
    log = tmp_path / "app.log"
    now = "2026-09-28T12:00:00.000Z"
    _write_log(
        log,
        [
            f"{now} ERROR order 3 failed: Division By Zero",
            f"{now} INFO order 4 processed avg_item_price=5.00",
        ],
    )
    monkeypatch.setenv("AMOP_LOG_SOURCE_PATH", str(log))
    result = await tools.search_logs("division by zero", _ctx(tmp_path), since_minutes=10_000_000)
    assert result.success
    assert len(result.output["matches"]) == 1
    assert result.output["matches"][0]["level"] == "ERROR"


async def test_list_recent_errors_returns_newest_first_and_respects_limit(tmp_path, monkeypatch):
    log = tmp_path / "app.log"
    _write_log(
        log,
        [
            "2026-09-28T12:00:00.000Z ERROR first",
            "2026-09-28T12:00:02.000Z ERROR second",
            "2026-09-28T12:00:01.000Z ERROR third",
            "2026-09-28T12:00:00.000Z INFO not an error",
        ],
    )
    monkeypatch.setenv("AMOP_LOG_SOURCE_PATH", str(log))
    result = await tools.list_recent_errors(_ctx(tmp_path), limit=2)
    assert result.success
    messages = [e["message"] for e in result.output]
    assert messages == ["second", "third"]


async def test_get_metrics_counts_by_level(tmp_path, monkeypatch):
    log = tmp_path / "app.log"
    now = "2026-09-28T12:00:00.000Z"
    _write_log(
        log,
        [
            f"{now} INFO a",
            f"{now} WARNING b",
            f"{now} ERROR c",
            f"{now} ERROR d",
        ],
    )
    monkeypatch.setenv("AMOP_LOG_SOURCE_PATH", str(log))
    result = await tools.get_metrics(_ctx(tmp_path), window_minutes=10_000_000)
    assert result.success
    assert result.output["by_level"] == {"INFO": 1, "WARNING": 1, "ERROR": 2}
    assert result.output["total_lines"] == 4
    assert result.output["error_rate_pct"] == pytest.approx(50.0)


async def test_unparseable_lines_are_skipped_not_fatal(tmp_path, monkeypatch):
    log = tmp_path / "app.log"
    _write_log(
        log,
        [
            "",
            "this is not a log line at all",
            f"2026-09-28T12:00:00.000Z ERROR real error",
        ],
    )
    monkeypatch.setenv("AMOP_LOG_SOURCE_PATH", str(log))
    result = await tools.get_error_rate(_ctx(tmp_path), window_minutes=10_000_000)
    assert result.success
    assert result.output["total"] == 1
    assert result.output["errors"] == 1


# ---------------------------------------------------------------------
# amop/orchestrator/monitoring.py -- threshold rule + real dedup/TRIAGING
# ---------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine():
    eng = make_engine(TEST_DATABASE_URL)
    await init_db(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    session_factory = make_session_factory(engine)
    async with session_factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE task_transitions, tasks RESTART IDENTITY CASCADE"))


async def test_check_log_threshold_returns_none_below_threshold(tmp_path, monkeypatch, session):
    log = tmp_path / "app.log"
    now = "2026-09-28T12:00:00.000Z"
    _write_log(log, [f"{now} INFO ok" for _ in range(100)])
    monkeypatch.setenv("AMOP_LOG_SOURCE_PATH", str(log))
    alert = await mon_mod.check_log_threshold(
        session, "acme/widgets", _ctx(tmp_path), threshold_pct=5.0, window_minutes=10_000_000
    )
    assert alert is None


async def test_check_log_threshold_fires_above_threshold(tmp_path, monkeypatch, session):
    log = tmp_path / "app.log"
    now = "2026-09-28T12:00:00.000Z"
    lines = [f"{now} INFO ok" for _ in range(9)] + [f"{now} ERROR bad"]
    _write_log(log, lines)  # 10% error rate
    monkeypatch.setenv("AMOP_LOG_SOURCE_PATH", str(log))

    alert = await mon_mod.check_log_threshold(
        session, "acme/widgets", _ctx(tmp_path), threshold_pct=5.0, window_minutes=10_000_000
    )
    assert alert is not None
    assert isinstance(alert, AnomalyAlert)
    assert alert.source == "logs"
    assert alert.detection_method == "threshold"
    assert alert.confidence == 1.0
    assert alert.repo == "acme/widgets"
    assert alert.github_issue_number == mon_mod.THRESHOLD_INCIDENT_ISSUE_NUMBER
    assert alert.severity == "high"  # 10% is exactly 2x a 5% threshold


async def test_check_log_threshold_severity_scales_with_breach_size(tmp_path, monkeypatch, session):
    log = tmp_path / "app.log"
    now = "2026-09-28T12:00:00.000Z"
    lines = [f"{now} INFO ok" for _ in range(4)] + [f"{now} ERROR bad" for _ in range(6)]
    _write_log(log, lines)  # 60% error rate, threshold 5% -> >= 3x -> critical
    monkeypatch.setenv("AMOP_LOG_SOURCE_PATH", str(log))

    alert = await mon_mod.check_log_threshold(
        session, "acme/widgets", _ctx(tmp_path), threshold_pct=5.0, window_minutes=10_000_000
    )
    assert alert.severity == "critical"


async def test_check_log_threshold_skips_when_incident_already_tracked(tmp_path, monkeypatch, session):
    log = tmp_path / "app.log"
    now = "2026-09-28T12:00:00.000Z"
    _write_log(log, [f"{now} ERROR bad"])  # 100% error rate
    monkeypatch.setenv("AMOP_LOG_SOURCE_PATH", str(log))

    await create_task(
        session,
        task_type="bug_fix",
        task_context={
            "repo": "acme/widgets",
            "github_issue_number": mon_mod.THRESHOLD_INCIDENT_ISSUE_NUMBER,
            "source": "logs_watcher",
        },
    )

    alert = await mon_mod.check_log_threshold(
        session, "acme/widgets", _ctx(tmp_path), threshold_pct=5.0, window_minutes=10_000_000
    )
    assert alert is None


async def test_check_log_threshold_read_failure_returns_none_not_raise(tmp_path, monkeypatch, session):
    monkeypatch.delenv("AMOP_LOG_SOURCE_PATH", raising=False)
    alert = await mon_mod.check_log_threshold(session, "acme/widgets", _ctx(tmp_path))
    assert alert is None


async def test_check_log_threshold_flows_through_real_dedup_and_triaging(tmp_path, monkeypatch, session):
    """Requirement: a threshold breach must produce a real AnomalyAlert
    through the EXISTING dedup/TRIAGING path (triage_anomaly), not a new
    parallel one."""
    log = tmp_path / "app.log"
    now = "2026-09-28T12:00:00.000Z"
    lines = [f"{now} INFO ok" for _ in range(9)] + [f"{now} ERROR bad"]
    _write_log(log, lines)
    monkeypatch.setenv("AMOP_LOG_SOURCE_PATH", str(log))

    alert = await mon_mod.check_log_threshold(
        session, "acme/widgets", _ctx(tmp_path), threshold_pct=5.0, window_minutes=10_000_000
    )
    assert alert is not None

    task = await watch_mod.triage_anomaly(session, alert, local_path=None, model=None)
    assert task is not None
    assert task.task_context["source"] == "logs_watcher"
    assert task.task_context["repo"] == "acme/widgets"
    assert task.task_context["github_issue_number"] == mon_mod.THRESHOLD_INCIDENT_ISSUE_NUMBER
    assert task.state == "TRIAGING"  # no local_path given this test -> stops here


async def test_github_sourced_alert_source_label_unaffected(session):
    """Regression check for the one-line watch.py fix (task_context.source
    is now f"{alert.source}_watcher" instead of a hardcoded literal) --
    a GitHub-sourced alert (alert.source defaults to "github") must still
    produce exactly "github_watcher", byte-identical to before."""
    alert = AnomalyAlert(
        severity="high", summary="unaffected by the logs addition", confidence=0.9, issue_index=1
    ).model_copy(update={"repo": "acme/widgets", "github_issue_number": 99})
    task = await watch_mod.triage_anomaly(session, alert, local_path=None, model=None)
    assert task.task_context["source"] == "github_watcher"
