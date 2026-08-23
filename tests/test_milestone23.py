"""Milestone 23 — Real `agent_actions` Table, Joined to the Chain.

Closes the gap Milestone 22 named as its top-priority unresolved item:
the Safety Engine's ALLOW/DENY decisions had no tamper evidence.

The most important test in this file is
`test_deleting_every_agent_action_row_is_detected` — that is the exact
attack Milestone 22's "one shared chain, never two" decision exists to
catch, and it is mutation-tested against a simulated separate-chain
design to prove the decision was load-bearing rather than theoretical.

Requires: Postgres at TEST_DATABASE_URL.
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select, text

from amop.audit.actions import MAX_RESULT_CHARS, record_action, summarize_result
from amop.audit.chain import (
    AGENT_ACTIONS_TABLE,
    CHAINED_TABLES,
    GENESIS_HASH,
    TRANSITIONS_TABLE,
    agent_action_content,
    hash_for_row,
    verify_chain,
)
from amop.database.models import AgentAction, TaskTransition
from amop.database.session import init_db, make_engine, make_session_factory
from amop.orchestrator.state_machine import TaskState
from amop.orchestrator.task import create_task, transition
from amop.safety.secret_scan import redact_arguments, redact_secrets
from amop.tools.registry import ToolContext, ToolResult, invoke_tool

import amop.codebase_intel.search  # noqa: F401,E402 -- registers search_code
import amop.sandbox.tools  # noqa: F401,E402 -- registers write_file/read_file

TEST_DATABASE_URL = "postgresql+asyncpg://localhost/amop_test"


@pytest_asyncio.fixture
async def engine():
    eng = make_engine(TEST_DATABASE_URL)
    await init_db(eng)
    async with eng.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE agent_actions, task_transitions, tasks "
                "RESTART IDENTITY CASCADE"
            )
        )
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return make_session_factory(engine)


@pytest_asyncio.fixture
async def session(session_factory):
    async with session_factory() as s:
        yield s


async def _interleaved(session, tmp_path, rounds=3):
    """Build a chain that genuinely ALTERNATES between the two tables.

    Interleaving is what makes the shared chain meaningful: if every
    agent_actions row happened to sit at the end, deleting them would be
    a pure truncation, which no in-database check can catch (see the
    documented limit at the bottom of this file). Alternating is also
    what really happens -- an agent acts between state changes.
    """
    task = await create_task(session, task_type="bug_fix", task_context={})
    ctx = ToolContext(
        agent_name="coder",
        scratch_dir=Path(tmp_path),
        mode="observer",  # every mutating call is DENIED -> a real audited decision
        db_session=session,
        task_id=task.id,
    )
    states = [TaskState.TRIAGING, TaskState.INVESTIGATING, TaskState.PLANNING_FIX]
    for i in range(rounds):
        await transition(session, task, states[i])
        await invoke_tool(
            "write_file",
            {"path": f"f{i}.py", "content": "x = 1\n"},
            ctx,
            agent_name="coder",
        )
    return task


# ---------------------------------------------------------------------
# The table itself + Safety Engine integration.
# ---------------------------------------------------------------------


async def test_a_denied_tool_call_writes_a_real_agent_actions_row(session, tmp_path):
    task = await create_task(session, task_type="bug_fix", task_context={})
    ctx = ToolContext(
        agent_name="coder",
        scratch_dir=Path(tmp_path),
        mode="observer",
        db_session=session,
        task_id=task.id,
    )
    result = await invoke_tool(
        "write_file", {"path": "x.py", "content": "y = 1\n"}, ctx, agent_name="coder"
    )
    assert result.error_code == "DENIED"

    rows = (await session.execute(select(AgentAction))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.decision == "DENY"
    assert row.decision_reason == "mode_forbids_mutation"
    assert row.tool_name == "write_file"
    assert row.agent_name == "coder"
    assert row.task_id == task.id
    assert row.row_hash is not None and row.chain_pos is not None


async def test_an_allowed_call_is_audited_as_ALLOW_even_when_it_then_fails(
    session, tmp_path
):
    """The point is a COMPLETE record, not an incident log -- permitted
    calls must be recorded too, or the trail only shows what was refused.

    This also pins a distinction that is easy to collapse and important
    to keep: the Safety Engine ALLOWED this call, and the tool then
    failed for an unrelated operational reason. That is decision=ALLOW
    with a failure result -- NOT a DENY. Conflating the two would make
    the audit trail claim the Safety Engine refused things it permitted.
    """
    task = await create_task(session, task_type="bug_fix", task_context={})
    ctx = ToolContext(
        agent_name="investigator",
        scratch_dir=Path(tmp_path),
        mode="operator",
        db_session=session,
        repo_path=str(tmp_path),
        task_id=task.id,
    )
    result = await invoke_tool(
        "search_code", {"query": "anything"}, ctx, agent_name="investigator"
    )
    assert not result.success, "sandbox-less search_code is expected to fail"

    rows = (await session.execute(select(AgentAction))).scalars().all()
    assert len(rows) == 1
    assert rows[0].decision == "ALLOW", "permitted-then-failed is not a DENY"
    assert rows[0].decision_reason is None
    assert rows[0].latency_ms is not None and rows[0].latency_ms >= 0
    assert "SANDBOX_UNAVAILABLE" in rows[0].result


async def test_a_tool_outside_the_agents_allowlist_is_audited(session, tmp_path):
    """Registry-level refusal, not a Safety Engine one -- still "what an
    agent actually attempted" (Section 12.6), so still audited."""
    task = await create_task(session, task_type="bug_fix", task_context={})
    ctx = ToolContext(
        agent_name="watcher",
        scratch_dir=Path(tmp_path),
        mode="operator",
        db_session=session,
        task_id=task.id,
    )
    result = await invoke_tool(
        "write_file",
        {"path": "x.py", "content": "y = 1\n"},
        ctx,
        agent_name="watcher",
        allowed_tools=("read_file",),
    )
    assert result.error_code == "TOOL_NOT_PERMITTED"

    rows = (await session.execute(select(AgentAction))).scalars().all()
    assert len(rows) == 1
    assert rows[0].decision == "DENY"
    assert rows[0].decision_reason == "tool_not_permitted"


async def test_a_malformed_call_is_not_audited(session, tmp_path):
    """The deliberate boundary: UNKNOWN_TOOL/INVALID_ARGS never reached a
    permission decision, so there is no decision to record."""
    ctx = ToolContext(
        agent_name="coder", scratch_dir=Path(tmp_path), mode="operator", db_session=session
    )
    await invoke_tool("no_such_tool", {}, ctx, agent_name="coder")
    await invoke_tool("write_file", {"path": "x.py"}, ctx, agent_name="coder")  # missing content

    assert (await session.execute(select(AgentAction))).scalars().all() == []


async def test_an_audit_write_failure_never_breaks_the_tool_call(tmp_path, caplog):
    """Deliberate tradeoff, pinned so it can't erode silently: audit
    failure is logged loudly and swallowed, because failing the tool call
    would make audit-log availability a DoS surface for the whole system.

    A ctx with no db_session is the benign version of "no audit row" --
    the tool call must still work normally."""
    ctx = ToolContext(
        agent_name="investigator",
        scratch_dir=Path(tmp_path),
        mode="operator",
        repo_path=str(tmp_path),
    )  # no db_session -> no audit row is possible
    result = await invoke_tool(
        "search_code", {"query": "x"}, ctx, agent_name="investigator"
    )
    # The call still ran and returned its own normal result. Asserting
    # the ABSENCE of an audit-caused failure rather than one specific
    # error code: which code search_code returns without a session is
    # its own business and could legitimately change, but it must never
    # be an audit-layer failure.
    assert result.error_code not in (None, "AUDIT_ERROR")
    assert "audit" not in (result.message or "").lower()


# ---------------------------------------------------------------------
# Redaction (Section 12.5).
# ---------------------------------------------------------------------


def test_redaction_reuses_the_pre_pr_gate_pattern_set():
    assert "[REDACTED]" in redact_secrets("ghp_" + "a" * 36)
    assert "[REDACTED]" in redact_secrets("AKIA" + "A" * 16)
    assert redact_secrets("nothing secret here") == "nothing secret here"


def test_redaction_recurses_and_preserves_non_string_values():
    out = redact_arguments(
        {"path": "x.py", "content": "ghp_" + "b" * 36, "n": 5, "deep": {"k": ["AKIA" + "B" * 16]}}
    )
    assert out["content"] == "[REDACTED]"
    assert out["deep"]["k"] == ["[REDACTED]"]
    assert out["n"] == 5, "non-strings must pass through untouched"


async def test_secrets_in_tool_arguments_do_not_reach_the_audit_row(session, tmp_path):
    task = await create_task(session, task_type="bug_fix", task_context={})
    ctx = ToolContext(
        agent_name="coder",
        scratch_dir=Path(tmp_path),
        mode="observer",
        db_session=session,
        task_id=task.id,
    )
    secret = "ghp_" + "c" * 36
    await invoke_tool(
        "write_file", {"path": "x.py", "content": f"TOKEN = '{secret}'"}, ctx, agent_name="coder"
    )
    row = (await session.execute(select(AgentAction))).scalars().one()
    assert secret not in str(row.arguments)
    assert "[REDACTED]" in str(row.arguments)


def test_a_huge_result_is_truncated_with_the_truncation_recorded():
    big = ToolResult(success=True, output="x" * (MAX_RESULT_CHARS * 3))
    out = summarize_result(big)
    assert len(out) < MAX_RESULT_CHARS * 2
    assert "truncated" in out, "a reader must be able to tell it was cut"


# ---------------------------------------------------------------------
# The shared chain.
# ---------------------------------------------------------------------


async def test_both_tables_are_registered_in_one_chain():
    names = {t.name for t in CHAINED_TABLES}
    assert names == {"task_transitions", "agent_actions"}


async def test_a_clean_interleaved_chain_across_both_tables_verifies(session, tmp_path):
    await _interleaved(session, tmp_path, rounds=3)
    result = await verify_chain(session)
    assert result.intact, result.reason
    assert result.checked == 6, "3 transitions + 3 audited denials"


async def test_the_chain_genuinely_interleaves_the_two_tables(session, tmp_path):
    """If chain_pos didn't interleave, single-table deletion would be a
    truncation and undetectable -- so this property is what the headline
    test below actually depends on."""
    await _interleaved(session, tmp_path, rounds=3)
    trs = (await session.execute(select(TaskTransition))).scalars().all()
    acts = (await session.execute(select(AgentAction))).scalars().all()
    merged = sorted(
        [(t.chain_pos, "T") for t in trs] + [(a.chain_pos, "A") for a in acts]
    )
    kinds = "".join(k for _, k in merged)
    assert kinds == "TATATA", f"expected alternating tables, got {kinds}"


async def test_a_mutated_agent_actions_row_is_detected_and_named(session, tmp_path):
    """Proves BOTH tables are genuinely walked -- a verifier that only
    checked task_transitions would pass this."""
    await _interleaved(session, tmp_path, rounds=3)
    target = (
        await session.execute(select(AgentAction).order_by(AgentAction.chain_pos))
    ).scalars().all()[1]
    target_id = target.id

    await session.execute(
        text("UPDATE agent_actions SET decision = 'ALLOW' WHERE id = :i"),
        {"i": target_id},
    )
    await session.commit()
    session.expire_all()

    result = await verify_chain(session)
    assert not result.intact
    assert result.first_divergence_table == "agent_actions"
    assert result.first_divergence_id == target_id
    assert "altered after it was written" in result.reason


async def test_deleting_every_agent_action_row_is_detected(session, tmp_path):
    """THE test this milestone exists to make pass.

    Wholesale deletion of one table's rows, the other left untouched.
    With SEPARATE chains this is undetectable: task_transitions would
    still verify perfectly against itself, and agent_actions' own chain
    would simply be empty -- and an empty chain verifies as INTACT, which
    is worse than no verification because it actively reports a clean
    bill of health over destroyed evidence.

    With ONE shared chain the surviving transitions' prev_hash values
    point at hashes of rows that no longer exist, so the break is
    detected and localized.
    """
    await _interleaved(session, tmp_path, rounds=3)
    assert (await verify_chain(session)).intact

    await session.execute(text("DELETE FROM agent_actions"))
    await session.commit()
    session.expire_all()

    assert (await session.execute(select(AgentAction))).scalars().all() == []
    surviving = (await session.execute(select(TaskTransition))).scalars().all()
    assert len(surviving) == 3, "the other table must be untouched"

    result = await verify_chain(session)
    assert not result.intact, (
        "wholesale deletion of agent_actions went undetected -- this is "
        "exactly the hole the single-shared-chain decision exists to close"
    )
    assert result.first_divergence_table == "task_transitions"
    assert "deleted" in result.reason or "broken" in result.reason


async def test_separate_chains_would_have_MISSED_the_deletion(session, tmp_path):
    """The counterfactual that justifies the locked decision, pinned as a
    permanent executable fact rather than a claim in a doc.

    Simulates the REJECTED design directly: chain each table only against
    itself. Then delete every `agent_actions` row and verify
    `task_transitions` against its own independent chain -- it comes back
    INTACT, because nothing in it ever referenced the deleted rows.

    That is the hole. A separate-chain design would hand an operator a
    clean bill of health over a destroyed tool-call audit trail, which is
    strictly worse than no verification at all, because it converts
    "I don't know" into a false "everything is fine".
    """
    await _interleaved(session, tmp_path, rounds=3)

    # Re-chain each table independently -- exactly what separate chains
    # would have produced.
    for table in CHAINED_TABLES:
        rows = (
            await session.execute(
                select(table.model).order_by(table.model.chain_pos)
            )
        ).scalars().all()
        prev = GENESIS_HASH
        for row in rows:
            row.prev_hash = prev
            row.row_hash = hash_for_row(row, prev)
            session.add(row)
            prev = row.row_hash
    await session.commit()

    await session.execute(text("DELETE FROM agent_actions"))
    await session.commit()
    session.expire_all()

    # Verify task_transitions against ITS OWN chain, the way a
    # separate-chain verifier would.
    rows = (
        await session.execute(
            select(TaskTransition).order_by(TaskTransition.chain_pos)
        )
    ).scalars().all()
    prev = GENESIS_HASH
    independently_intact = True
    for row in rows:
        if row.prev_hash != prev or hash_for_row(row, prev) != row.row_hash:
            independently_intact = False
            break
        prev = row.row_hash

    assert independently_intact, (
        "the separate-chain simulation should verify clean -- if it does "
        "not, this test is no longer demonstrating the hole it exists for"
    )
    # ...and that is precisely the problem: 3 agent_actions rows were
    # destroyed and this design reports everything as fine.


async def test_deleting_every_transition_row_is_also_detected(session, tmp_path):
    """The symmetric case -- the shared chain protects both directions,
    not just the newly-added table."""
    await _interleaved(session, tmp_path, rounds=3)
    await session.execute(text("DELETE FROM task_transitions"))
    await session.commit()
    session.expire_all()

    result = await verify_chain(session)
    assert not result.intact
    assert result.first_divergence_table == "agent_actions"


# ---------------------------------------------------------------------
# Concurrency across both tables (Milestone 16/20 style).
# ---------------------------------------------------------------------


async def test_concurrent_writes_to_both_tables_stay_linear(session_factory, tmp_path):
    """Real connections, both row types racing. The shared chain_pos must
    still come out strictly ordered with no duplicate prev_hash (a
    duplicate IS a fork)."""
    task_ids = []
    async with session_factory() as s:
        for _ in range(6):
            t = await create_task(s, task_type="bug_fix", task_context={})
            task_ids.append(t.id)

    async def do_transition(task_id):
        async with session_factory() as s:
            from amop.database.models import Task

            t = await s.get(Task, task_id)
            await transition(s, t, TaskState.TRIAGING)

    async def do_tool_call(task_id):
        async with session_factory() as s:
            ctx = ToolContext(
                agent_name="coder",
                scratch_dir=Path(tmp_path),
                mode="observer",
                db_session=s,
                task_id=task_id,
            )
            await invoke_tool(
                "write_file", {"path": "z.py", "content": "q = 1\n"}, ctx, agent_name="coder"
            )

    await asyncio.gather(
        *(do_transition(t) for t in task_ids),
        *(do_tool_call(t) for t in task_ids),
    )

    async with session_factory() as s:
        trs = (await s.execute(select(TaskTransition))).scalars().all()
        acts = (await s.execute(select(AgentAction))).scalars().all()
        assert len(trs) == 6 and len(acts) == 6

        positions = [r.chain_pos for r in list(trs) + list(acts)]
        assert len(set(positions)) == len(positions), "duplicate chain_pos -- fork"
        prevs = [r.prev_hash for r in list(trs) + list(acts)]
        assert len(set(prevs)) == len(prevs), "duplicate prev_hash -- chain forked"

        result = await verify_chain(s)
        assert result.intact, result.reason
        assert result.checked == 12


# ---------------------------------------------------------------------
# Canonicalization for the new table.
# ---------------------------------------------------------------------


def test_agent_action_content_excludes_latency():
    """Latency is a measurement about the call, not a claim about what
    was attempted -- and hashing it would make a legitimate
    re-measurement look like tampering."""
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    common = dict(
        task_id=None,
        agent_name="coder",
        tool_name="write_file",
        arguments={"path": "x"},
        result="OK",
        decision="ALLOW",
        decision_reason=None,
        timestamp=ts,
    )
    assert agent_action_content(**common) == agent_action_content(**common)
    assert "latency" not in agent_action_content(**common)


def test_agent_action_argument_order_does_not_change_the_hash():
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    a = agent_action_content(
        task_id=None, agent_name="c", tool_name="t",
        arguments={"a": 1, "b": 2}, result="OK", decision="ALLOW",
        decision_reason=None, timestamp=ts,
    )
    b = agent_action_content(
        task_id=None, agent_name="c", tool_name="t",
        arguments={"b": 2, "a": 1}, result="OK", decision="ALLOW",
        decision_reason=None, timestamp=ts,
    )
    assert a == b, "nested dict ordering must not affect the hash"


async def test_agent_action_hashes_survive_a_postgres_round_trip(
    session, session_factory, tmp_path
):
    """Same pin Milestone 22 put on task_transitions, for the new table:
    hash-on-write must equal hash-on-verify after a real DB read.

    The JSONB `arguments` column makes this worth re-pinning rather than
    assuming it carries over -- a dict round-tripping through Postgres
    JSONB is a genuinely different question from a TEXT column, and a
    key-order or type change on the way back would break every hash
    while each half looked correct alone.
    """
    await _interleaved(session, tmp_path, rounds=2)

    # A FRESH session, so values come from Postgres rather than the
    # identity map that wrote them.
    async with session_factory() as fresh:
        rows = (
            await fresh.execute(select(AgentAction).order_by(AgentAction.chain_pos))
        ).scalars().all()
        assert rows
        for row in rows:
            assert hash_for_row(row, row.prev_hash) == row.row_hash
            assert isinstance(row.arguments, dict)
