"""Milestone 24 — Reviewer Reliability, Attempt 4: Sandbox-Executed
Counterexample Verification.

Three prior attempts (ROADMAP, Milestones 4/5) each changed the
Reviewer's behavior without fixing its judgment — 1 approval across 15
verdicts. Attempt 2's mechanical override was proven gameable by output
that had the right *shape* and false *content*.

The centerpiece here is `test_MILESTONE_14_REPLAY_...`: the exact
fabricated claim from that finding, which passed the old shape check and
must now be caught. Everything else exists to make that result
trustworthy — that the safety boundary holds, that a TRUE claim is not
also refuted (or the mechanism would just be "always override"), and
that failure modes degrade to the pre-existing behavior rather than
stalling review.

Real Docker sandbox for anything that executes; no Ollama needed.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from amop.agents.handoffs import Evidence, ReviewVerdict, RootCauseReport
from amop.orchestrator.chain import enforce_review_checks
from amop.orchestrator.counterexample import (
    CounterexampleStatus,
    InvalidClaim,
    evaluate_relation,
    parse_claim,
    verify_counterexample,
)
from amop.sandbox.manager import SandboxManager

# The exact diff under review in Milestone 14's finding: the additive
# formula the Reviewer objected to.
_M14_DIFF_SOURCE = (
    "URGENCY_WEIGHT = 3\n"
    "IMPACT_WEIGHT = 2\n"
    "EFFORT_WEIGHT = 0.5\n"
    "\n"
    "def compute_priority_score(urgency, impact, effort):\n"
    "    return urgency*URGENCY_WEIGHT + impact*IMPACT_WEIGHT + effort*EFFORT_WEIGHT\n"
)

# Verbatim from ROADMAP's Milestone 5 findings: claimed (2,1,10)
# outranks (9,8,1). They really score 13.0 and 43.5, so the claim
# contradicts itself -- yet it satisfied the old shape check (has
# digits, sufficient length).
_M14_FABRICATED_CLAIM = {
    "module": "priority",
    "function": "compute_priority_score",
    "calls": [
        {"name": "a", "args": {"urgency": 2, "impact": 1, "effort": 10}, "expect": 13.0},
        {"name": "b", "args": {"urgency": 9, "impact": 8, "effort": 1}, "expect": 43.5},
    ],
    "claim": "a > b",
}


@pytest.fixture
def sandbox_manager():
    manager = SandboxManager()
    yield manager
    manager.destroy_all()


@pytest.fixture
def m14_repo():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "priority.py").write_text(_M14_DIFF_SOURCE)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def _verdict(**kwargs) -> ReviewVerdict:
    base = dict(
        task_id="t", approved=False, addresses_reported_symptom=True,
        rejection_reason="the formula looks wrong",
    )
    base.update(kwargs)
    return ReviewVerdict(**base)


def _report() -> RootCauseReport:
    return RootCauseReport(
        task_id="t",
        root_cause="priority scoring divides by effort",
        confidence=0.9,
        affected_files=["priority.py"],
        evidence=[
            Evidence(type="code", ref="priority.py:17", excerpt="return (u*U + i*I) / effort")
        ],
        suggested_fix_plan="weight effort additively instead of dividing by it",
    )


# ---------------------------------------------------------------------
# The safety boundary. Model-supplied claims are untrusted input
# (Milestone 19's principle) -- everything here must be rejected by the
# PARSER, before the sandbox is ever asked to run it.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,claim",
    [
        (
            "code injection via the claim relation",
            {"module": "priority", "function": "f",
             "calls": [{"name": "a", "args": {}, "expect": 1}],
             "claim": '__import__("os").system("rm -rf /") > 0'},
        ),
        (
            "file read via the claim relation",
            {"module": "priority", "function": "f",
             "calls": [{"name": "a", "args": {}, "expect": 1}],
             "claim": "open('/etc/passwd').read() > a"},
        ),
        (
            "dotted module escapes the repo",
            {"module": "os.path", "function": "join",
             "calls": [{"name": "a", "args": {}, "expect": 1}]},
        ),
        (
            "path traversal in the module name",
            {"module": "../../etc/passwd", "function": "f",
             "calls": [{"name": "a", "args": {}, "expect": 1}]},
        ),
        (
            "dunder function name",
            {"module": "priority", "function": "__import__",
             "calls": [{"name": "a", "args": {}, "expect": 1}]},
        ),
        (
            "non-scalar argument widens the surface",
            {"module": "priority", "function": "f",
             "calls": [{"name": "a", "args": {"x": {"nested": 1}}, "expect": 1}]},
        ),
        (
            "relation references an undeclared name",
            {"module": "priority", "function": "f",
             "calls": [{"name": "a", "args": {}, "expect": 1}], "claim": "a > zzz"},
        ),
        (
            "asserts nothing at all",
            {"module": "priority", "function": "f", "calls": [{"name": "a", "args": {}}]},
        ),
    ],
)
def test_the_parser_rejects_unsafe_or_meaningless_claims(label, claim):
    """Fails CLOSED at the parser, not contained by the sandbox
    afterwards. The sandbox is the second layer; this is the first."""
    with pytest.raises(InvalidClaim):
        parse_claim(claim)


def test_a_relation_is_never_evaluated_with_eval():
    """The whitelist is the guarantee, so evaluation walks the parsed
    tree by hand. A mistake in the whitelist must not be able to become
    code execution."""
    import ast
    import inspect

    from amop.orchestrator import counterexample as mod

    # Checked against the parsed AST, not the source text: an earlier
    # version grepped the source and matched the word "eval()" inside
    # this module's own docstring explaining why it does NOT use eval.
    # Substring matching on source cannot tell code from prose.
    tree = ast.parse(inspect.getsource(mod))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "eval" not in called, "model-supplied text must never reach eval()"
    assert "exec" not in called
    assert "compile" not in called


def test_evaluate_relation_computes_the_real_answer():
    assert evaluate_relation("a > b", {"a": 43.5, "b": 13.0}) is True
    assert evaluate_relation("a > b", {"a": 13.0, "b": 43.5}) is False
    assert evaluate_relation("a == b", {"a": 1.0, "b": 1.0}) is True


# ---------------------------------------------------------------------
# THE CENTERPIECE — Milestone 14's exact documented failure.
# ---------------------------------------------------------------------


async def test_MILESTONE_14_REPLAY_the_fabricated_claim_is_now_refuted(
    m14_repo, sandbox_manager
):
    """The direct measure of whether attempt 4 succeeds where 2 failed.

    This exact claim passed Milestone 5's shape check -- it has digits
    and sufficient length -- so the override did not fire and a
    rejection of a correct, test-passing diff stood. Executing it shows
    the claim contradicts itself: 13.0 is not greater than 43.5.
    """
    sandbox = sandbox_manager.create("m24-replay", m14_repo)
    try:
        result = await verify_counterexample(_M14_FABRICATED_CLAIM, sandbox)
    finally:
        sandbox_manager.destroy("m24-replay")

    assert result.status is CounterexampleStatus.REFUTED, (
        f"the Milestone 14 fabrication was not caught: {result.detail}"
    )
    assert result.observed == {"a": 13.0, "b": 43.5}
    assert "does not hold" in result.detail


async def test_MILESTONE_14_REPLAY_end_to_end_the_rejection_is_overridden(
    m14_repo, sandbox_manager
):
    """The same case through the real gate: refuted claim -> the
    rejection of a passing, in-scope diff is mechanically overridden."""
    sandbox = sandbox_manager.create("m24-replay-e2e", m14_repo)
    try:
        result = await verify_counterexample(_M14_FABRICATED_CLAIM, sandbox)
    finally:
        sandbox_manager.destroy("m24-replay-e2e")

    verdict = _verdict(counterexample_claim=_M14_FABRICATED_CLAIM)
    checked = enforce_review_checks(
        verdict,
        changed_files=["priority.py"],
        report=_report(),
        tests_passed=True,
        counterexample_status=result.status,
    )

    assert checked.approved is True, "a refuted claim must not keep the rejection alive"
    assert "EXECUTED against the real code and contradicted by it" in checked.rejection_reason


# ---------------------------------------------------------------------
# The converse — without this, "always override" would pass the test
# above and the mechanism would be worthless.
# ---------------------------------------------------------------------


async def test_a_TRUE_counterexample_is_verified_and_the_rejection_stands(
    m14_repo, sandbox_manager
):
    true_claim = {
        "module": "priority",
        "function": "compute_priority_score",
        "calls": [
            {"name": "a", "args": {"urgency": 9, "impact": 8, "effort": 1}, "expect": 43.5},
            {"name": "b", "args": {"urgency": 2, "impact": 1, "effort": 10}, "expect": 13.0},
        ],
        "claim": "a > b",
    }
    sandbox = sandbox_manager.create("m24-true", m14_repo)
    try:
        result = await verify_counterexample(true_claim, sandbox)
    finally:
        sandbox_manager.destroy("m24-true")

    assert result.status is CounterexampleStatus.VERIFIED, result.detail

    checked = enforce_review_checks(
        _verdict(counterexample_claim=true_claim),
        changed_files=["priority.py"],
        report=_report(),
        tests_passed=True,
        counterexample_status=result.status,
    )
    assert checked.approved is False, "a verified claim must keep the rejection"


async def test_a_fabricated_VALUE_is_refuted_by_check_one(m14_repo, sandbox_manager):
    """The other independent check: invented numbers, no relation at
    all."""
    claim = {
        "module": "priority",
        "function": "compute_priority_score",
        "calls": [
            {"name": "a", "args": {"urgency": 2, "impact": 1, "effort": 10}, "expect": 999.0}
        ],
    }
    sandbox = sandbox_manager.create("m24-badvalue", m14_repo)
    try:
        result = await verify_counterexample(claim, sandbox)
    finally:
        sandbox_manager.destroy("m24-badvalue")

    assert result.status is CounterexampleStatus.REFUTED
    assert "actually returns 13.0" in result.detail


# ---------------------------------------------------------------------
# Failure modes degrade to the pre-existing behavior, never stall.
# ---------------------------------------------------------------------


async def test_an_unrunnable_claim_is_unavailable_not_refuted(m14_repo, sandbox_manager):
    """UNAVAILABLE, not REFUTED: when a claim never ran we learned
    NOTHING about whether the diff is wrong, and recording that as
    'refuted' would be its own fabrication."""
    claim = {
        "module": "nosuchmodule", "function": "f",
        "calls": [{"name": "a", "args": {}, "expect": 1}],
    }
    sandbox = sandbox_manager.create("m24-missing", m14_repo)
    try:
        result = await verify_counterexample(claim, sandbox)
    finally:
        sandbox_manager.destroy("m24-missing")

    assert result.status is CounterexampleStatus.UNAVAILABLE
    assert "did not run" in result.detail


async def test_a_malformed_claim_never_reaches_the_sandbox():
    """sandbox=None proves the parser rejected it first -- if execution
    were attempted this would raise instead of returning UNAVAILABLE."""
    result = await verify_counterexample({"module": "os.path", "function": "join"}, None)
    assert result.status is CounterexampleStatus.UNAVAILABLE
    assert "malformed" in result.detail


async def test_no_claim_at_all_is_NONE_and_still_overrides():
    result = await verify_counterexample(None, None)
    assert result.status is CounterexampleStatus.NONE

    checked = enforce_review_checks(
        _verdict(),
        changed_files=["priority.py"],
        report=_report(),
        tests_passed=True,
        counterexample_status=result.status,
    )
    assert checked.approved is True
    assert "named no concrete input/output counterexample" in checked.rejection_reason


# ---------------------------------------------------------------------
# The gate's existing guarantees must be untouched by all of this.
# ---------------------------------------------------------------------


async def test_a_verified_claim_cannot_rescue_a_failing_test_suite(m14_repo, sandbox_manager):
    """Ground truth still outranks everything: the override never fires
    when tests fail, regardless of counterexample status."""
    checked = enforce_review_checks(
        _verdict(),
        changed_files=["priority.py"],
        report=_report(),
        tests_passed=False,
        counterexample_status=CounterexampleStatus.REFUTED,
    )
    assert checked.approved is False, "a refuted claim must not override a red suite"


def test_a_refuted_claim_cannot_override_a_scope_violation():
    """The override only ever applies to a rejection the MODEL made --
    never one the orchestrator itself produced from a repo fact."""
    checked = enforce_review_checks(
        _verdict(approved=True),  # model approved; the gate should downgrade
        changed_files=["priority.py", "unrelated.py"],
        report=_report(),
        tests_passed=True,
        counterexample_status=CounterexampleStatus.REFUTED,
    )
    assert checked.approved is False
    assert "scope violation" in checked.rejection_reason


def test_existing_call_sites_keep_working_without_the_new_argument():
    """Backward compatibility: `counterexample_status` defaults to NONE,
    so every pre-Milestone-24 call behaves exactly as before."""
    checked = enforce_review_checks(_verdict(), ["priority.py"], _report(), tests_passed=True)
    assert checked.approved is True


async def test_execution_is_bounded_by_a_timeout(m14_repo, sandbox_manager):
    """A hung claim is UNAVAILABLE, not an indefinite stall of the
    review cycle."""
    (m14_repo / "slowmod.py").write_text(
        "import time\ndef spin(n):\n    time.sleep(n)\n    return 1\n"
    )
    claim = {
        "module": "slowmod", "function": "spin",
        "calls": [{"name": "a", "args": {"n": 30}, "expect": 1}],
    }
    sandbox = sandbox_manager.create("m24-timeout", m14_repo)
    try:
        result = await asyncio.wait_for(
            verify_counterexample(claim, sandbox, timeout=2.0), timeout=60
        )
    finally:
        sandbox_manager.destroy("m24-timeout")

    assert result.status is CounterexampleStatus.UNAVAILABLE


# ---------------------------------------------------------------------
# The behavioral change this milestone makes beyond catching M14, pinned
# here so it is a deliberate contract rather than a side effect.
# ---------------------------------------------------------------------


def test_PROSE_ONLY_counterexamples_no_longer_preserve_a_rejection():
    """The real contract change, and it is not a small one.

    Before: prose with digits and enough length kept a rejection alive.
    Milestone 14 proved that gameable -- a well-shaped, arithmetically
    false claim protected a rejection of a correct diff.

    After: only an EXECUTED, verified claim does. Prose is still carried
    for humans, but decides nothing.

    The cost, stated plainly because it is real: a Reviewer that is
    RIGHT but cannot express its objection executably (a diff touching
    I/O, side effects, or anything not callable with JSON-ish arguments)
    now gets overridden too, provided tests pass and the diff is in
    scope. That is a deliberate trade -- the prose gate was demonstrably
    broken, and the observed failure mode is over-rejection (1 approval
    in 15) -- but it is a trade, not a free win.
    """
    verdict = _verdict(
        counterexample="f(1) returns 2 but should return 3 -- clearly wrong",
        counterexample_claim=None,
    )
    checked = enforce_review_checks(
        verdict,
        changed_files=["priority.py"],
        report=_report(),
        tests_passed=True,
        counterexample_status=CounterexampleStatus.NONE,
    )
    assert checked.approved is True


async def test_a_method_on_a_class_can_be_verified(sandbox_manager):
    """Method support exists because the buggy_calculator fixture -- one
    of this project's two main fixtures -- has NO module-level functions
    at all. A function-only format could not express a single
    counterexample about it."""
    repo = Path("tests/fixtures/buggy_calculator").resolve()
    sandbox = sandbox_manager.create("m24-method-test", repo)
    try:
        result = await verify_counterexample(
            {
                "module": "calculator",
                "function": "Calculator.average",
                "calls": [
                    {"name": "a", "args": {"values": [2, 4]}, "expect": 3.0}
                ],
            },
            sandbox,
        )
    finally:
        sandbox_manager.destroy("m24-method-test")

    # The fixture's average() divides by len-1, so [2,4] gives 6.0.
    assert result.status is CounterexampleStatus.REFUTED
    assert "actually returns 6.0" in result.detail


@pytest.mark.parametrize(
    "bad_function", ["a.b.c", "os.system.__call__", "__import__", "a..b"]
)
def test_method_support_did_not_widen_the_safety_boundary(bad_function):
    with pytest.raises(InvalidClaim):
        parse_claim(
            {"module": "m", "function": bad_function,
             "calls": [{"name": "a", "args": {}, "expect": 1}]}
        )
