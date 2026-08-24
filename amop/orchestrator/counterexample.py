"""Sandbox-executed counterexample verification — Milestone 24,
Reviewer Reliability attempt 4.

Three prior attempts (ROADMAP, Milestones 4/5) each changed the
Reviewer's behavior without fixing its judgment: 1 approval across 15
verdicts. Attempt 2 added a mechanical override that fired when a
rejection named no concrete counterexample -- and the model learned to
satisfy its *shape* check (some length, at least one digit) with a claim
that was arithmetically false. The documented case: for the diff
`urgency*3 + impact*2 + effort*0.5`, it claimed `(2,1,10)` outranks
`(9,8,1)`. Those actually score 13.0 and 43.5, so the claim contradicts
itself -- and it passed, because the check read form, not truth.

This module executes the claim instead of inspecting it.

THE SAFETY MODEL, first, because it is the whole design:

    The model supplies DATA. It never supplies code.

The runner script below is a fixed constant written here. The model's
claim is serialized to JSON, written to a file, and `json.load`ed inside
the sandbox -- never interpolated into Python source, so there is no
injection surface even in principle. The `claim` relation is parsed with
`ast` into a whitelist of comparisons and evaluated ON THE HOST against
numbers the sandbox returned; model-supplied text never reaches an
interpreter anywhere. Execution happens in the existing Milestone 3
sandbox (no network, dropped capabilities, pid/memory limits) under a
short timeout.

TWO INDEPENDENT CHECKS, which is what actually kills the documented
fabrication:

    1. Do the claimed `expect` values match what the code really
       returns?  (catches invented numbers)
    2. Does the claimed relation hold GIVEN the real values?  (catches
       false reasoning -- this is the check the Milestone 14 case fails,
       because 13.0 > 43.5 is simply False)

WHAT THIS DOES NOT DO -- stated plainly, because overclaiming here would
repeat the mistake of the last three attempts:

  * It does not fix the Reviewer's judgment. It makes an unfounded
    rejection mechanically unable to survive, which is a different and
    smaller thing. Any improvement in approval rate is the orchestrator
    overriding bad rejections more reliably, NOT the model reviewing
    better.
  * Verifying arithmetic is not verifying that a defect exists. A claim
    can be perfectly true and still demonstrate nothing (`f(1,1,1) ==
    5.5` is correct and meaningless); such a rejection still stands.
    This raises the bar from *plausible* to *true*, not to *relevant*.
    Relevance is the judgment that cannot be executed.
  * It only applies where the changed code is callable with scalar
    arguments. Diffs touching I/O, side effects, or complex objects
    return UNAVAILABLE and fall back to the pre-existing behavior.
"""

import ast
import json
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Identifiers only -- no dots, no slashes, no dunder. Applied to the
# module and function names before anything touches the filesystem, so a
# traversal or an import of something outside the repo is rejected as
# malformed rather than defended against later.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Scalars only. A list/dict argument is not rejected because it would be
# dangerous (it is JSON either way) but because it widens the surface
# for no demonstrated need -- every counterexample this problem has ever
# produced is scalar. Narrow now; widen if a real case needs it.
_ALLOWED_ARG_TYPES = (int, float, str, bool, type(None))

_MAX_LIST_LEN = 32


def _is_allowed_arg(value) -> bool:
    """Scalars, plus a FLAT list of scalars.

    Lists were not in the original design and were added when the first
    real case needed one: `average([])` -- a counterexample about a
    function taking a collection cannot be expressed without it, and
    that is an ordinary shape, not an exotic one. Still pure JSON data
    loaded via `json.load` in the sandbox, never code, so the safety
    story is unchanged. Deliberately FLAT and bounded: nesting buys no
    demonstrated expressiveness and only widens what has to be reasoned
    about.
    """
    if isinstance(value, _ALLOWED_ARG_TYPES):
        return True
    if isinstance(value, list) and len(value) <= _MAX_LIST_LEN:
        return all(isinstance(item, _ALLOWED_ARG_TYPES) for item in value)
    return False


_MAX_CALLS = 4
_MAX_ARGS = 8
# Floating-point comparison tolerance for "does the claimed value match
# reality". Absolute-and-relative, because scores here span single
# digits to hundreds and a pure absolute epsilon would be wrong at one
# end or the other.
_REL_TOL = 1e-6
_ABS_TOL = 1e-9

DEFAULT_TIMEOUT_SECONDS = 10.0


class CounterexampleStatus(str, Enum):
    """The executed outcome, passed to the (still pure) review gate.

    NONE and UNAVAILABLE are deliberately distinct even though both
    currently lead to the same override: "the Reviewer offered nothing"
    and "the Reviewer offered something that could not be run" are
    different facts about the review, and collapsing them would make the
    audit trail unable to tell them apart.
    """

    NONE = "none"  # no claim supplied
    VERIFIED = "verified"  # ran, and the claim holds -- rejection stands
    REFUTED = "refuted"  # ran, and the claim is false -- override
    UNAVAILABLE = "unavailable"  # malformed, un-runnable, or timed out


@dataclass
class CounterexampleResult:
    status: CounterexampleStatus
    detail: str
    observed: dict[str, Any] | None = None


# The runner. FIXED -- written here, never model-generated. It reads a
# JSON payload and emits JSON. Its only inputs from the model are
# already-validated identifiers and JSON scalars.
_RUNNER = '''
import json, sys, importlib, traceback

def main():
    with open(sys.argv[1]) as fh:
        payload = json.load(fh)
    sys.path.insert(0, payload["repo_root"])
    out = {}
    try:
        module = importlib.import_module(payload["module"])
        segments = payload["function"].split(".")
        target = getattr(module, segments[0])
        if len(segments) == 2:
            # Class.method -- instantiate with no args, then bind.
            if isinstance(target, type):
                target = target()
            target = getattr(target, segments[1])
        func = target
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return
    raised = {}
    for call in payload["calls"]:
        try:
            value = func(**call["args"])
        except Exception as exc:
            # A raise is DATA here, not a failure: the claim may be
            # asserting precisely that this call raises.
            raised[call["name"]] = type(exc).__name__
            out[call["name"]] = None
            continue
        try:
            json.dumps(value)
        except TypeError:
            value = repr(value)
        out[call["name"]] = value
    print(json.dumps({"observed": out, "raised": raised}))

try:
    main()
except Exception:
    print(json.dumps({"error": traceback.format_exc(limit=1)}))
'''


class InvalidClaim(ValueError):
    """The claim is not a well-formed, safe-to-execute counterexample."""


def parse_claim(raw: Any) -> dict:
    """Validate a model-supplied claim into a safe, executable shape.

    Everything here fails CLOSED: anything not explicitly recognized
    raises InvalidClaim, which the caller turns into UNAVAILABLE. There
    is no branch that passes something through unvalidated.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InvalidClaim(f"not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise InvalidClaim("claim must be a JSON object")

    module = raw.get("module")
    function = raw.get("function")
    if not isinstance(module, str) or not _IDENT_RE.match(module):
        raise InvalidClaim(f"module must be a bare Python identifier, got {module!r}")
    if module.startswith("__"):
        raise InvalidClaim("module may not be a dunder name")

    # `function` may be `name` or `Class.name`. Method support was added
    # when the first real case needed it: the buggy_calculator fixture
    # -- one of this project's two main fixtures -- has NO module-level
    # functions at all, only Calculator methods, so a function-only
    # format could not express a single counterexample about it.
    #
    # Still identifiers only, at most two segments, no dunders. Resolving
    # a method means the sandbox instantiates the class with no
    # arguments, which is repo code running in the sandbox -- exactly the
    # trust boundary already crossed by calling the function at all, so
    # no new exposure.
    if not isinstance(function, str):
        raise InvalidClaim(f"function must be a string, got {function!r}")
    segments = function.split(".")
    if not (1 <= len(segments) <= 2):
        raise InvalidClaim(
            f"function must be `name` or `Class.name`, got {function!r}"
        )
    for segment in segments:
        if not _IDENT_RE.match(segment):
            raise InvalidClaim(f"function segment {segment!r} is not an identifier")
        if segment.startswith("__"):
            raise InvalidClaim("function may not contain a dunder name")

    calls_raw = raw.get("calls")
    if not isinstance(calls_raw, list) or not (1 <= len(calls_raw) <= _MAX_CALLS):
        raise InvalidClaim(f"calls must be a list of 1..{_MAX_CALLS} entries")

    calls, names = [], set()
    for entry in calls_raw:
        if not isinstance(entry, dict):
            raise InvalidClaim("each call must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not _IDENT_RE.match(name):
            raise InvalidClaim(f"call name must be an identifier, got {name!r}")
        if name in names:
            raise InvalidClaim(f"duplicate call name {name!r}")
        names.add(name)

        args = entry.get("args")
        if not isinstance(args, dict) or len(args) > _MAX_ARGS:
            raise InvalidClaim(f"call {name!r}: args must be an object of <= {_MAX_ARGS}")
        for key, value in args.items():
            if not isinstance(key, str) or not _IDENT_RE.match(key):
                raise InvalidClaim(f"call {name!r}: bad argument name {key!r}")
            if not _is_allowed_arg(value):
                raise InvalidClaim(
                    f"call {name!r}: argument {key!r} must be a JSON scalar or "
                    f"a flat list of scalars, got {type(value).__name__}"
                )
        expect = entry.get("expect")
        if expect is not None and not isinstance(expect, _ALLOWED_ARG_TYPES):
            raise InvalidClaim(f"call {name!r}: expect must be a JSON scalar")
        # "this call raises X" is an extremely common counterexample
        # shape ("average([]) raises ZeroDivisionError") that a
        # value-only format simply cannot express. An exception TYPE
        # NAME only -- an identifier, compared as a string, never
        # resolved or constructed.
        expect_raises = entry.get("expect_raises")
        if expect_raises is not None:
            if not isinstance(expect_raises, str) or not _IDENT_RE.match(expect_raises):
                raise InvalidClaim(
                    f"call {name!r}: expect_raises must be an exception type "
                    f"name, got {expect_raises!r}"
                )
            if expect is not None:
                raise InvalidClaim(
                    f"call {name!r}: a call cannot both return {expect!r} and raise"
                )
        calls.append(
            {"name": name, "args": args, "expect": expect, "expect_raises": expect_raises}
        )

    claim = raw.get("claim")
    if claim is not None:
        if not isinstance(claim, str):
            raise InvalidClaim("claim must be a string")
        _validate_relation(claim, names)

    if claim is None and not any(
        c["expect"] is not None or c["expect_raises"] is not None for c in calls
    ):
        raise InvalidClaim(
            "a counterexample must assert something: give `expect` or "
            "`expect_raises` on at least one call, or a `claim` relation"
        )
    return {"module": module, "function": function, "calls": calls, "claim": claim}


_ALLOWED_COMPARE_OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)


def _validate_relation(claim: str, names: set[str]) -> ast.Expression:
    """Parse `claim` into a whitelist of comparisons. NEVER eval'd.

    Only a single comparison expression over known call names and
    literal constants is allowed -- no calls, no attributes, no
    subscripts, no names other than the declared results. So a claim
    like `__import__("os").system("rm -rf /")` is rejected here as
    malformed, long before anything could execute it, rather than being
    contained by the sandbox afterwards. The sandbox is the second
    layer, not the first.
    """
    try:
        tree = ast.parse(claim, mode="eval")
    except SyntaxError as exc:
        raise InvalidClaim(f"claim is not a parsable expression: {exc}") from exc

    if not isinstance(tree.body, ast.Compare):
        raise InvalidClaim("claim must be a comparison, e.g. 'a > b'")
    for op in tree.body.ops:
        if not isinstance(op, _ALLOWED_COMPARE_OPS):
            raise InvalidClaim(f"comparison operator {type(op).__name__} not allowed")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Compare, ast.Constant)):
            continue
        if isinstance(node, _ALLOWED_COMPARE_OPS):
            continue
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise InvalidClaim(
                    f"claim references {node.id!r}, which is not one of the "
                    f"declared calls {sorted(names)}"
                )
            continue
        if isinstance(node, ast.Load):
            continue
        raise InvalidClaim(f"claim may not contain {type(node).__name__}")
    return tree


def evaluate_relation(claim: str, observed: dict[str, Any]) -> bool:
    """Evaluate a validated relation on the HOST against real values.

    Walks the parsed tree by hand rather than calling eval() even on the
    validated subset: the validation above is the guarantee, and eval()
    would make that guarantee the only thing standing between a future
    edit to the whitelist and arbitrary execution. Manual evaluation
    means a mistake in the whitelist cannot become code execution.
    """
    tree = _validate_relation(claim, set(observed))
    compare = tree.body

    def value_of(node):
        if isinstance(node, ast.Constant):
            return node.value
        return observed[node.id]

    left = value_of(compare.left)
    for op, comparator in zip(compare.ops, compare.comparators):
        right = value_of(comparator)
        if isinstance(op, ast.Lt):
            ok = left < right
        elif isinstance(op, ast.LtE):
            ok = left <= right
        elif isinstance(op, ast.Gt):
            ok = left > right
        elif isinstance(op, ast.GtE):
            ok = left >= right
        elif isinstance(op, ast.Eq):
            ok = left == right
        else:
            ok = left != right
        if not ok:
            return False
        left = right
    return True


def _matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) <= max(_ABS_TOL, _REL_TOL * max(abs(expected), abs(actual)))
    return expected == actual


async def verify_counterexample(
    raw_claim: Any,
    sandbox,
    repo_root: str = "/workspace",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> CounterexampleResult:
    """Run a model-supplied counterexample against the real code."""
    import asyncio

    if raw_claim in (None, "", {}):
        return CounterexampleResult(CounterexampleStatus.NONE, "no counterexample given")

    try:
        claim = parse_claim(raw_claim)
    except InvalidClaim as exc:
        return CounterexampleResult(
            CounterexampleStatus.UNAVAILABLE, f"malformed counterexample: {exc}"
        )

    if sandbox is None:
        return CounterexampleResult(
            CounterexampleStatus.UNAVAILABLE, "no sandbox available to execute the claim"
        )

    token = uuid.uuid4().hex
    payload_path = f"/tmp/amop-ce-{token}.json"
    runner_path = f"/tmp/amop-ce-{token}.py"
    payload = {
        "repo_root": repo_root,
        "module": claim["module"],
        "function": claim["function"],
        "calls": [{"name": c["name"], "args": c["args"]} for c in claim["calls"]],
    }

    try:
        await asyncio.to_thread(sandbox.write_file, payload_path, json.dumps(payload))
        await asyncio.to_thread(sandbox.write_file, runner_path, _RUNNER)
        exec_result = await asyncio.to_thread(
            sandbox.exec_run, f"python {runner_path} {payload_path}", timeout
        )
    except Exception as exc:  # noqa: BLE001 -- a failure to run is UNAVAILABLE, never fatal
        return CounterexampleResult(
            CounterexampleStatus.UNAVAILABLE, f"could not execute: {type(exc).__name__}: {exc}"
        )
    finally:
        try:
            await asyncio.to_thread(sandbox.exec_run, f"rm -f {payload_path} {runner_path}")
        except Exception:  # noqa: BLE001 -- cleanup is best-effort in an ephemeral container
            pass

    stdout = (exec_result.stdout or "").strip()
    if exec_result.exit_code != 0 or not stdout:
        return CounterexampleResult(
            CounterexampleStatus.UNAVAILABLE,
            f"runner failed (exit {exec_result.exit_code}): "
            f"{(exec_result.stderr or stdout or '')[:300]}",
        )
    try:
        parsed = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return CounterexampleResult(
            CounterexampleStatus.UNAVAILABLE, f"runner output not JSON: {stdout[:300]}"
        )
    if "error" in parsed:
        # The claim named something that does not exist, or calling it
        # raised. Ungrounded, not refuted: we learned nothing about
        # whether the diff is wrong.
        return CounterexampleResult(
            CounterexampleStatus.UNAVAILABLE, f"claim did not run: {parsed['error'][:300]}"
        )

    observed = parsed.get("observed") or {}
    raised = parsed.get("raised") or {}

    # CHECK 1 -- claimed values (or claimed exceptions) against reality.
    for call in claim["calls"]:
        name = call["name"]
        if call["expect_raises"] is not None:
            actually_raised = raised.get(name)
            if actually_raised != call["expect_raises"]:
                return CounterexampleResult(
                    CounterexampleStatus.REFUTED,
                    f"claimed {claim['function']}(**{call['args']}) raises "
                    f"{call['expect_raises']}, but it "
                    + (
                        f"raised {actually_raised}"
                        if actually_raised
                        else f"returned {observed.get(name)!r} without raising"
                    ),
                    observed,
                )
            continue
        if name in raised:
            # The claim asserted a value; the call blew up instead. The
            # claim is contradicted, not un-runnable -- we DID learn the
            # stated value is not what happens.
            return CounterexampleResult(
                CounterexampleStatus.REFUTED,
                f"claimed {name}={call['expect']!r} but "
                f"{claim['function']}(**{call['args']}) raised {raised[name]}",
                observed,
            )
        if call["expect"] is None:
            continue
        actual = observed.get(name)
        if not _matches(call["expect"], actual):
            return CounterexampleResult(
                CounterexampleStatus.REFUTED,
                f"claimed {name}={call['expect']!r} but "
                f"{claim['function']}(**{call['args']}) actually returns {actual!r}",
                observed,
            )

    # CHECK 2 -- the claimed relation, against the REAL values.
    if claim["claim"]:
        if raised:
            return CounterexampleResult(
                CounterexampleStatus.UNAVAILABLE,
                f"cannot evaluate {claim['claim']!r}: {sorted(raised)} raised "
                f"({raised}), so there is no value to compare",
                observed,
            )
        try:
            holds = evaluate_relation(claim["claim"], observed)
        except InvalidClaim as exc:
            return CounterexampleResult(
                CounterexampleStatus.UNAVAILABLE, f"claim relation invalid: {exc}", observed
            )
        if not holds:
            return CounterexampleResult(
                CounterexampleStatus.REFUTED,
                f"claimed {claim['claim']!r} but the real values are "
                f"{observed} -- the relation does not hold",
                observed,
            )

    return CounterexampleResult(
        CounterexampleStatus.VERIFIED,
        f"executed against the real code: {observed}"
        + (f", and {claim['claim']!r} holds" if claim["claim"] else ""),
        observed,
    )
