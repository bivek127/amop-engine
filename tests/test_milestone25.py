"""Milestone 25 — Multi-Language Support (JavaScript). Spec Section 7.2
(chunking, D-5), 7.1 (indexing/detect_stack), 9.2 (per-stack images),
8.6 (run_tests).

Every module touched here was already language-agnostic except
chunker.py (Milestone 5, Python-only via `ast`) -- this file proves
that claim for real on JavaScript rather than leaving it asserted.
"""

import json
import os
import shutil
import uuid
from pathlib import Path

import pytest

from amop.codebase_intel.chunker import CHUNK_MAX_TOKENS, chunk_source
from amop.sandbox.manager import PIDS_LIMIT, SandboxManager
from amop.sandbox.tools import _parse_jest_json, run_tests
from amop.tools.registry import ToolContext

# No `pytestmark = pytest.mark.asyncio` needed -- pyproject.toml's
# asyncio_mode = "auto" already covers every `async def test_*` below.


# =======================================================================
# Checkpoint 1 -- JS chunking (no DB, no Docker) and language detection
# =======================================================================


def test_js_chunk_function_class_and_arrow_boundaries():
    """Mirrors test_milestone5.py's test_chunk_function_and_class_
    boundaries exactly, in JS: a plain function, an arrow function
    assigned to a const, a class with a method, and a leftover
    module-level chunk that doesn't double-count either."""
    src = (
        'import { readFile } from "fs";\n'
        "const CONST = 1;\n\n"
        "function foo(x) {\n"
        "  return x + 1;\n"
        "}\n\n"
        "const bar = (y) => {\n"
        "  return y * 2;\n"
        "};\n\n"
        "class Widget {\n"
        "  constructor() {\n"
        "    this.x = 1;\n"
        "  }\n"
        "}\n"
    )
    chunks = chunk_source(src, "widget.js")
    by_name = {c.symbol_name: c for c in chunks}

    assert by_name["foo"].symbol_type == "function"
    assert by_name["foo"].content == "function foo(x) {\n  return x + 1;\n}"

    assert by_name["bar"].symbol_type == "arrow_function"
    assert "return y * 2;" in by_name["bar"].content

    assert by_name["Widget"].symbol_type == "class"
    assert "constructor" in by_name["Widget"].content

    module_chunk = by_name["widget.js"]
    assert module_chunk.symbol_type == "module"
    assert 'import { readFile }' in module_chunk.content
    assert "CONST = 1" in module_chunk.content
    assert "function foo" not in module_chunk.content  # not double-counted


def test_js_chunk_async_function_and_exported_forms():
    src = (
        "async function slow(x) {\n"
        "  return x;\n"
        "}\n\n"
        "export function baz(z) {\n"
        "  return z - 1;\n"
        "}\n\n"
        "export class Gadget {\n"
        "  run() { return 1; }\n"
        "}\n"
    )
    chunks = chunk_source(src, "x.js")
    by_name = {c.symbol_name: c for c in chunks}

    assert by_name["slow"].symbol_type == "async_function"
    assert by_name["baz"].symbol_type == "function"
    assert by_name["baz"].content.startswith("export function baz")
    assert by_name["Gadget"].symbol_type == "class"
    assert by_name["Gadget"].content.startswith("export class Gadget")


def test_js_chunk_splits_oversized_class_into_per_method_chunks():
    """Mirrors test_milestone5.py's identical Python test -- same
    padding trick, same assertions, proving D-5's split rule holds for
    the tree-sitter backend too, not just ast."""
    padding = "x" * 50
    methods = "\n\n".join(
        f"  method_{i}() {{\n    // {padding}\n    return {i};\n  }}"
        for i in range(40)
    )
    src = f"class Big {{\n{methods}\n}}\n"
    assert len(src) // 4 > CHUNK_MAX_TOKENS  # sanity: genuinely over budget

    chunks = chunk_source(src, "big.js")
    assert len(chunks) == 40
    assert {c.symbol_type for c in chunks} == {"method"}
    assert chunks[0].symbol_name == "Big.method_0"


def test_js_chunk_syntax_error_propagates():
    """tree-sitter is error-tolerant by design (always returns *some*
    tree) -- chunk_source has to convert a bad parse into SyntaxError
    itself so callers (indexer.py) can't mistake a parse failure for a
    real, empty file."""
    with pytest.raises(SyntaxError):
        chunk_source("function broken(: {\n", "broken.js")


def test_chunk_source_rejects_an_unrecognized_extension():
    with pytest.raises(ValueError):
        chunk_source("print('hi')", "script.rb")


def test_detect_stack_reports_a_real_extension_histogram(tmp_path):
    from amop.codebase_intel.indexer import detect_stack

    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    (tmp_path / "c.js").write_text("const z = 3;\n")

    result = detect_stack(tmp_path)
    assert result["languages"] == {"python": 2, "javascript": 1}
    assert result["primary"] == "python"


def test_detect_stack_respects_gitignore(tmp_path):
    """node_modules is already in the indexer's always-skip set, but
    prove the .gitignore path independently too -- a vendored .js file
    excluded by the repo's own .gitignore must not count toward the
    histogram, same rule the source walk already follows."""
    from amop.codebase_intel.indexer import detect_stack

    (tmp_path / ".gitignore").write_text("vendor/\n")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.js").write_text("const ignored = 1;\n")
    (tmp_path / "app.js").write_text("const kept = 1;\n")

    result = detect_stack(tmp_path)
    assert result["languages"] == {"javascript": 1}


def test_detect_stack_on_empty_repo_has_no_primary(tmp_path):
    from amop.codebase_intel.indexer import detect_stack

    result = detect_stack(tmp_path)
    assert result["languages"] == {}
    assert result["primary"] is None


# =======================================================================
# Checkpoint 2 -- Node sandbox image isolation (real Docker daemon
# required, same bar as test_milestone3.py's original Python-image
# tests -- confirmed independently, not assumed inherited)
# =======================================================================


@pytest.fixture
def sandbox_manager():
    manager = SandboxManager()
    yield manager
    manager.destroy_all()


def test_node_image_runs_as_non_root_with_the_right_toolbelt(tmp_path, sandbox_manager):
    sandbox = sandbox_manager.create("t-node-whoami", tmp_path, stack="javascript")
    try:
        result = sandbox.exec_run("whoami && id -u && node --version && jest --version")
        assert result.exit_code == 0
        lines = result.stdout.strip().splitlines()
        assert lines[0] == "amop"
        assert lines[1] == "1000"  # non-root uid, not 0
        assert lines[2].startswith("v")  # node --version
        assert lines[3][0].isdigit()  # jest --version
    finally:
        sandbox_manager.destroy("t-node-whoami")


def test_node_container_cannot_read_a_real_host_secret(tmp_path, sandbox_manager):
    """Mirrors test_milestone3.py's identical Python-image test."""
    host_secret_dir = tmp_path.parent / f"host-secret-node-{tmp_path.name}"
    host_secret_dir.mkdir()
    secret_path = host_secret_dir / "secret.txt"
    secret_path.write_text("do-not-leak-me")
    assert secret_path.exists()  # sanity: it's real, on the host

    scratch = tmp_path / "scratch"
    sandbox = sandbox_manager.create("t-node-host-secret", scratch, stack="javascript")
    try:
        result = sandbox.exec_run(f"cat {secret_path} 2>&1; echo EXIT:$?")
        assert "do-not-leak-me" not in result.stdout
        assert "EXIT:0" not in result.stdout
    finally:
        sandbox_manager.destroy("t-node-host-secret")


def test_node_container_write_outside_workspace_never_reaches_host(tmp_path, sandbox_manager):
    sandbox = sandbox_manager.create("t-node-write-escape", tmp_path, stack="javascript")
    try:
        nonce = uuid.uuid4().hex
        container_path = f"/tmp/host_escape_{nonce}.txt"
        sandbox.exec_run(f"echo leaked > {container_path}")

        inside = sandbox.exec_run(f"cat {container_path}")
        assert inside.stdout.strip() == "leaked"

        assert not os.path.exists(container_path)
    finally:
        sandbox_manager.destroy("t-node-write-escape")


def test_node_network_is_disabled(tmp_path, sandbox_manager):
    """Same assertion as test_milestone3.py's test_network_is_disabled,
    but the outbound attempt is made from Node itself (`node -e ...`),
    not Python -- proves the Node *runtime* is actually subject to the
    container's network_mode="none", not merely that python3 happens to
    be absent from this image."""
    sandbox = sandbox_manager.create("t-node-network", tmp_path, stack="javascript")
    try:
        result = sandbox.exec_run(
            "node -e \"require('net').connect(53, '8.8.8.8')"
            ".on('error', e => { console.log(e.message); process.exit(1); })\" "
            ">/tmp/out 2>&1; echo EXIT:$?; cat /tmp/out",
            timeout=10,
        )
        assert "EXIT:0" not in result.stdout
        assert "unreachable" in result.stdout.lower() or "ENETUNREACH" in result.stdout
    finally:
        sandbox_manager.destroy("t-node-network")


def test_node_pids_limit_caps_process_creation(tmp_path, sandbox_manager):
    """Identical mechanism to test_milestone3.py's Python-image version
    -- the fork-bomb payload is plain shell, not language-specific, so
    it's reused verbatim against the Node image to confirm the cgroup
    limit applies regardless of which image is running."""
    host_procs_before = len(os.listdir("/proc")) if os.path.isdir("/proc") else None

    sandbox = sandbox_manager.create("t-node-pids-limit", tmp_path, stack="javascript")
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

        assert ok <= PIDS_LIMIT
        assert fail > 0

        if host_procs_before is not None:
            host_procs_after = len(os.listdir("/proc"))
            assert host_procs_after - host_procs_before < 50
    finally:
        sandbox_manager.destroy("t-node-pids-limit")


def test_create_without_stack_still_defaults_to_python_image(tmp_path, sandbox_manager):
    """Backward-compat guarantee: every pre-Milestone-25 caller of
    create() (including test_milestone3.py's own fixtures) doesn't pass
    `stack` at all -- confirm that keeps selecting the Python image, not
    a KeyError or a silent switch to Node."""
    sandbox = sandbox_manager.create("t-default-stack", tmp_path)
    try:
        result = sandbox.exec_run("python3 --version")
        assert result.exit_code == 0
        assert "Python" in result.stdout
    finally:
        sandbox_manager.destroy("t-default-stack")


# =======================================================================
# Checkpoint 3 -- Jest output normalization (no Docker; a real captured
# Jest --json payload, not a hand-built idealized one -- see
# tests/fixtures/jest_sample_output/report.json's own header comment on
# how it was produced)
# =======================================================================

_JEST_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "jest_sample_output" / "report.json"
)


def test_parse_jest_json_matches_the_normalized_shape_on_a_real_payload():
    """The payload here is a real `jest --json --testLocationInResults`
    run against a 3-test fixture (1 pass, 1 genuine failure, 1 skip) --
    captured live, not constructed -- same discipline as pytest's own
    junit-xml parsing tests. This is the actual runner-agnosticism
    proof: a structurally different report format feeding the exact
    same {passed, failed, skipped, failures} shape _parse_junit_xml
    produces."""
    json_text = _JEST_FIXTURE_PATH.read_text()
    summary = _parse_jest_json(json_text)

    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["skipped"] == 1
    assert len(summary["failures"]) == 1

    failure = summary["failures"][0]
    assert failure["test_name"] == "this one is deliberately wrong"
    assert failure["file"] == "/workspace/math.test.js"
    assert failure["line"] == 7  # --testLocationInResults' real line number
    assert "Expected: 5" in failure["message"]
    assert "Received: 4" in failure["message"]


def test_parse_jest_json_on_an_all_passing_run():
    payload = json.dumps(
        {
            "numPassedTests": 2,
            "numFailedTests": 0,
            "numPendingTests": 0,
            "testResults": [
                {
                    "name": "/workspace/ok.test.js",
                    "assertionResults": [
                        {"title": "a", "fullName": "a", "status": "passed",
                         "location": {"line": 1}, "failureMessages": []},
                        {"title": "b", "fullName": "b", "status": "passed",
                         "location": {"line": 5}, "failureMessages": []},
                    ],
                }
            ],
        }
    )
    summary = _parse_jest_json(payload)
    assert summary == {"passed": 2, "failed": 0, "skipped": 0, "failures": []}


async def test_jest_run_tests_matches_a_known_fixture_outcome(tmp_path, sandbox_manager):
    """run_tests invoked directly, no LLM -- same style as the
    orchestrator's own ground-truth call (_run_tests_ground_truth in
    chain.py). Fixture: known, documented pass/fail counts (see the
    fixture's own README.md) -- asserted exactly, not "something ran"."""
    fixture_src = Path(__file__).parent / "fixtures" / "buggy_js_calculator"
    shutil.copytree(fixture_src, tmp_path, dirs_exist_ok=True)

    sandbox = sandbox_manager.create("t-jest-run-tests", tmp_path, stack="javascript")
    try:
        ctx = ToolContext(
            agent_name="test", scratch_dir=tmp_path, mode="suggestor",
            sandbox=sandbox, stack="javascript",
        )
        result = await run_tests(ctx)
        assert result.success is True
        assert result.output["all_passed"] is False
        # Matches the fixture's own documented baseline exactly (README.md):
        # 8 passed, 2 failed, same root cause as buggy_calculator's Python
        # counterpart (average() divides by length - 1).
        assert result.output["passed"] == 8
        assert result.output["failed"] == 2
        failing_names = {f["test_name"] for f in result.output["failures"]}
        assert failing_names == {"average of three numbers", "average of single value"}
    finally:
        sandbox_manager.destroy("t-jest-run-tests")
