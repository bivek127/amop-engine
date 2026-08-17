"""Milestone 11 — Targeted Edits (`patch_file`).

Milestone 10 closed the search/strategy bugs that kept Coder from ever
finding the right file, but surfaced the wall behind them: write_file
needs the whole file in context to safely rewrite it, and a real file
(invoke/runners.py, 65KB) doesn't fit. patch_file (spec 6.3.3, 8.3)
sidesteps that structurally -- apply a unified diff to specific lines,
never needing the whole file in context.

Applied via `git apply` (already in the sandbox image since Milestone 4,
for branching/committing) -- no hand-rolled diff parsing. The only new
logic is header normalization (rewriting/synthesizing the --- / +++
lines to unambiguously target the `path` argument, regardless of what
the model wrote there) -- git still does 100% of the actual patching and
conflict detection.

Tiers, same discipline as every prior milestone: real-Docker-sandbox
tests here (no mock sandbox path, per Milestone 3's own discipline) need
no Ollama; the real end-to-end proof against invoke/runners.py is opt-in
via AMOP_E2E_OLLAMA=1, run live once for the human's own verification,
matching Milestones 5/8/10's convention for real-repo/real-model checks.
"""

import json
import os
import uuid
from pathlib import Path

import pytest

from amop.agents.coder import CoderAgent
from amop.agents.handoffs import RootCauseReport
from amop.models.base import BaseLLM, ModelResponse
from amop.sandbox import repo as git_repo
from amop.sandbox.manager import SandboxManager
from amop.tools.registry import ToolContext, get_tool, invoke_tool

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "buggy_calculator"
AMOP_INVOKE_REPO = Path(
    os.environ.get(
        "AMOP_INVOKE_REPO", "/Users/bivekmohanbhattarai/bmb/amop-invoke-scratch"
    )
)

FIVE_LINES = "line 1\nline 2\nline 3\nline 4\nline 5\n"

CLEAN_DIFF = """--- a/target.txt
+++ b/target.txt
@@ -1,5 +1,5 @@
 line 1
 line 2
-line 3
+line THREE
 line 4
 line 5
"""

BARE_PATH_DIFF = """--- target.txt
+++ target.txt
@@ -1,5 +1,5 @@
 line 1
 line 2
-line 3
+line THREE
 line 4
 line 5
"""

HEADERLESS_DIFF = """@@ -1,5 +1,5 @@
 line 1
 line 2
-line 3
+line THREE
 line 4
 line 5
"""

STALE_DIFF = """--- a/target.txt
+++ b/target.txt
@@ -1,5 +1,5 @@
 line 1
 line 2
-this context does not match the real file
+line THREE
 line 4
 line 5
"""


# ---------------------------------------------------------------------
# Tier (a) — patch_file itself, real Docker sandbox (Milestone 3's "no
# mock sandbox path" discipline)
# ---------------------------------------------------------------------


@pytest.fixture
def sandbox_manager():
    manager = SandboxManager()
    yield manager
    manager.destroy_all()


def make_ctx(scratch_dir, sandbox, mode="suggestor"):
    return ToolContext(agent_name="test", scratch_dir=scratch_dir, mode=mode, sandbox=sandbox)


async def test_patch_file_applies_a_clean_diff(tmp_path, sandbox_manager):
    (tmp_path / "target.txt").write_text(FIVE_LINES)
    sandbox = sandbox_manager.create("t-patch-clean", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "patch_file", {"path": "target.txt", "diff": CLEAN_DIFF}, ctx, agent_name="test"
        )
        assert result.success, result.message
        assert (tmp_path / "target.txt").read_text() == (
            "line 1\nline 2\nline THREE\nline 4\nline 5\n"
        )
    finally:
        sandbox_manager.destroy("t-patch-clean")


async def test_patch_file_rejects_stale_diff_without_corrupting_the_file(tmp_path, sandbox_manager):
    (tmp_path / "target.txt").write_text(FIVE_LINES)
    sandbox = sandbox_manager.create("t-patch-stale", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "patch_file", {"path": "target.txt", "diff": STALE_DIFF}, ctx, agent_name="test"
        )
        assert not result.success
        assert result.error_code == "PATCH_CONFLICT"
        # File untouched -- git apply --check caught it before any write.
        assert (tmp_path / "target.txt").read_text() == FIVE_LINES
    finally:
        sandbox_manager.destroy("t-patch-stale")


async def test_patch_file_normalizes_bare_path_headers(tmp_path, sandbox_manager):
    (tmp_path / "target.txt").write_text(FIVE_LINES)
    sandbox = sandbox_manager.create("t-patch-bare", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "patch_file", {"path": "target.txt", "diff": BARE_PATH_DIFF}, ctx, agent_name="test"
        )
        assert result.success, result.message
        assert "line THREE" in (tmp_path / "target.txt").read_text()
    finally:
        sandbox_manager.destroy("t-patch-bare")


async def test_patch_file_synthesizes_missing_headers(tmp_path, sandbox_manager):
    (tmp_path / "target.txt").write_text(FIVE_LINES)
    sandbox = sandbox_manager.create("t-patch-headerless", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "patch_file", {"path": "target.txt", "diff": HEADERLESS_DIFF}, ctx, agent_name="test"
        )
        assert result.success, result.message
        assert "line THREE" in (tmp_path / "target.txt").read_text()
    finally:
        sandbox_manager.destroy("t-patch-headerless")


async def test_patch_file_missing_target_is_not_found(tmp_path, sandbox_manager):
    sandbox = sandbox_manager.create("t-patch-missing", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "patch_file",
            {"path": "does_not_exist.txt", "diff": CLEAN_DIFF},
            ctx,
            agent_name="test",
        )
        assert not result.success
        assert result.error_code == "NOT_FOUND"
    finally:
        sandbox_manager.destroy("t-patch-missing")


async def test_patch_file_outside_scratch_dir_denied(tmp_path, sandbox_manager):
    (tmp_path / "target.txt").write_text(FIVE_LINES)
    sandbox = sandbox_manager.create("t-patch-escape", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "patch_file",
            {"path": "../../etc/passwd", "diff": CLEAN_DIFF},
            ctx,
            agent_name="test",
        )
        assert not result.success
        assert result.error_code in ("PATH_NOT_PERMITTED", "DENIED")
    finally:
        sandbox_manager.destroy("t-patch-escape")


# ---------------------------------------------------------------------
# Tier (a continued) — read_file(with_line_numbers=True), real sandbox
# ---------------------------------------------------------------------


async def test_read_file_with_line_numbers_prefixes_whole_file(tmp_path, sandbox_manager):
    (tmp_path / "target.txt").write_text(FIVE_LINES)
    sandbox = sandbox_manager.create("t-lineno-whole", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "read_file", {"path": "target.txt", "with_line_numbers": True}, ctx, agent_name="test"
        )
        assert result.success
        assert result.output == "1: line 1\n2: line 2\n3: line 3\n4: line 4\n5: line 5\n"
    finally:
        sandbox_manager.destroy("t-lineno-whole")


async def test_read_file_with_line_numbers_prefixes_a_range_using_real_line_numbers(
    tmp_path, sandbox_manager
):
    (tmp_path / "target.txt").write_text(FIVE_LINES)
    sandbox = sandbox_manager.create("t-lineno-range", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "read_file",
            {"path": "target.txt", "start_line": 3, "end_line": 4, "with_line_numbers": True},
            ctx,
            agent_name="test",
        )
        assert result.success
        # Prefix reflects the file's real line numbers (3, 4), not 1, 2 --
        # this is the whole point: knowing the true @@ header start line.
        assert result.output == "3: line 3\n4: line 4\n"
    finally:
        sandbox_manager.destroy("t-lineno-range")


async def test_read_file_default_behavior_unchanged_from_milestone_10(tmp_path, sandbox_manager):
    # with_line_numbers omitted entirely -- must be byte-identical to
    # Milestone 10's existing, tested, unprefixed contract.
    (tmp_path / "target.txt").write_text(FIVE_LINES)
    sandbox = sandbox_manager.create("t-lineno-default", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        whole = await invoke_tool("read_file", {"path": "target.txt"}, ctx, agent_name="test")
        ranged = await invoke_tool(
            "read_file", {"path": "target.txt", "start_line": 2, "end_line": 3}, ctx, agent_name="test"
        )
        assert whole.output == FIVE_LINES
        assert ranged.output == "line 2\nline 3\n"
    finally:
        sandbox_manager.destroy("t-lineno-default")


# ---------------------------------------------------------------------
# Tier (a continued) — Milestone 11's own fix: a whole-file read on a
# genuinely large file (>24,000 chars) returns a bounded scoped default
# instead of the full content, avoiding the failure mode confirmed live
# against the real runners.py case (a prepended notice on the full
# content doesn't get acted on -- the model's own context truncation
# drops the instruction along with everything else before it can act).
# ---------------------------------------------------------------------


def _make_numbered_lines_file(n: int) -> str:
    return "".join(f"content of line {i}\n" for i in range(1, n + 1))


async def test_read_file_whole_file_over_threshold_returns_scoped_default_not_full_content(
    tmp_path, sandbox_manager
):
    # Each line is ~20 chars; 2000 lines is well over the 24,000-char
    # threshold and has clearly distinguishable early vs. late content.
    big = _make_numbered_lines_file(2000)
    assert len(big) > 24_000
    (tmp_path / "big.py").write_text(big)
    sandbox = sandbox_manager.create("t-scoped-default", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool("read_file", {"path": "big.py"}, ctx, agent_name="test")

        assert result.success
        assert result.output.startswith("[NOTE:")
        assert "content of line 1\n" in result.output
        assert "1: content of line 1" in result.output  # line-numbered
        assert "60: content of line 60" in result.output
        # The defining behavior: content past the scoped default is
        # genuinely absent, not just "hopefully truncated by the model."
        assert "content of line 61" not in result.output
        assert "content of line 2000" not in result.output
    finally:
        sandbox_manager.destroy("t-scoped-default")


async def test_read_file_just_under_scoped_threshold_still_returns_full_content(
    tmp_path, sandbox_manager
):
    # Below the new, higher bar -- Milestone 10's existing notice-plus-
    # full-content behavior for moderately large files is unaffected.
    medium = "x" * 10_000  # over the 8K notice threshold, under 24K
    (tmp_path / "medium.py").write_text(medium)
    sandbox = sandbox_manager.create("t-scoped-under", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool("read_file", {"path": "medium.py"}, ctx, agent_name="test")

        assert result.success
        assert result.output.startswith("[NOTE:")
        assert medium in result.output  # full content still present
    finally:
        sandbox_manager.destroy("t-scoped-under")


async def test_read_file_ranged_read_of_a_huge_file_bypasses_the_scoped_default(
    tmp_path, sandbox_manager
):
    # A caller who already knows what range it wants (e.g. from
    # search_code) shouldn't be second-guessed by the whole-file-only
    # scoped-default logic.
    big = _make_numbered_lines_file(2000)
    (tmp_path / "big2.py").write_text(big)
    sandbox = sandbox_manager.create("t-scoped-ranged", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "read_file",
            {"path": "big2.py", "start_line": 500, "end_line": 502},
            ctx,
            agent_name="test",
        )
        assert result.success
        assert not result.output.startswith("[NOTE:")
        assert result.output == "content of line 500\ncontent of line 501\ncontent of line 502\n"
    finally:
        sandbox_manager.destroy("t-scoped-ranged")


async def test_read_file_scoped_default_on_missing_file_is_not_found(tmp_path, sandbox_manager):
    sandbox = sandbox_manager.create("t-scoped-missing", tmp_path)
    try:
        ctx = make_ctx(tmp_path, sandbox)
        result = await invoke_tool(
            "read_file", {"path": "does_not_exist.py"}, ctx, agent_name="test"
        )
        assert not result.success
        assert result.error_code == "NOT_FOUND"
    finally:
        sandbox_manager.destroy("t-scoped-missing")


# ---------------------------------------------------------------------
# Tier (b) — CoderAgent wiring, pure (no I/O)
# ---------------------------------------------------------------------


def test_coder_tools_include_patch_file():
    assert "patch_file" in CoderAgent.tools
    assert "write_file" in CoderAgent.tools  # still available, not replaced


def test_coder_system_prompt_explains_patch_file_preference():
    class _Stub(BaseLLM):
        async def complete(self, messages, tools=None):
            raise NotImplementedError

        async def embed(self, texts):
            raise NotImplementedError

    ctx = ToolContext(agent_name="coder", scratch_dir=Path("."), mode="suggestor")
    coder = CoderAgent(_Stub(), ctx, task_id="t1")

    prompt = coder.system_prompt()

    assert "prefer patch_file over write_file" in prompt
    assert "PATCH_CONFLICT" in prompt
    assert "patch_file" in prompt


# ---------------------------------------------------------------------
# Tier (c) — Coder actually calling patch_file mid-loop, scripted model,
# real Docker sandbox (mirrors test_milestone4.py's write_file
# container-path integration test)
# ---------------------------------------------------------------------


class ScriptedLLM(BaseLLM):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, messages, tools=None) -> ModelResponse:
        content = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return ModelResponse(content=content, model="scripted")

    async def embed(self, texts):
        raise NotImplementedError


def tool_call(name: str, **arguments) -> str:
    return json.dumps({"tool_call": {"name": name, "arguments": arguments}})


def final(answer) -> str:
    return json.dumps({"final_answer": answer})


@pytest.fixture
def workspace(tmp_path):
    scratch = tmp_path / "workspace"
    git_repo.materialize(FIXTURE_REPO, scratch)

    manager = SandboxManager()
    task_id = f"m11-{uuid.uuid4().hex[:8]}"
    sandbox = manager.create(task_id, scratch)
    try:
        git_repo.init_baseline(sandbox)
        git_repo.create_branch(sandbox, "amop/fix-test")
        ctx = ToolContext(
            agent_name="chain", scratch_dir=scratch, mode="operator", sandbox=sandbox
        )
        yield ctx, scratch, sandbox
    finally:
        manager.destroy(task_id)


async def test_coder_can_apply_a_patch_mid_loop(workspace):
    ctx, scratch, _sandbox = workspace
    original_lines = (scratch / "calculator.py").read_text().splitlines(keepends=True)

    # A real, minimal diff against the actual fixture file: replace the
    # buggy average() denominator, using a couple of lines of real
    # surrounding context on each side -- exactly the "targeted, not
    # whole-file" shape this milestone is about. Built from the file's
    # own real lines (not hand-typed whitespace) so indentation matches
    # exactly.
    buggy_idx = next(
        i for i, line in enumerate(original_lines) if "len(values) - 1" in line
    )
    context_before = original_lines[buggy_idx - 1]  # the raise ValueError(...) line
    buggy_line = original_lines[buggy_idx]
    fixed_line = buggy_line.replace("len(values) - 1", "len(values)")
    start = buggy_idx  # 0-indexed -> 1-indexed start line is buggy_idx (context_before's line number)

    body = f" {context_before}-{buggy_line}+{fixed_line}"
    diff = (
        "--- a/calculator.py\n+++ b/calculator.py\n"
        f"@@ -{start},2 +{start},2 @@\n"
        f"{body}"
    )

    llm = ScriptedLLM([tool_call("patch_file", path="calculator.py", diff=diff), final("done")])
    coder = CoderAgent(llm, ctx, task_id="t1")

    result = await coder.run("fix the average() off-by-one bug")

    assert result.success, result.error
    assert any(c["name"] == "patch_file" and c["success"] for c in result.tool_calls), result.tool_calls
    patched = (scratch / "calculator.py").read_text()
    assert "len(values) - 1" not in patched
    assert "len(values)" in patched


# ---------------------------------------------------------------------
# Tier (d) — real Ollama, real repo, opt-in via AMOP_E2E_OLLAMA=1. The
# actual Milestone 8/10 blocker (invoke/runners.py, 65KB) -- does Coder
# now construct and apply a working patch. Not run in default pytest;
# run live, once, for the human's own review (this milestone's own bar).
# ---------------------------------------------------------------------

# Recovered verbatim from the real Milestone 8 transcript, reused
# identically in Milestone 10's diagnosis -- kept as one real, consistent
# repro across milestones rather than a fresh synthetic case each time.
ROOT_CAUSE = (
    "The `invoke` package is hardcoded to expect the shell to be located "
    "at `/bin/bash`, which does not exist on NixOS where the shell is "
    "located at `/run/current-system/sw/bin/bash`. This hardcoded path "
    "causes a `FileNotFoundError` when trying to run shell commands."
)
SUGGESTED_FIX_PLAN = (
    "In invoke/runners.py, stop hardcoding '/bin/bash' as the shell path. "
    "Resolve the shell at runtime (e.g. via shutil.which('bash') or "
    "os.environ, falling back to '/bin/sh') so it works on systems like "
    "NixOS where bash isn't at that fixed path."
)


@pytest.mark.skipif(
    os.environ.get("AMOP_E2E_OLLAMA") != "1",
    reason="real-model E2E: set AMOP_E2E_OLLAMA=1 (needs Ollama running)",
)
async def test_real_model_patches_the_real_large_file_that_defeated_it_before():
    from amop.models.ollama import DEFAULT_MODEL, OllamaProvider

    assert AMOP_INVOKE_REPO.is_dir(), (
        f"AMOP_INVOKE_REPO ({AMOP_INVOKE_REPO}) not found -- set the env var "
        "to a local checkout of bivek127/amop-invoke-scratch"
    )
    scratch = Path("/tmp") / f"amop-m11-e2e-{uuid.uuid4().hex[:8]}"
    git_repo.materialize(AMOP_INVOKE_REPO, scratch)

    manager = SandboxManager()
    task_id = f"m11-e2e-{uuid.uuid4().hex[:8]}"
    sandbox = manager.create(task_id, scratch)
    try:
        git_repo.init_baseline(sandbox)
        git_repo.create_branch(sandbox, "amop/fix-e2e")
        ctx = ToolContext(
            agent_name="chain", scratch_dir=scratch, mode="operator", sandbox=sandbox
        )
        report = RootCauseReport(
            task_id=task_id,
            root_cause=ROOT_CAUSE,
            confidence=0.9,
            affected_files=["invoke/runners.py"],
            suggested_fix_plan=SUGGESTED_FIX_PLAN,
        )
        prompt = (
            f"Root cause: {report.root_cause}\n"
            f"Suggested fix plan: {report.suggested_fix_plan}\n\n"
            "Affected files (from the investigation -- read these directly "
            "with read_file first; only use search_code if you need more "
            f"context beyond them): {report.affected_files}\n\n"
            "Apply the smallest change that fixes this root cause. Fix the "
            "source code, not the tests. The repository is at /workspace."
        )
        provider = OllamaProvider(model=DEFAULT_MODEL)
        coder = CoderAgent(provider, ctx, task_id=task_id)

        result = await coder.run(prompt)

        mutating = [
            c for c in result.tool_calls
            if c["success"] and (spec := get_tool(c["name"])) is not None and spec.mutating
        ]
        assert mutating, (
            f"expected at least one successful mutating tool call, got: {result.tool_calls}"
        )
        changed = (scratch / "invoke" / "runners.py").read_text()
        assert "/bin/bash" not in changed or "which" in changed or "environ" in changed, (
            "expected the hardcoded /bin/bash default to be resolved at "
            "runtime instead -- inspect the actual diff manually"
        )
    finally:
        manager.destroy(task_id)
