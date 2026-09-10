"""Milestone 26 — TypeScript + JSX Support. Spec Section 7.2 (chunking,
D-5), 7.1 (indexing/detect_stack), 9.2 (per-stack images), 8.6
(run_tests).

Extends Milestone 25's pattern rather than duplicating it: one
tree-sitter backend serving three grammars, the same Node image serving
two stacks, and the same normalized run_tests shape serving two
runners. The tests below deliberately pin the places where "extend, not
duplicate" could silently go wrong.
"""

import shutil
from pathlib import Path

import pytest
import tree_sitter
import tree_sitter_typescript

from amop.codebase_intel.chunker import _TS_LANGUAGE, _TSX_LANGUAGE, chunk_source
from amop.codebase_intel.indexer import detect_stack
from amop.sandbox.manager import NODE_IMAGE_NAME, STACK_IMAGES, SandboxManager
from amop.sandbox.tools import JEST_STACKS, run_tests
from amop.tools.registry import ToolContext

FIXTURE = Path(__file__).parent / "fixtures" / "buggy_react_cart"


# =======================================================================
# The grammar split -- load-bearing, not cosmetic
# =======================================================================


def test_tsx_source_needs_the_tsx_grammar_and_fails_under_the_typescript_one():
    """This is *why* there are two grammars rather than one, pinned as a
    fact rather than a comment: `<T>` is genuinely ambiguous between a
    type assertion and a JSX element, so the typescript grammar cannot
    parse real JSX. If a future refactor collapsed both extensions onto
    one grammar, this test is what catches it."""
    jsx_source = b'export const Badge = ({ n }: { n: number }) => <b>{n}</b>;'

    tsx_tree = tree_sitter.Parser(_TSX_LANGUAGE).parse(jsx_source)
    assert tsx_tree.root_node.has_error is False

    ts_tree = tree_sitter.Parser(_TS_LANGUAGE).parse(jsx_source)
    assert ts_tree.root_node.has_error is True


def test_the_two_grammars_are_actually_distinct_objects():
    """Guards against both extensions being wired to the same grammar by
    a copy-paste slip -- which would still pass most other tests here,
    since plain TS parses fine under the tsx grammar."""
    assert _TS_LANGUAGE is not _TSX_LANGUAGE
    assert tree_sitter_typescript.language_typescript() != (
        tree_sitter_typescript.language_tsx()
    )


# =======================================================================
# TypeScript chunking (.ts -- typescript grammar)
# =======================================================================


def test_ts_chunks_interface_type_alias_and_enum():
    src = (
        "export interface CartItem { name: string; qty: number; }\n"
        'type Currency = "USD" | "EUR";\n'
        "export enum Level { Low = 1, High = 2 }\n"
    )
    by_name = {c.symbol_name: c for c in chunk_source(src, "types.ts")}

    assert by_name["CartItem"].symbol_type == "interface"
    assert by_name["Currency"].symbol_type == "type_alias"
    assert by_name["Level"].symbol_type == "enum"


def test_ts_chunks_a_generic_function_whole():
    src = (
        "export function sumBy<T>(rows: T[], pick: (r: T) => number): number {\n"
        "  return rows.reduce((t, r) => t + pick(r), 0);\n"
        "}\n"
    )
    chunk = next(c for c in chunk_source(src, "u.ts") if c.symbol_name == "sumBy")
    assert chunk.symbol_type == "function"
    # The generic parameter list and return type annotation are part of
    # the chunk, not sliced off at the signature.
    assert "<T>" in chunk.content
    assert "): number {" in chunk.content
    assert chunk.content.rstrip().endswith("}")


def test_ts_chunks_an_abstract_class():
    """`abstract class` is a *different node type* than `class`
    (`abstract_class_declaration`) -- found by probing the real grammar,
    and easy to miss because the source differs by one keyword."""
    src = "export abstract class Base {\n  abstract run(): void;\n}\n"
    chunk = next(c for c in chunk_source(src, "b.ts") if c.symbol_name == "Base")
    assert chunk.symbol_type == "class"
    assert "abstract run(): void;" in chunk.content


def test_ts_syntax_error_propagates():
    with pytest.raises(SyntaxError):
        chunk_source("export interface Broken { name: ;\n", "broken.ts")


# =======================================================================
# TSX/JSX chunking (.tsx -- tsx grammar)
# =======================================================================


def test_tsx_chunks_a_jsx_component_and_its_props_interface():
    src = (
        'import * as React from "react";\n'
        "\n"
        "interface Props { items: string[]; label?: string; }\n"
        "\n"
        'export function CartSummary({ items, label = "Cart" }: Props) {\n'
        "  return (\n"
        '    <div className="cart">\n'
        "      <h2>{label}</h2>\n"
        "    </div>\n"
        "  );\n"
        "}\n"
    )
    by_name = {c.symbol_name: c for c in chunk_source(src, "CartSummary.tsx")}

    assert by_name["Props"].symbol_type == "interface"

    component = by_name["CartSummary"]
    assert component.symbol_type == "function"
    # JSX lives inside the function body, so it must not split the chunk:
    # the whole component including its returned markup is one unit.
    assert "<div className=" in component.content
    assert "<h2>{label}</h2>" in component.content
    assert component.content.rstrip().endswith("}")


def test_tsx_chunks_an_arrow_component():
    src = "export const Badge = ({ n }: { n: number }) => <b>{n}</b>;\n"
    chunk = next(c for c in chunk_source(src, "Badge.tsx") if c.symbol_name == "Badge")
    assert chunk.symbol_type == "arrow_function"
    assert "<b>{n}</b>" in chunk.content


def test_tsx_module_chunk_does_not_swallow_the_component():
    src = (
        'import * as React from "react";\n'
        "export function App() {\n"
        "  return <div />;\n"
        "}\n"
    )
    by_name = {c.symbol_name: c for c in chunk_source(src, "App.tsx")}
    module_chunk = by_name["App.tsx"]
    assert module_chunk.symbol_type == "module"
    assert 'import * as React' in module_chunk.content
    assert "function App" not in module_chunk.content  # not double-counted


# =======================================================================
# Language detection + stack/image/runner wiring
# =======================================================================


def test_detect_stack_reports_typescript_for_both_ts_and_tsx(tmp_path):
    (tmp_path / "a.ts").write_text("export const x = 1;\n")
    (tmp_path / "b.tsx").write_text("export const B = () => <i />;\n")

    result = detect_stack(tmp_path)
    assert result["languages"] == {"typescript": 2}
    assert result["primary"] == "typescript"


def test_typescript_maps_to_the_same_node_image_not_a_third_one():
    """Milestone 26's scope decision, pinned: TS reuses M25's Node image
    rather than adding a third image."""
    assert STACK_IMAGES["typescript"][0] == NODE_IMAGE_NAME
    assert STACK_IMAGES["typescript"] == STACK_IMAGES["javascript"]


def test_typescript_runs_under_jest_not_pytest():
    """The dispatch bug this milestone would otherwise have introduced
    silently. `run_tests` branched on `ctx.stack == "javascript"`; a
    "typescript" stack would have fallen through to `python -m pytest`
    inside the Node image, failing far downstream with `python: not
    found` rather than at the dispatch. Mutation-verified: removing
    "typescript" from JEST_STACKS reproduces exactly that."""
    assert "typescript" in JEST_STACKS
    assert "javascript" in JEST_STACKS
    assert "python" not in JEST_STACKS


# =======================================================================
# The fixture, end to end (real Docker daemon required)
# =======================================================================


@pytest.fixture
def sandbox_manager():
    manager = SandboxManager()
    yield manager
    manager.destroy_all()


def test_the_fixture_chunks_both_grammars():
    """Done-When #1 at the chunker level; the Postgres-level proof is
    the indexer run shown in the milestone's live verification."""
    ts_chunks = chunk_source((FIXTURE / "src" / "types.ts").read_text(), "types.ts")
    ts_by_name = {c.symbol_name: c for c in ts_chunks}
    assert ts_by_name["CartItem"].symbol_type == "interface"
    assert ts_by_name["Currency"].symbol_type == "type_alias"
    assert ts_by_name["sumBy"].symbol_type == "function"

    tsx_chunks = chunk_source(
        (FIXTURE / "src" / "CartSummary.tsx").read_text(), "CartSummary.tsx"
    )
    tsx_by_name = {c.symbol_name: c for c in tsx_chunks}
    assert tsx_by_name["CartSummaryProps"].symbol_type == "interface"
    assert tsx_by_name["CartSummary"].symbol_type == "function"
    # The seeded bug is inside the component's chunk, which is what makes
    # it findable by search_code at all.
    assert "items.length &&" in tsx_by_name["CartSummary"].content


async def test_jest_runs_the_tsx_fixture_at_its_documented_baseline(
    tmp_path, sandbox_manager
):
    """run_tests invoked directly, no LLM -- the same style as the
    orchestrator's own ground-truth call. Asserts the fixture's exact
    documented baseline (README.md: 5 passed, 1 failed), including which
    test fails, not just that something failed."""
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    sandbox = sandbox_manager.create("t-m26-jest", tmp_path, stack="typescript")
    try:
        ctx = ToolContext(
            agent_name="test",
            scratch_dir=tmp_path,
            mode="suggestor",
            sandbox=sandbox,
            stack="typescript",
        )
        result = await run_tests(ctx)
    finally:
        sandbox_manager.destroy("t-m26-jest")

    assert result.success is True
    assert result.output["passed"] == 5
    assert result.output["failed"] == 1
    assert result.output["all_passed"] is False
    failure = result.output["failures"][0]
    assert failure["test_name"] == "renders no count element at all for an empty cart"
    assert failure["file"].endswith("CartSummary.test.tsx")


async def test_the_seeded_bug_is_actually_fixable(tmp_path, sandbox_manager):
    """A fixture whose bug can't be fixed would make every live run a
    false negative. Applies the known-good one-line fix and confirms the
    suite goes green -- so a FAILED chain run is a statement about the
    agents, not about an impossible fixture."""
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    component = tmp_path / "src" / "CartSummary.tsx"
    component.write_text(
        component.read_text().replace("{items.length &&", "{items.length > 0 &&")
    )

    sandbox = sandbox_manager.create("t-m26-fixable", tmp_path, stack="typescript")
    try:
        ctx = ToolContext(
            agent_name="test",
            scratch_dir=tmp_path,
            mode="suggestor",
            sandbox=sandbox,
            stack="typescript",
        )
        result = await run_tests(ctx)
    finally:
        sandbox_manager.destroy("t-m26-fixable")

    assert result.output["passed"] == 6
    assert result.output["failed"] == 0
    assert result.output["all_passed"] is True
