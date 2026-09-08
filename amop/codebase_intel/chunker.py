"""Chunking — spec Section 7.2, Design Decision D-5. Milestone 5 scoped
this to Python-only via the stdlib `ast` module; Milestone 25 adds a
second backend (tree-sitter + tree-sitter-javascript) for `.js`,
dispatched by file extension in `chunk_source`. Each backend produces
the same `Chunk` shape, so nothing downstream (indexer, embedding,
search) needs to know which language a chunk came from.

D-5's rule: a chunk boundary follows the AST, never a fixed token window
and never a mid-function cut -- "the single most common cause of
retrieval returning code that doesn't make sense in isolation." One
function/method is one chunk; one class is one chunk unless it exceeds
chunk_max_tokens, in which case it splits into one chunk per method.
Both backends hold to this rule identically.

Metadata per chunk matches CLAUDE.md's item 1 exactly: file_path,
symbol_name, symbol_type, start_line, end_line. One deliberate addition
beyond that literal list: a `module`-type chunk for each file's
leftover top-level code (imports, module docstring, top-level
constants) that isn't inside any function or class -- without it that
code would silently never be indexed at all. Both backends produce it.
"""

import ast
from dataclasses import dataclass

import tree_sitter
import tree_sitter_javascript

CHUNK_MAX_TOKENS = 800

_JS_LANGUAGE = tree_sitter.Language(tree_sitter_javascript.language())


def _estimate_tokens(text: str) -> int:
    """~4 chars/token, a common rough heuristic. Deliberately not tied to
    any specific model's real tokenizer -- this project has no tokenizer
    dependency, and the split threshold only needs to be roughly right,
    not exact."""
    return len(text) // 4


@dataclass
class Chunk:
    file_path: str
    symbol_name: str
    symbol_type: str  # "function" | "async_function" | "class" | "method" | "module"
    start_line: int
    end_line: int
    content: str


def _slice_source(lines: list[str], start_line: int, end_line: int) -> str:
    """1-indexed, inclusive on both ends -- matches ast's lineno/end_lineno."""
    return "\n".join(lines[start_line - 1 : end_line])


def _function_type(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"


def _start_line(node: ast.AST) -> int:
    """node.lineno points at `def`/`class`, not any decorators above it --
    decorator_list entries have their own, earlier line numbers. Without
    this, a decorator like @staticmethod or @property would be silently
    dropped from the chunk's content and leak into the module-level
    leftover chunk instead (caught before it shipped, not after)."""
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        return min(d.lineno for d in decorators)
    return node.lineno


def _chunk_class(
    node: ast.ClassDef, file_path: str, lines: list[str]
) -> list[Chunk]:
    start = _start_line(node)
    whole_body = _slice_source(lines, start, node.end_lineno)
    if _estimate_tokens(whole_body) <= CHUNK_MAX_TOKENS:
        return [
            Chunk(
                file_path=file_path,
                symbol_name=node.name,
                symbol_type="class",
                start_line=start,
                end_line=node.end_lineno,
                content=whole_body,
            )
        ]

    # Over budget: one chunk per method instead (D-5) -- never a
    # mid-function cut, so this is the finest granularity available, not
    # a token-window fallback.
    chunks = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            item_start = _start_line(item)
            chunks.append(
                Chunk(
                    file_path=file_path,
                    symbol_name=f"{node.name}.{item.name}",
                    symbol_type="method",
                    start_line=item_start,
                    end_line=item.end_lineno,
                    content=_slice_source(lines, item_start, item.end_lineno),
                )
            )
    return chunks


def chunk_source(source: str, file_path: str) -> list[Chunk]:
    """Dispatch by file extension to the right backend. `.py` -> the
    original ast-based path (Milestone 5), `.js` -> tree-sitter
    (Milestone 25). Anything else raises -- callers (indexer.py) only
    ever call this for extensions they've already decided to index, so
    an unrecognized extension reaching here is a caller bug, not a
    file to skip silently."""
    if file_path.endswith(".py"):
        return _chunk_python(source, file_path)
    if file_path.endswith(".js"):
        return _chunk_javascript(source, file_path)
    raise ValueError(f"chunk_source: unsupported file extension: {file_path!r}")


def _chunk_python(source: str, file_path: str) -> list[Chunk]:
    """Chunk one Python file's source. Raises SyntaxError on unparseable
    source -- callers (indexer.py) decide whether to skip-and-warn or
    propagate; this function itself doesn't swallow the error, so a test
    can assert on it directly."""
    tree = ast.parse(source, filename=file_path)
    lines = source.splitlines()

    chunks: list[Chunk] = []
    covered_lines: set[int] = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = _start_line(node)
            chunks.append(
                Chunk(
                    file_path=file_path,
                    symbol_name=node.name,
                    symbol_type=_function_type(node),
                    start_line=start,
                    end_line=node.end_lineno,
                    content=_slice_source(lines, start, node.end_lineno),
                )
            )
            covered_lines.update(range(start, node.end_lineno + 1))
        elif isinstance(node, ast.ClassDef):
            start = _start_line(node)
            class_chunks = _chunk_class(node, file_path, lines)
            chunks.extend(class_chunks)
            covered_lines.update(range(start, node.end_lineno + 1))

    # Leftover top-level lines (imports, module docstring, constants,
    # any top-level statement not inside a def/class) become one
    # "module" chunk, so nothing silently falls out of the index.
    leftover_lines = [
        i for i in range(1, len(lines) + 1) if i not in covered_lines and lines[i - 1].strip()
    ]
    if leftover_lines:
        chunks.append(
            Chunk(
                file_path=file_path,
                symbol_name=file_path,
                symbol_type="module",
                start_line=min(leftover_lines),
                end_line=max(leftover_lines),
                content="\n".join(lines[i - 1] for i in leftover_lines),
            )
        )

    return chunks


# --- JavaScript backend (Milestone 25) -------------------------------
#
# tree-sitter is error-tolerant by design -- it always returns *some*
# tree, marking bad regions as error nodes rather than raising. To keep
# chunk_source's contract identical across backends (SyntaxError on
# unparseable source, not a best-effort partial chunk set), a JS parse
# whose root reports has_error is turned into an explicit SyntaxError
# here, mirroring _chunk_python's propagation exactly.
#
# Line numbers: tree-sitter's start_point/end_point are 0-indexed, with
# end_point being exclusive-ish (it points at the position just past the
# node, which for a `}`-terminated construct lands on the same row as
# the closing brace) -- empirically verified against _slice_source's
# 1-indexed inclusive convention: start_point.row + 1 / end_point.row +
# 1 gives exactly the right bounds, same as ast's lineno/end_lineno.

_JS_TOP_LEVEL_TYPES = {"function_declaration", "class_declaration", "lexical_declaration"}


def _js_span(node: "tree_sitter.Node") -> tuple[int, int]:
    return node.start_point.row + 1, node.end_point.row + 1


def _js_is_async(node: "tree_sitter.Node") -> bool:
    return any(c.type == "async" for c in node.children)


def _js_unwrap_export(node: "tree_sitter.Node") -> "tree_sitter.Node | None":
    """`export function foo() {}` / `export default class {}` wrap the
    real declaration as a child. Unwrapping it lets exported and
    non-exported top-level forms dispatch identically below. `export {
    a, b }` (re-export list) and `export default <expression>` (e.g. an
    anonymous arrow function) have no function/class/lexical child to
    unwrap to -- returns None, and the caller leaves those lines to fall
    into the leftover "module" chunk, same as any other top-level
    statement chunk_source doesn't specifically recognize."""
    for child in node.children:
        if child.type in _JS_TOP_LEVEL_TYPES:
            return child
    return None


def _chunk_js_class(
    node: "tree_sitter.Node",
    file_path: str,
    lines: list[str],
    outer_start: int,
    outer_end: int,
) -> list[Chunk]:
    """Mirrors _chunk_class: whole class as one chunk unless it exceeds
    CHUNK_MAX_TOKENS, in which case one chunk per method (D-5 -- never a
    mid-function cut, so per-method is the finest split available)."""
    name_node = node.child_by_field_name("name")
    class_name = name_node.text.decode("utf-8") if name_node is not None else "<anonymous>"
    whole_body = _slice_source(lines, outer_start, outer_end)
    if _estimate_tokens(whole_body) <= CHUNK_MAX_TOKENS:
        return [
            Chunk(
                file_path=file_path,
                symbol_name=class_name,
                symbol_type="class",
                start_line=outer_start,
                end_line=outer_end,
                content=whole_body,
            )
        ]

    chunks = []
    body = node.child_by_field_name("body")
    if body is not None:
        for member in body.children:
            if member.type != "method_definition":
                continue
            member_name_node = member.child_by_field_name("name")
            if member_name_node is None:
                continue
            m_start, m_end = _js_span(member)
            chunks.append(
                Chunk(
                    file_path=file_path,
                    symbol_name=f"{class_name}.{member_name_node.text.decode('utf-8')}",
                    symbol_type="method",
                    start_line=m_start,
                    end_line=m_end,
                    content=_slice_source(lines, m_start, m_end),
                )
            )
    return chunks


def _chunk_javascript(source: str, file_path: str) -> list[Chunk]:
    """Chunk one JavaScript file via tree-sitter. Raises SyntaxError on
    unparseable source, matching _chunk_python's contract exactly."""
    parser = tree_sitter.Parser(_JS_LANGUAGE)
    tree = parser.parse(source.encode("utf-8"))
    if tree.root_node.has_error:
        raise SyntaxError(f"tree-sitter reported a parse error in {file_path!r}")

    lines = source.splitlines()
    chunks: list[Chunk] = []
    covered_lines: set[int] = set()

    for node in tree.root_node.children:
        outer_start, outer_end = _js_span(node)
        target = node
        if node.type == "export_statement":
            unwrapped = _js_unwrap_export(node)
            if unwrapped is None:
                continue
            target = unwrapped

        if target.type == "function_declaration":
            name_node = target.child_by_field_name("name")
            if name_node is None:
                continue  # `export default function() {}` -- anonymous, not in our fixtures
            chunks.append(
                Chunk(
                    file_path=file_path,
                    symbol_name=name_node.text.decode("utf-8"),
                    symbol_type="async_function" if _js_is_async(target) else "function",
                    start_line=outer_start,
                    end_line=outer_end,
                    content=_slice_source(lines, outer_start, outer_end),
                )
            )
            covered_lines.update(range(outer_start, outer_end + 1))

        elif target.type == "class_declaration":
            chunks.extend(_chunk_js_class(target, file_path, lines, outer_start, outer_end))
            covered_lines.update(range(outer_start, outer_end + 1))

        elif target.type == "lexical_declaration":
            # const/let/var. Only declarators whose value is a function
            # expression or arrow function become their own chunk (a
            # function assigned to a name is chunked the same as one
            # declared with `function name() {}` -- both are "a
            # function," per D-5's own reasoning); a plain value (`const
            # X = 1`) is left uncovered and falls into the leftover
            # "module" chunk below, same as Python's top-level constants.
            found_function = False
            for declarator in target.children:
                if declarator.type != "variable_declarator":
                    continue
                value = declarator.child_by_field_name("value")
                if value is None or value.type not in ("arrow_function", "function"):
                    continue
                name_node = declarator.child_by_field_name("name")
                if name_node is None:
                    continue
                found_function = True
                if _js_is_async(value):
                    symbol_type = "async_function"
                elif value.type == "arrow_function":
                    symbol_type = "arrow_function"
                else:
                    symbol_type = "function"
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        symbol_name=name_node.text.decode("utf-8"),
                        symbol_type=symbol_type,
                        start_line=outer_start,
                        end_line=outer_end,
                        content=_slice_source(lines, outer_start, outer_end),
                    )
                )
            if found_function:
                covered_lines.update(range(outer_start, outer_end + 1))

    # Leftover top-level lines -- same rule as Python's module chunk.
    leftover_lines = [
        i for i in range(1, len(lines) + 1) if i not in covered_lines and lines[i - 1].strip()
    ]
    if leftover_lines:
        chunks.append(
            Chunk(
                file_path=file_path,
                symbol_name=file_path,
                symbol_type="module",
                start_line=min(leftover_lines),
                end_line=max(leftover_lines),
                content="\n".join(lines[i - 1] for i in leftover_lines),
            )
        )

    return chunks
