"""Chunking — spec Section 7.2, Design Decision D-5, scoped to Python-only
via the stdlib `ast` module per CLAUDE.md's Milestone 5 authorization
(tree-sitter's multi-language grammar setup is out of scope for one
milestone).

D-5's rule: a chunk boundary follows the AST, never a fixed token window
and never a mid-function cut -- "the single most common cause of
retrieval returning code that doesn't make sense in isolation." One
function/method is one chunk; one class is one chunk unless it exceeds
chunk_max_tokens, in which case it splits into one chunk per method.

Metadata per chunk matches CLAUDE.md's item 1 exactly: file_path,
symbol_name, symbol_type, start_line, end_line. One deliberate addition
beyond that literal list: a `module`-type chunk for each file's
leftover top-level code (imports, module docstring, top-level
constants) that isn't inside any function or class -- without it that
code would silently never be indexed at all.
"""

import ast
from dataclasses import dataclass

CHUNK_MAX_TOKENS = 800


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
