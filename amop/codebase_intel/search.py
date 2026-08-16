"""Search Tools — spec Section 7.3, trimmed to Milestone 5's declared
scope: `search_code(query, repo_path, mode="hybrid")` only (no
`find_references`/`get_file_outline` -- CLAUDE.md's item 6 asks for
search_code alone), and hybrid without the cross-encoder rerank pass
(reciprocal rank fusion only).

Semantic and keyword deliberately read from two different places, and
that split is itself the (partial) answer to Section 7.3.1's staleness
problem -- which CLAUDE.md otherwise defers in full:

  - Semantic searches the vector index (Postgres), built once before the
    chain starts. It can go stale the moment Coder edits a file.
  - Keyword greps the live sandbox filesystem via sandbox.exec_run, so it
    always reflects whatever Coder has actually written, no matter how
    stale the index has become.

Hybrid mode gets both for free by simply running both. This isn't the
full dirty-file bypass (D-14) or index consistency model (7.3.2) -- there
is no per-file dirty-tracking here, no provenance envelope on results,
and semantic results can still be stale -- but it is a real, deliberate
reason the two halves aren't just "index twice", not an accident of
where the data happens to live.
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from amop.codebase_intel.embeddings import embed_texts
from amop.database.models import CodeChunk
from amop.tools.registry import ToolContext, ToolResult, tool

SEMANTIC_LIMIT = 10
KEYWORD_LIMIT = 10
HYBRID_RETURN = 5
RRF_K = 60  # spec 7.3's own constant


@dataclass
class SearchResult:
    file_path: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    content: str
    score: float

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "score": self.score,
        }


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Merge N ranked lists of keys (rank 0 = best) into one fused
    ranking by RRF score = sum over lists of 1/(k + rank). A key absent
    from a list contributes nothing from that list -- it doesn't need to
    appear in all of them.

    Pure function, no I/O -- independently unit-testable with the same
    discipline as safety/engine.py's evaluate() and chain.py's routing
    functions. Returns (key, score) pairs sorted best-first.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


async def _semantic_search(
    query: str, repo_path: str, db_session: AsyncSession, limit: int = SEMANTIC_LIMIT
) -> list[CodeChunk]:
    query_vector = (await embed_texts([query]))[0]
    stmt = (
        select(CodeChunk)
        .where(CodeChunk.repo_path == repo_path)
        .order_by(CodeChunk.embedding.cosine_distance(query_vector))
        .limit(limit)
    )
    result = await db_session.execute(stmt)
    return list(result.scalars().all())


def _shell_quote(value: str) -> str:
    escaped = value.replace("'", "'\\''")
    return f"'{escaped}'"


async def _keyword_search(
    query: str,
    repo_path: str,
    sandbox,
    db_session: AsyncSession,
    limit: int = KEYWORD_LIMIT,
) -> list[CodeChunk]:
    """Literal, case-insensitive grep against the live sandbox filesystem
    -- Section 7.3's "exact for known symbol/string search" mode.
    Multi-word natural-language queries will typically match nothing
    here, by design (see module docstring): that's the intended contrast
    with semantic search, not a bug in this half.

    Each grep hit (file:line) is resolved back to the code_chunks row
    that contains that line, so keyword and semantic results share one
    identity space (chunk id) for reciprocal_rank_fusion() to merge --
    otherwise "line 42 in foo.py" and "the chunk covering line 42" would
    never be recognized as the same thing.
    """
    if not query.strip():
        return []

    result = sandbox.exec_run(
        f"grep -rn -i -F {_shell_quote(query)} /workspace --include='*.py'",
        timeout=15,
    )
    if result.exit_code not in (0, 1):  # 1 == grep found nothing, not an error
        return []

    hits: list[tuple[str, int]] = []
    seen = set()
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 2:
            continue
        raw_path, raw_line = parts[0], parts[1]
        if not raw_line.isdigit():
            continue
        file_path = raw_path.removeprefix("/workspace/").removeprefix("./")
        key = (file_path, int(raw_line))
        if key in seen:
            continue
        seen.add(key)
        hits.append(key)
        if len(hits) >= limit:
            break

    chunks: list[CodeChunk] = []
    for file_path, line_number in hits:
        stmt = (
            select(CodeChunk)
            .where(
                CodeChunk.repo_path == repo_path,
                CodeChunk.file_path == file_path,
                CodeChunk.start_line <= line_number,
                CodeChunk.end_line >= line_number,
            )
            .limit(1)
        )
        chunk = (await db_session.execute(stmt)).scalars().first()
        if chunk is not None:
            chunks.append(chunk)
    return chunks


def _chunk_key(chunk: CodeChunk) -> str:
    return f"{chunk.file_path}:{chunk.symbol_name}:{chunk.start_line}"


async def search_code_impl(
    query: str,
    repo_path: str,
    sandbox,
    db_session: AsyncSession,
    mode: str = "hybrid",
) -> list[SearchResult]:
    if mode not in ("semantic", "keyword", "hybrid"):
        # The message itself is the fix for a model that guesses a mode
        # name anyway despite the schema/description (observed live: a
        # real model tried 'natural-language', 'exact-symbol', 'symbol',
        # 'exact', 'default' in sequence, never right, burning its whole
        # tool-call budget on this alone before ever writing a file).
        # Naming the valid options IN the error is what lets a model
        # self-correct on its next attempt instead of guessing again.
        raise ValueError(
            f"unknown search mode {mode!r} -- must be exactly one of: "
            "'semantic', 'keyword', 'hybrid' (or omit mode entirely for "
            "the 'hybrid' default)"
        )

    semantic_chunks = (
        await _semantic_search(query, repo_path, db_session)
        if mode in ("semantic", "hybrid")
        else []
    )
    keyword_chunks = (
        await _keyword_search(query, repo_path, sandbox, db_session)
        if mode in ("keyword", "hybrid")
        else []
    )

    by_key = {_chunk_key(c): c for c in (*semantic_chunks, *keyword_chunks)}

    if mode == "semantic":
        ordered = semantic_chunks
    elif mode == "keyword":
        ordered = keyword_chunks
    else:
        fused = reciprocal_rank_fusion(
            [
                [_chunk_key(c) for c in semantic_chunks],
                [_chunk_key(c) for c in keyword_chunks],
            ]
        )
        ordered = [by_key[key] for key, _score in fused]

    top = ordered[:HYBRID_RETURN]
    return [
        SearchResult(
            file_path=c.file_path,
            symbol_name=c.symbol_name,
            symbol_type=c.symbol_type,
            start_line=c.start_line,
            end_line=c.end_line,
            content=c.content,
            score=1.0,  # RRF/cosine scores aren't meaningfully comparable across modes; rank order is what matters
        )
        for c in top
    ]


@tool(
    name="search_code",
    description=(
        "Search the repository for code relevant to a natural-language "
        "query or an exact symbol/string. Returns up to 5 relevant code "
        "chunks (function/method/class) with file path and line range. "
        "Search first, then use read_file on the specific candidates "
        "this returns, instead of reading files one by one. "
        "The 'mode' argument is OPTIONAL -- omit it, the default "
        "('hybrid') is what you want almost always. If you do set it, "
        "it must be exactly one of: 'semantic', 'keyword', 'hybrid'. No "
        "other value is valid -- there is no 'exact' or 'symbol' mode."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "mode": {
                "type": "string",
                "enum": ["semantic", "keyword", "hybrid"],
            },
        },
        "required": ["query"],
    },
    mutating=False,
    timeout_seconds=30,
)
async def search_code(query: str, ctx: ToolContext, mode: str = "hybrid") -> ToolResult:
    if ctx.repo_path is None or ctx.db_session is None:
        return ToolResult(
            success=False,
            error_code="INDEX_UNAVAILABLE",
            message="no repo index available for this task",
        )
    if ctx.sandbox is None and mode in ("keyword", "hybrid"):
        return ToolResult(
            success=False,
            error_code="SANDBOX_UNAVAILABLE",
            message="no sandbox session for this task",
        )

    try:
        results = await search_code_impl(
            query, ctx.repo_path, ctx.sandbox, ctx.db_session, mode=mode
        )
    except ValueError as exc:
        return ToolResult(success=False, error_code="INVALID_ARGS", message=str(exc))
    except Exception as exc:
        return ToolResult(success=False, error_code="SEARCH_ERROR", message=str(exc))

    return ToolResult(success=True, output=[r.to_dict() for r in results])
