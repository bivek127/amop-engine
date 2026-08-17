"""Milestone 8 — Fix Indexing at Scale, Re-Validate.

Three tiers, same discipline as every prior milestone (spec 19.2/19.3):

  * Deterministic, always-on, pure (tiers a/b below): batch_by_budget()'s
    list-splitting algorithm and indexer.py's _prepare_for_embedding()
    truncation policy. Both are pure functions, no I/O, no Ollama needed.
  * Real-Ollama, opt-in via AMOP_E2E_OLLAMA=1 (tier c): the actual
    Milestone 7 crash, reproduced and confirmed fixed, against the real
    pyinvoke/invoke chunk corpus -- not synthetic text, since tokenizer
    behavior on repetitive/lorem-ipsum content isn't guaranteed
    equivalent to real Python (test_milestone5.py's own embedding tests
    are unconditional because they're a 2-string call; these are not
    that cheap by design -- they're deliberately at crash-reproducing
    scale, so they follow the file's cost-based reasoning, not a
    blanket rule).

Requires: Postgres at TEST_DATABASE_URL for tier c's DB test. Tier c also
needs a running Ollama and the real amop-invoke-scratch checkout at
AMOP_INVOKE_REPO (default: the local path from Milestone 7's setup).
"""

import os
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text

from amop.codebase_intel.chunker import Chunk, chunk_source
from amop.codebase_intel.embeddings import embed_texts
from amop.codebase_intel.indexer import (
    MAX_CHUNK_CHARS_FOR_EMBEDDING,
    _TRUNCATION_MARKER,
    _prepare_for_embedding,
    _walk_python_files,
    index_repo,
)
from amop.database.models import CodeChunk
from amop.database.session import init_db, make_engine, make_session_factory
from amop.models.ollama import (
    EMBED_MAX_CHARS_PER_REQUEST,
    EMBED_MAX_ITEMS_PER_REQUEST,
    batch_by_budget,
)

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://localhost/amop_test"
)
# Milestone 7/8's real scratch checkout -- mirrors TEST_DATABASE_URL's
# os.environ.get(..., default) pattern exactly.
AMOP_INVOKE_REPO = Path(
    os.environ.get(
        "AMOP_INVOKE_REPO", "/Users/bivekmohanbhattarai/bmb/amop-invoke-scratch"
    )
)


# ---------------------------------------------------------------------
# Tier (a) — batch_by_budget, pure, no I/O
# ---------------------------------------------------------------------


def test_batch_by_budget_respects_char_limit():
    batches = batch_by_budget(["a" * 10] * 5, max_chars=25, max_items=100)
    assert [len(b) for b in batches] == [2, 2, 1]
    assert all(sum(len(t) for t in b) <= 25 for b in batches)


def test_batch_by_budget_respects_item_limit():
    batches = batch_by_budget(["a"] * 10, max_chars=10_000, max_items=3)
    assert [len(b) for b in batches] == [3, 3, 3, 1]


def test_batch_by_budget_singleton_oversized_item_never_dropped():
    # An item alone bigger than max_chars still gets its own batch --
    # never dropped, never causes a zero-progress infinite loop.
    batches = batch_by_budget(["x" * 50, "y", "z"], max_chars=10, max_items=100)
    assert batches[0] == ["x" * 50]
    all_items = [t for b in batches for t in b]
    assert all_items == ["x" * 50, "y", "z"]


def test_batch_by_budget_empty_input():
    assert batch_by_budget([], max_chars=10, max_items=10) == []


def test_batch_by_budget_preserves_order():
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    flat = [t for b in batch_by_budget(texts, max_chars=5, max_items=100) for t in b]
    assert flat == texts


def test_batch_by_budget_single_item_is_one_trivial_batch():
    # search.py's query-time embed_texts([query])[0] depends on exactly
    # this: a 1-item input must round-trip as one batch, unchanged.
    batches = batch_by_budget(["a single query string"], max_chars=100, max_items=100)
    assert batches == [["a single query string"]]


def test_batch_by_budget_never_exceeds_either_limit_on_mixed_sizes():
    # Regression guard for the real crash shape: many items of varying
    # size, both limits must be respected simultaneously.
    texts = [("x" * (i % 50 + 1)) for i in range(500)]
    batches = batch_by_budget(texts, max_chars=1000, max_items=50)
    for b in batches:
        assert len(b) <= 50
        assert sum(len(t) for t in b) <= 1000 or len(b) == 1  # singleton exempt
    assert [t for b in batches for t in b] == texts


# ---------------------------------------------------------------------
# Tier (b) — indexer._prepare_for_embedding, pure, no I/O
# ---------------------------------------------------------------------


def _chunk(content: str, symbol_name: str = "f") -> Chunk:
    return Chunk(
        file_path="x.py",
        symbol_name=symbol_name,
        symbol_type="function",
        start_line=1,
        end_line=10,
        content=content,
    )


def test_prepare_for_embedding_returns_unchanged_when_under_cap():
    chunk = _chunk("short content")
    assert _prepare_for_embedding(chunk) == "short content"


def test_prepare_for_embedding_returns_unchanged_at_exact_cap():
    chunk = _chunk("z" * MAX_CHUNK_CHARS_FOR_EMBEDDING)
    assert _prepare_for_embedding(chunk) == "z" * MAX_CHUNK_CHARS_FOR_EMBEDDING


def test_prepare_for_embedding_truncates_and_warns_when_over_cap():
    chunk = _chunk("z" * (MAX_CHUNK_CHARS_FOR_EMBEDDING + 1000), symbol_name="Runner.run")
    with pytest.warns(UserWarning, match=r"x\.py::Runner\.run.*truncating"):
        result = _prepare_for_embedding(chunk)
    assert result.startswith("z" * 100)  # kept the start (signature/docstring end)
    assert result.endswith(_TRUNCATION_MARKER)
    assert len(result) == MAX_CHUNK_CHARS_FOR_EMBEDDING + len(_TRUNCATION_MARKER)


def test_prepare_for_embedding_never_shrinks_below_cap_length():
    # The truncated text (minus the marker) is exactly the cap -- not
    # accidentally shorter from an off-by-one.
    chunk = _chunk("a" * 50_000)
    result = _prepare_for_embedding(chunk)
    content_part = result[: -len(_TRUNCATION_MARKER)]
    assert len(content_part) == MAX_CHUNK_CHARS_FOR_EMBEDDING


# ---------------------------------------------------------------------
# Tier (c) — real Ollama, opt-in via AMOP_E2E_OLLAMA=1
# ---------------------------------------------------------------------


def _load_real_invoke_chunks() -> list[Chunk]:
    assert AMOP_INVOKE_REPO.is_dir(), (
        f"AMOP_INVOKE_REPO ({AMOP_INVOKE_REPO}) not found -- set the env var "
        "to a local checkout of bivek127/amop-invoke-scratch"
    )
    all_chunks: list[Chunk] = []
    for file in _walk_python_files(AMOP_INVOKE_REPO):
        relative = str(file.relative_to(AMOP_INVOKE_REPO))
        try:
            source = file.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        try:
            all_chunks.extend(chunk_source(source, relative))
        except SyntaxError:
            continue
    return all_chunks


@pytest.mark.skipif(
    os.environ.get("AMOP_E2E_OLLAMA") != "1",
    reason="real-model E2E: set AMOP_E2E_OLLAMA=1 (needs Ollama running)",
)
async def test_real_repo_scale_embedding_no_longer_crashes():
    """The actual Milestone 7 failure, reproduced against the real corpus
    that caused it (not synthetic text), now expected to succeed."""
    chunks = _load_real_invoke_chunks()
    assert len(chunks) > 1000, f"expected real invoke-scale corpus, got {len(chunks)} chunks"

    prepared = [_prepare_for_embedding(c) for c in chunks]
    embeddings = await embed_texts(prepared)

    assert len(embeddings) == len(chunks)
    assert all(len(e) == 768 for e in embeddings)


@pytest.mark.skipif(
    os.environ.get("AMOP_E2E_OLLAMA") != "1",
    reason="real-model E2E: set AMOP_E2E_OLLAMA=1 (needs Ollama running)",
)
async def test_oversized_chunk_embeds_end_to_end_without_crash():
    chunk = _chunk("x = 1\n" * 5000, symbol_name="huge_synthetic_function")
    assert len(chunk.content) > MAX_CHUNK_CHARS_FOR_EMBEDDING

    prepared = _prepare_for_embedding(chunk)
    embeddings = await embed_texts([prepared])

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768


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
        await conn.execute(text("TRUNCATE code_chunks"))


@pytest.mark.skipif(
    os.environ.get("AMOP_E2E_OLLAMA") != "1",
    reason="real-model E2E: set AMOP_E2E_OLLAMA=1 (needs Ollama running)",
)
async def test_reindex_real_invoke_repo_populates_code_chunks(session):
    """Done-When #1's actual bar: a real code_chunks row count in
    Postgres, not just index_repo()'s returned int."""
    assert AMOP_INVOKE_REPO.is_dir()
    repo_path = str(AMOP_INVOKE_REPO)

    chunk_count = await index_repo(session, repo_path, AMOP_INVOKE_REPO)
    assert chunk_count > 1000

    result = await session.execute(
        select(func.count()).select_from(CodeChunk).where(CodeChunk.repo_path == repo_path)
    )
    row_count = result.scalar_one()
    assert row_count == chunk_count

    # Spot-check: the known-large chunk (Runner.run) is stored in full,
    # not truncated -- only its embedding vector was based on truncated
    # text, per _prepare_for_embedding's contract.
    result = await session.execute(
        select(CodeChunk).where(
            CodeChunk.repo_path == repo_path,
            CodeChunk.symbol_name == "Runner.run",
        )
    )
    runner_run = result.scalar_one_or_none()
    if runner_run is not None:
        assert len(runner_run.content) > MAX_CHUNK_CHARS_FOR_EMBEDDING
        assert _TRUNCATION_MARKER not in runner_run.content
