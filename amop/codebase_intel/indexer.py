"""Repository Indexing — spec Section 7.1, trimmed to Milestone 5's
declared scope: `.py` files only, freshly re-indexed every run rather
than incrementally (both explicit CLAUDE.md simplifications). Milestone
25 widens the walk to `.py` + `.js` and adds `detect_stack`, spec
6.3.4's "primary language(s) by file-extension histogram" -- previously
`repositories.detected_stack` existed only as an unpopulated DB column
(Milestone 20 added the column, nothing ever wrote it).

Reads files directly from the host-side scratch directory rather than
issuing one sandbox.exec_run/read_file per matched file. This is
deliberate, not a shortcut around Section 9's isolation model: the
scratch directory *is* the live, bind-mounted content of /workspace --
host and container see byte-identical bytes in real time (Milestone 3's
own mount design). Indexing is an orchestrator-level, non-agent-driven,
read-only operation, the same trust tier as sandbox/repo.py's
materialize() (which already reads/writes that directory directly) --
not an LLM tool call, so it doesn't need the container-exec boundary
that exists specifically to gate *agent-driven* actions. Contrast with
codebase_intel/search.py's keyword half, which agents DO call mid-task
and which does go through sandbox.exec_run, to keep that boundary
consistent for every agent-facing tool.
"""

import warnings
from pathlib import Path

import pathspec
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from amop.codebase_intel.chunker import Chunk, _estimate_tokens, chunk_source
from amop.codebase_intel.embeddings import embed_texts
from amop.database.models import CodeChunk

_ALWAYS_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv"}

# Milestone 25: extension -> language name, used by both detect_stack
# (image/routing decision) and the source walk below (which extensions
# get indexed at all). Deliberately just the two spec 6.3.4 calls for --
# framework/manifest sniffing (package.json dependency inspection) is
# explicitly out of scope per CLAUDE.md's Milestone 25 scope decisions;
# this milestone only needs "is this repo Python or JavaScript."
_LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    # Milestone 26: `.ts` and `.tsx` are one *language* here even though
    # chunker.py parses them with two different grammars -- the grammar
    # split (`<T>` is ambiguous between a type assertion and a JSX
    # element) is a parsing detail, not a stack detail. Both run in the
    # same Node sandbox image, so collapsing them keeps image selection
    # honest and simple.
    ".ts": "typescript",
    ".tsx": "typescript",
}

# Milestone 8: chunker.py's own CHUNK_MAX_TOKENS=800 budget is only
# enforced on classes (splitting an over-budget class into one chunk per
# method) -- standalone functions, individual methods, and the "module"
# leftover chunk have no further cap ("never a mid-function cut...
# finest granularity available" per chunker.py's own docstring). A real
# chunk can still come out huge: pyinvoke/invoke's Runner.run (a single
# method) is 12,674 chars. Ollama's /api/embed silently truncates an
# oversized individual item under its own default (truncate=true,
# confirmed in its docs) -- but that's an implicit, undocumented-limit
# provider default we don't want to be the only protection. This cap is
# our own explicit, visible one: nomic-embed-text's 2048-token context
# window via chunker.py's own len(text)//4 heuristic (2048*4=8192 chars),
# with a small safety margin under it.
MAX_CHUNK_CHARS_FOR_EMBEDDING = 8_000


def _load_gitignore(source_dir: Path) -> pathspec.PathSpec | None:
    gitignore = source_dir / ".gitignore"
    if not gitignore.is_file():
        return None
    lines = gitignore.read_text().splitlines()
    return pathspec.PathSpec.from_lines("gitignore", lines)


def _walk_source_files(source_dir: Path) -> list[Path]:
    """Every `.py`/`.js` file under source_dir, respecting .gitignore
    (Section 7.1) via pathspec's real gitwildmatch semantics -- not a
    hand-rolled fnmatch that would get negation/anchoring/directory
    patterns subtly wrong. Milestone 5 walked `.py` only; Milestone 25
    widens to every extension chunk_source knows how to dispatch."""
    spec = _load_gitignore(source_dir)
    matches = []
    for pattern in (f"*{ext}" for ext in _LANGUAGE_EXTENSIONS):
        for path in source_dir.rglob(pattern):
            relative = path.relative_to(source_dir)
            if any(part in _ALWAYS_SKIP_DIRS for part in relative.parts):
                continue
            if spec is not None and spec.match_file(str(relative)):
                continue
            matches.append(path)
    return sorted(matches)


def detect_stack(source_dir: Path) -> dict:
    """Spec 6.3.4: "primary language(s) by file-extension histogram."
    Deliberately just an extension count, not manifest/framework
    sniffing (package.json dependency inspection is explicitly out of
    scope -- CLAUDE.md's Milestone 25 scope decisions) -- this only
    needs to answer "is this repo Python or JavaScript" well enough to
    pick a chunker and a sandbox image, nothing more.

    Called from two places for one reason: sandbox image selection
    (chain.py) has to happen before index_repo can run at all -- the
    container has to exist before indexing can populate it -- so
    detection can't be a byproduct of indexing here. Cheap enough
    (a single directory walk) that running it twice costs nothing
    concurrency- or correctness-relevant.

    Returns {"languages": {name: file_count, ...}, "primary": name |
    None}. `primary` is None only for an empty/unrecognized repo.
    """
    source_dir = Path(source_dir)
    spec = _load_gitignore(source_dir)
    languages: dict[str, int] = {}
    for ext, language in _LANGUAGE_EXTENSIONS.items():
        for path in source_dir.rglob(f"*{ext}"):
            relative = path.relative_to(source_dir)
            if any(part in _ALWAYS_SKIP_DIRS for part in relative.parts):
                continue
            if spec is not None and spec.match_file(str(relative)):
                continue
            languages[language] = languages.get(language, 0) + 1

    primary = max(languages, key=languages.get) if languages else None
    return {"languages": languages, "primary": primary}


_TRUNCATION_MARKER = "\n# [AMOP: content truncated for embedding -- see indexer.py]"


def _prepare_for_embedding(chunk: Chunk) -> str:
    """The text actually sent to the embedding model for this chunk.

    Milestone 8 kept only the head (first MAX_CHUNK_CHARS_FOR_EMBEDDING
    chars) on the theory that a function's signature/docstring/opening
    logic is usually the most useful part to keep. Milestone 10's
    real-repo diagnosis found a concrete counterexample: pyinvoke/
    invoke's Runner.run (12,674 chars, one method, chunker.py never
    splits below function granularity) has its actually-relevant line --
    the hardcoded '/bin/bash' default -- at char offset 10,328, past the
    old 8,000-char head-only cutoff. Its embedding never represented
    that text, search_code's ranking never surfaced the file for a query
    about it, and the Coder gave up without ever finding the right file.

    Splitting the same char budget between head and tail catches both
    "the interesting part is the setup" and "the interesting part is
    deep in the body/return path" without exceeding the embedding
    model's own context window (still the same total char budget as
    before -- see MAX_CHUNK_CHARS_FOR_EMBEDDING's own reasoning).

    This is ONLY ever used as embed_texts() input. The full, untruncated
    chunk.content is still what gets stored in CodeChunk.content and
    what search_code shows an agent -- a lower-fidelity embedding vector
    for one chunk is an acceptable tradeoff; losing a real chunk from the
    index or from what's shown to an agent is not.
    """
    content = chunk.content
    if len(content) <= MAX_CHUNK_CHARS_FOR_EMBEDDING:
        return content

    warnings.warn(
        f"chunk {chunk.file_path}::{chunk.symbol_name} ({chunk.symbol_type}) "
        f"is {len(content)} chars (~{_estimate_tokens(content)} est. tokens), "
        f"keeping {MAX_CHUNK_CHARS_FOR_EMBEDDING} chars (head+tail) for "
        "embedding -- the full chunk is still indexed and shown to agents "
        "unchanged, only its embedding vector is based on the reduced text",
        stacklevel=2,
    )
    half = MAX_CHUNK_CHARS_FOR_EMBEDDING // 2
    head = content[:half]
    tail = content[-half:]
    return f"{head}{_TRUNCATION_MARKER}{tail}"


async def index_repo(
    session: AsyncSession, repo_path: str, source_dir: Path
) -> int:
    """Chunk + embed + store every .py/.js file under source_dir, tagged
    with `repo_path` as the stable index identity (the resolved SOURCE
    repo path -- see database/models.py's CodeChunk docstring for why
    that, not the ephemeral scratch dir, is the right key).

    "Index fresh each time" (CLAUDE.md's authorized simplification over
    Section 7.1's incremental re-index): every call deletes all existing
    rows for this repo_path first, so re-running never accumulates
    duplicate/stale chunks for the same repo.

    Returns the number of chunks stored. A file that fails to parse
    (SyntaxError) is skipped with a warning rather than aborting the
    whole run -- one bad file shouldn't block indexing the other 13.
    """
    source_dir = Path(source_dir)
    await session.execute(delete(CodeChunk).where(CodeChunk.repo_path == repo_path))

    all_chunks = []
    for file in _walk_source_files(source_dir):
        relative_path = str(file.relative_to(source_dir))
        try:
            source = file.read_text()
            chunks = chunk_source(source, relative_path)
        except SyntaxError as exc:
            warnings.warn(f"skipping unparseable file {relative_path}: {exc}", stacklevel=2)
            continue
        all_chunks.extend(chunks)

    if not all_chunks:
        await session.commit()
        return 0

    embeddings = await embed_texts([_prepare_for_embedding(c) for c in all_chunks])

    for chunk, embedding in zip(all_chunks, embeddings, strict=True):
        session.add(
            CodeChunk(
                repo_path=repo_path,
                file_path=chunk.file_path,
                symbol_name=chunk.symbol_name,
                symbol_type=chunk.symbol_type,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content=chunk.content,
                embedding=embedding,
            )
        )
    await session.commit()
    return len(all_chunks)
