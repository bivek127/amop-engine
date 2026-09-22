# AMOP — Autonomous Multi-Agent Operations Platform

AMOP is a self-hosted platform that runs specialized LLM agents against a
real code repository. It detects bugs (via GitHub webhook, polling, or a
manually submitted report), investigates root cause, writes a fix, runs
the real test suite, reviews its own diff, and opens a pull request.
Humans control autonomy per repo and per agent — every consequential
action still passes through an approval gate before it's allowed to
happen unsupervised.

## The Rule

**LLMs decide what to do; software decides what's allowed.**

Every effectful action an agent attempts — writing a file, running a
shell command, opening a PR — passes through a deterministic Safety
Engine gate before it executes. That gate is code, not a system prompt.
"The agent is instructed not to do X" is not a safety property anywhere
in this codebase; if a rule matters, it's enforced mechanically (a
permission check, a protected-path list, a diff-size cap, a hash-chained
audit trail) in a place an LLM's output can't talk its way around. This
shows up concretely: a model can *propose* writing to `.github/workflows/`,
but the write is denied before it reaches disk, in every permission mode,
including the most permissive one — because the check is hardcoded, not
configurable away.

## Architecture

Single Python process (the Orchestrator) driving ephemeral Docker
containers for sandboxed execution, backed by Postgres. Single-host,
single-operator — this is not a microservices system.

```
Orchestrator → Agent Runtime → (loop: Model Router ⇄ Tool Registry ⇄
                                 Safety Engine ⇄ Sandbox) → Orchestrator
```

An agent (Investigator, Coder, Tester, Reviewer, Optimizer,
DependencyUpdater, Reporter, Watcher) runs inside the Agent Runtime and
proposes tool calls. Every tool call is routed through the Model Router
(which LLM handles this step), the Tool Registry (what tools exist and
their contracts), the Safety Engine (is this specific call allowed right
now, for this repo, this agent, this mode), and finally executes inside
an isolated, network-off-by-default Docker sandbox. Agents never call
each other directly — coordination only happens through the Orchestrator
via validated handoff payloads, so there's one place, not N, where a
task's state transitions are decided and recorded.

### Tech stack actually in use
- Python 3.11+, asyncio, pydantic
- Postgres + pgvector (single database for task state, audit trail,
  code embeddings, memory — no separate datastore per concern)
- Docker for sandboxed tool execution (a Python-stack image and a
  Node-stack image, selected per repo)
- FastAPI (REST API), a Telegram bot, and a Jinja2-templated web
  dashboard — all three call the same internal API layer, none has a
  privileged bypass path
- Model providers behind a `BaseLLM` interface: `OllamaProvider` (real,
  local, the default and the model every agent has been validated
  against — `qwen2.5-coder:14b`), and real `ClaudeProvider` /
  `OpenAIProvider` implementations (native tool-calling over each
  provider's own API, with a router and a one-level fallback chain) —
  built and unit-tested, but never yet run against a real paid API key
  in this project (see Current Status)
- Tree-sitter for JS/TS/TSX chunking, Python's own `ast` for Python

**Not in use, deliberately:** Redis. The original design routed events
through Redis Streams; every time that decision came up again
(concurrency locking, multi-repo slot limits, webhook idempotency) it
was re-examined and skipped in favor of a Postgres-only mechanism
(advisory locks, an in-process concurrency counter, a
`processed_events` table with a unique constraint), because a
single-process receiver doesn't need a message broker in front of a
table it already writes to transactionally. The real, named tradeoff:
no redelivery/replay if the process is down when an event arrives —
accepted for a single-operator, human-supervised system, not an
oversight.

## Current Status

### Core agent chain
A full bug-fix chain (Investigator → Coder → Tester → Reviewer → PR
creation) runs end to end against real repositories, with Postgres-backed
task persistence and the complete `bug_fix` state machine. Codebase
search is AST/tree-sitter-based (Python, JavaScript, TypeScript, TSX),
embedded via Ollama and served through pgvector hybrid search rather than
agents reading every file blind. Additional agents beyond the core
bug-fix path are built and verified: Optimizer (revert-on-no-improvement,
proven live), DependencyUpdater (real CVE fixes via OSV, with a
file-count cap that can't be talked past by a false success claim),
Reporter (counts read from the database, never trusted from a model's own
summary), and a long-term memory system (semantic retrieval on past
incidents, with dispute/undo).

### Safety & security
Every tool call is gated by the Safety Engine — a command blacklist, a
sandboxed filesystem restricted to a scratch directory, and a
hardcoded, mode-unconditional protected-path check (`.github/workflows/`,
`.git/hooks/`, CI config, etc. — cannot be relaxed by config in any
permission mode, including `autonomous`). Externally-sourced text (GitHub
issue bodies, PR comments) is tagged as untrusted before it reaches an
agent's prompt. The audit trail (state transitions and every Safety
Engine ALLOW/DENY decision) is sha256 hash-chained into a single shared
chain, so tampering with either table is detectable — demonstrated live
against the real dev database, including the specific attack a
two-separate-chains design would have missed (deleting every row of one
table while leaving the other intact). This is **tamper evidence, not
tamper prevention**: an attacker who alters a row and recomputes every
later hash forward produces a chain that still verifies clean. The one
mitigation is recording the chain's tip hash somewhere outside the
database — real, but conditional on an operator actually doing it.

### Concurrency & crash recovery
Real optimistic concurrency control (SQLAlchemy `version_id_col`) guards
every task write; file-grained Postgres advisory locks serialize
overlapping edits across concurrent tasks while letting disjoint work run
in genuine parallel — proven with real timing, not just asserted.
Multiple repositories can run concurrently with independently-enforced
per-repo permission overrides. Every successful edit is committed locally
as it happens (not just at hand-off), so a crashed orchestrator can
reconcile a task's real git state against the database on restart across
six defined divergence cases, rather than silently losing or duplicating
work — demonstrated with a real kill -9 mid-task and a real recovered
resume.

### Interfaces
A REST API, a Telegram bot, and a web dashboard all share the same
underlying task-creation and control-layer code — no interface has a
privileged bypass path. Bugs reach AMOP via manual submission (CLI, API,
Telegram, or a submit form on the dashboard), polling (`amop watch`), or
GitHub webhooks (real HMAC-SHA256 signature verification, delivery-ID
idempotency). Submitting a `bug_fix` task through the API or the
dashboard now schedules the fix chain to run in the background
automatically — you don't need a second terminal command to make it
start. The dashboard's task-detail page shows a task's real pipeline
(state transitions, stage narration, PR link, diff) rather than a raw
metadata dump.

### Evaluation
A real benchmark harness (`amop evaluate`) runs a fixed scenario suite
against existing seeded-bug fixtures and scores outcomes from ground
truth — the hash-chained transition trail plus a fresh re-run of each
fixture's own test suite — never from a chain's self-reported result. The
one clean baseline recorded so far, `qwen2.5-coder:14b` over 18 real
chain runs: **28.57% task success**, **8.33% verified fix success**, and
a **3.29x tool-call cost** for failing tasks versus succeeding ones. A
size-matched comparison against `gemma3:12b` scored 0/18 on the same
suite. These are real, low numbers, reported as measured — not smoothed
into a more flattering summary.

### Known, current limitations — stated plainly, not hidden
- **The local model can diagnose correctly and still fail to produce a
  correct edit.** Across many milestones (found first with `write_file`
  destructively overwriting a file it meant to patch, later narrowed to
  specific `patch_file` failure shapes), the pattern holds: root-cause
  diagnosis is often right; turning that diagnosis into an applying diff
  is where the local model most often breaks. Two of the diagnosed causes
  (missing trailing newline, stale-context diffs from an earlier turn or
  a prior failed retry) are now fixed and enforced mechanically. A third,
  narrower cause — the model targeting the right region but
  mistranscribing it — is identified but still open. This gap is
  **partially closed, not solved**.
- **The Reviewer's live approval rate is largely unmeasured under the
  current mechanism.** The current Reviewer actually executes a
  rejection's claimed counterexample in the sandbox rather than trusting
  its shape — and, replayed against a known historical fabricated claim,
  it correctly catches what the previous mechanism missed. But live runs
  attempting to reach that stage have repeatedly died earlier in the
  chain (the diagnosis-vs-mechanical-output gap above), so there is
  very little real approval-rate data behind the current mechanism yet.
- **Claude/OpenAI providers are implemented but not yet run against a
  real paid API in this project.** Every agent still defaults to free
  local Ollama; real-provider code exists, is unit-tested against mocked
  responses, and has never made a live billed call here. A bounded, cost-
  capped live comparison against the local-model baseline above is
  planned but not yet done.
- **A dashboard control surface is in progress**, not fully closed out:
  background execution on task creation and a redesigned task-detail
  view exist, but the original milestone's visual-polish and live-status
  UI work is not yet complete or independently verified end to end.

## Setup & Running

Requires Python 3.11+, a running Postgres instance with the `pgvector`
extension available, Docker (for sandboxed tool execution), and Ollama
running locally with `qwen2.5-coder:14b` pulled (or another model passed
via `--model`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Set `DATABASE_URL` (e.g. in a `.env` file at the repo root):

```bash
DATABASE_URL=postgresql+asyncpg://localhost/amop_dev
```

There's no separate migration step — every CLI command initializes the
schema (including the `vector` extension) on startup if it isn't already
there. Run the test suite:

```bash
python3 -m pytest -q
```

Core CLI commands (`amop <command>`, once the package is installed):

```bash
amop fix --repo <path> --description "<bug description>"   # run a bug-fix chain once
amop resume <task-id>                                        # resume a crashed/interrupted task
amop watch --repo <owner/name>                                # poll a repo for new issues
amop status <task-id>                                         # show a task's current state
amop optimize --repo <path> --entry-point "<module:function>" # run the Optimizer
amop update-deps --repo <path> --manifest <path>              # run the DependencyUpdater
amop report --since <date>                                    # ground-truth activity report
amop repos add <path>                                         # register a repo, trigger indexing
amop repos list                                                # list registered repos
amop evaluate --repeat <n>                                     # run the benchmark suite
amop audit verify                                              # verify the hash-chained audit trail
amop serve-api                                                 # run the REST API
amop serve-telegram                                            # run the Telegram bot
amop cleanup containers --confirm                              # reap orphaned sandbox containers
amop cleanup scratch-dirs --confirm                            # remove orphaned scratch directories
amop index prune-stale --confirm                               # prune code_chunks for deregistered repos
```

Run `amop --help` or `amop <command> --help` for the full option list —
most commands take `--model` to select a provider/model and `--mode` to
select a Safety Engine permission mode (`observer`, `supervised`,
`autonomous`, etc.).

## Design Philosophy

**Ground-truth verification over model self-report.** Nothing in this
project trusts an agent's own account of what it did. A "tests pass"
claim is checked by re-running the real suite in a fresh sandbox; a
Reviewer's rejection is checked by executing its claimed counterexample
against the real code, not by inspecting whether the rejection looks
well-formed; a "resolved N tasks" report is a database count, not a
number the model was asked to remember. This discipline exists because
the alternative failed concretely and repeatedly during this project's
own history — a mechanical check that verified *shape* instead of
*truth* was proven gameable by a fabricated-but-well-formed counterexample
before the sandbox-execution approach replaced it.

**Mechanical enforcement over prompt instructions.** A safety rule that
lives only in a system prompt is a suggestion, not a guarantee — this
project treats "the agent is told not to" as equivalent to "the agent can
do it." Every rule that actually matters (protected paths, permission
modes, diff-size caps, the forced-fresh-read gate before a patch is
accepted) is a code path an LLM's output cannot argue its way past,
proven by testing what happens when the model is deliberately pointed at
the forbidden action.

**Deliberate avoidance of infrastructure the system doesn't concretely
need yet.** Redis/Streams, a distributed job queue, a microservices
split, formal database migrations — none of these are here, and each
absence was a considered decision revisited multiple times as the system
grew, not an oversight waiting to be "fixed." A single-process orchestrator
talking to one Postgres database has, so far, been sufficient for every
concurrency, idempotency, and event-ordering guarantee this project has
actually needed; the tradeoffs of skipping each piece of infrastructure
are recorded honestly rather than assumed away.
