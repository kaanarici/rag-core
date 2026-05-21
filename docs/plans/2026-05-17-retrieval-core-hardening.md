# RAGCore Retrieval Core Hardening

Date: 2026-05-17
Status: Active current-state plan
Last refreshed: 2026-05-18
Project maturity: pre-prod

## Goal

Keep `rag-core` on the path to being a first-party retrieval engine for embedded and self-hosted RAG systems without turning it into a hosted platform or reopening already-completed cleanup.

This plan is intentionally current-state only. It keeps the durable architecture, product-shape, open questions, and validation rules that should guide the next goal. It is not an execution ledger.

## Source Boundary

Planning in this repo stays grounded in repo-local docs, code, tests, ADRs, public OSS comparisons, and public benchmarks.

## Verified Current State

Current repo direction:

- `MISSION.md` defines `rag-core` as a first-party retrieval engine for embedded and self-hosted RAG systems.
- Repo policy keeps the library core-sized, typed, and embeddable.
- The optional runtime boundary is documented in ADR-0004; the core library remains the source of truth.
- Qdrant is the accessible default vector store.
- v1 ships Qdrant only in the wheel; TurboPuffer returns as a first-party optional adapter in v1.1 (ADR-0001 addendum).

Current strong surfaces:

- `RAGCore` is the main facade for parse, prepare, ingest, search, retrieve-context, manifest, delete, and runtime-description flows.
- The CLI already covers first-run local search, persistent ingest/search loops, diagnostics, traces, evals, and explicit remote/archive flows.
- The retrieval stack already exposes typed search profiles, explicit query-plan control, rerank budgets, context packs, citations, traces, and eval reporting.
- Mixed-source ingest, source fingerprints, manifest reconciliation, and bounded batch execution are already part of the engine shape.
- First-party provider/runtime diagnostics are already inspectable through `doctor` and runtime description paths.
- The codebase has already been split back toward concept-sized modules; future work should preserve that shape instead of redoing it.

Current constraints:

- Library import and normal embedded use must stay independent of any server/runtime package.
- "Vendor-flexible" does not mean all providers are equally supported.
- First-party support means docs, diagnostics, tests, and explicit behavior limits.
- Advanced retrieval controls should compose through typed contracts, not a graph DSL or hidden orchestration layer.
- New work should strengthen the developer mental model: sources, corpora, collections, indexes, search profiles, traces, evals, and citations.

Already-landed baseline that should not be reopened without a concrete regression:

- Mission and ADR alignment for the optional runtime boundary.
- No-key local folder retrieval and the inspectable doctor/search/trace/eval loop.
- Query-plan capability honesty and provider preflight behavior.
- Mixed-source ingest, manifest durability, and source-fingerprint consistency.
- Trace and eval reporting surfaces for retrieval inspection.
- Broad structure work that reduced pressure on `core.py`, provider orchestration files, and other hotspots.

## Product-Shape Graph

```text
rag-core as an OSS retrieval engine
├── first 10 minutes
│   ├── local-search on a real folder with no key
│   ├── doctor for runtime and provider shape
│   ├── events JSONL + summarize_search_trace for inspectable behavior
│   └── library evals + examples/retrieval_eval for retrieval-quality proof
├── embedded app path
│   ├── RAGCore as the stable facade
│   ├── application-owned async lifecycle and shutdown
│   ├── file, URL, and archive ingest through shared core contracts
│   ├── retrieve_context with citations and source previews
│   └── runtime/provider diagnostics without shell-only dependence
├── CLI and library parity
│   ├── commands map to the same concepts as the library
│   ├── JSON output stays machine-readable
│   └── shared diagnostics and retrieval catalogs do not drift by surface
├── retrieval quality
│   ├── search profiles for recognizable defaults
│   ├── QueryPlan for explicit advanced control
│   ├── reranker budgets and fallback policy
│   ├── traceable search/eval behavior
│   └── corpus-specific quality gates
├── source and durability model
│   ├── append-friendly manifests with reconciliation and compaction
│   ├── per-source fingerprints for file, URL, and archive ingest
│   ├── bounded concurrency and partial-failure reporting
│   └── index/cache guards before backend writes
├── document understanding
│   ├── format support matrix as the support contract
│   ├── parser quality metadata and locators
│   ├── optional PDF/OCR understanding
│   └── citation-preserving context packs
├── first-party provider policy
│   ├── Qdrant as the default self-hosted path
│   ├── TurboPuffer deferred to v1.1 (managed first-party path)
│   ├── declared capability limits fail closed
│   └── first-party support requires docs, diagnostics, and tests
└── codebase trust
    ├── public names stay familiar and deliberate
    ├── files stay concept-sized
    ├── aggregator files stay single-owner surfaces
    └── new work should simplify or split, not regrow orchestration hotspots
```

## Remaining Hardening Lanes

These are the forward-looking lanes that still justify plan-level attention.

### Lane 1: First-Run and Adoption Quality

Keep the first successful path short, accurate, and inspectable.

Priority:

- README and examples should continue to reflect the actual first-run loop.
- Doctor, local-search, search, retrieve-context, events JSONL, and library evals should remain the canonical inspect loop.
- Packaging and release metadata should stay aligned with the public engine shape.

Do not reopen:

- Old "prove local folder retrieval exists" work. That baseline is already landed.

### Lane 2: First-Party Provider Maturity

Keep provider breadth honest and support-level-driven.

Priority:

- Deepen first-party providers only when docs, diagnostics, tests, and explicit limitations move together.
- Keep capability declarations and failure behavior aligned.
- Avoid provider features that leak into the base protocol before multiple first-party implementations justify them.

Do not reopen:

- Generic "adapter exists therefore it is supported" framing.

### Lane 3: Retrieval Quality and Evaluation

Keep retrieval improvements measurable instead of anecdotal.

Priority:

- Preserve search-profile clarity and explicit query-plan control.
- Expand eval usefulness only when the metric/reporting surface stays legible.
- Keep trace artifacts and eval reports sufficient for regression triage without exposing sensitive content by default.

Do not reopen:

- Older immediate recommendations to add trace/eval surfaces; that groundwork is already complete.

### Lane 4: Source Ingest and Durability

Keep file, URL, and archive ingest aligned under one durable source model.

Priority:

- Preserve manifest identity, reconciliation semantics, and source-fingerprint correctness.
- Add new source/fetching behavior only behind explicit safety and validation boundaries.
- Keep batch behavior resumable and inspectable without introducing platform machinery prematurely.

Do not reopen:

- Broad "add manifest durability" or "support mixed-source ingest" goals as if they are still missing.

### Lane 5: Optional Self-Hostable Runtime

Treat runtime work as a dependent layer over a stable engine, not as the center of gravity.

Priority:

- Define the smallest useful runtime contract before adding operational surface area.
- Reuse core contracts directly; do not fork retrieval behavior into server-only code paths.
- Keep auth, tenancy, jobs, and deployment assumptions minimal and explicit.

Entry condition:

- Runtime work should only advance when the specific core contract it exposes is already crisp, tested, and inspectable from the library side.

## Validation and QA

Baseline validation for every meaningful implementation slice:

```bash
uv run ruff check .
uv run mypy src tests examples
uv run pytest -q
```

Additional validation by slice:

- Package or public-surface changes: `uv build` plus wheel smoke.
- Eval changes: `uv run pytest -m eval -q`.
- Provider changes: provider contract tests plus env-gated integration smoke.
- CLI changes: command smoke with JSON output checks.
- Runtime changes: local API smoke and dependency-isolation checks.
- Parser changes: fixture-based parse and chunk snapshots.
- Fetching changes: SSRF, redirect, content-type, timeout, size-limit, and path-safety tests.
- Search changes: trace snapshot, latency budget check, and retrieval eval fixture.

Quality gates:

- No new broad optional dependency imports at `import rag_core`.
- No advertised first-party provider without docs and tests.
- No runtime path that bypasses core library behavior.
- No trace artifact that leaks secrets by default.
- No archive or fetch path without security tests.
- Shared aggregator files should have one integration owner per slice.

## Flow Impact

Likely flow surfaces for future retrieval-core work:

- CLI entrypoint and parser families: `src/rag_core/cli*.py`
- Core facade and assembly: `src/rag_core/core.py`, `src/rag_core/facade/*`, `src/rag_core/core_assembly.py`
- Ingest and manifest durability: `src/rag_core/core_ingest.py`, `src/rag_core/manifest_persistence.py`, local/remote/archive ingest support modules
- Document preparation: `src/rag_core/documents/*`
- Search and planning: `src/rag_core/search/*`
- Providers: `src/rag_core/search/providers/*`
- Events and traces: `src/rag_core/events/*`
- Evals: `src/rag_core/evals/*`
- Optional runtime, if added later: isolated runtime package or module path

Flow files should be updated only when implementation changes public entrypoints, persistence semantics, provider contracts, runtime jobs, or user-visible data flow.

## Rollout and Rollback

Rollout rules:

- Ship provider changes behind explicit config when default behavior could change.
- Ship fetch/scrape behavior as explicit opt-in, never as implicit local ingest behavior.
- Ship runtime work as optional extras or isolated packaging so library-only users are unaffected.
- Ship parser/document-understanding work format by format with independent tests.

Rollback rules:

- If a first-party provider is unstable, reduce its support claim before removing core contracts.
- If a parser regresses, prefer a documented safe fallback over silent behavior drift.
- If runtime packaging destabilizes the library, isolate or remove runtime packaging without touching core behavior.
- If an eval floor is wrong, correct the fixture and document the reason instead of quietly lowering gates.

## Data, Auth, and Security Notes

Local files:

- Prevent path traversal in folder and archive ingest.
- Ignore known temporary, cache, and lock files by default when appropriate.
- Redact secrets in traces.
- Do not store raw file content outside configured stores.

Network fetching:

- Block loopback, private, link-local, and metadata IP ranges by default.
- Validate redirects.
- Enforce max-size and timeout policies.
- Keep content-type allowlists.
- Preserve redacted source identity in diagnostics.

Archives:

- Limit total uncompressed size, file count, path length, and supported-format scope.
- Reject absolute paths and parent traversal.
- Skip unsupported binaries unless an explicit converter owns them.

Runtime:

- Keep authentication pluggable first.
- Document reverse-proxy or token-gate deployment patterns before inventing product concepts.
- Keep tenancy and authorization as explicit API questions, not hidden globals.

Traces and evals:

- Default to metadata, counters, and timings rather than full private content.
- Make full-content trace behavior explicit.
- Keep committed eval fixtures safe to publish.

## Lane Ownership

One lane, one owner, one acceptance gate.

Aggregator files remain single-owner surfaces:

- `src/rag_core/__init__.py`
- `src/rag_core/search/__init__.py`
- `src/rag_core/search/providers/__init__.py`
- `src/rag_core/documents/converters/__init__.py`

No future lane should regrow `core.py` or provider orchestration hotspots when a focused module is the cleaner boundary.

## Open Questions

These remain plan-level ambiguities rather than missing cleanup:

- Which embedding providers deserve first-party status after the current baseline?
- Which reranker providers deserve first-party support first?
- What is the smallest useful self-hostable runtime contract?
- Should the runtime live in the same package extras or in a sibling package?
- What is the minimum auth story for a self-hosted runtime without creating a hosted-product surface?
- Which scraping path, if any, deserves first-party support before broader runtime work?
- What public benchmark corpus should become the stable quality bar?
- Which current beta names should be treated as intentionally stable before a broader external release?

## Execution Readiness

Execution remains ready slice-by-slice.

Required before any broad new lane:

- Define one owner and one write scope.
- State the exact contract being hardened.
- Choose the matching validation path from this document.
- Confirm the slice is advancing a still-open lane instead of reopening completed baseline work.
