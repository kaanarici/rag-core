# Naming conventions

Stable vocabulary for humans and agents working in `rag-core`. Renames follow a **phased backlog** — one concept per slice, with tests and public exports updated together.

## Principles

1. **One concept, one term at each boundary.** Public APIs, CLI flags, events, and HTTP JSON should use the same word (e.g. `lexical_search`, not `sidecar` in one layer and `lexical_search` in another unless the mapping is explicit and documented).

2. **Reserve `core` for `RAGCore`.** Package modules use domain prefixes (`ingest_`, `manifest_`, `search_`, `document_`). Avoid `_for_core`, `_with_core`, and `_from_facade` in function names — the import path already shows layering.

3. **Verb prefixes by intent:**
   - `build_*` — assemble dataclasses/requests (no I/O)
   - `create_*` — construct wired providers/clients
   - `resolve_*` — derive IDs or config from inputs
   - `read_*` / `write_*` — local filesystem or store I/O
   - `fetch_*` — HTTP remote retrieval only

4. **Protocols and callbacks name the role**, not the payload: `PrepareDocumentCallback`, not `PrepareBytes`; `IngestDocumentCallback`, not `IngestBytes`.

5. **Files match the primary export or CLI subcommand.** Example: `cli_search.py` for `search` / `retrieve-context`; not `cli_query.py` for search handlers.

6. **Store-agnostic names above the adapter layer.** Types that take `VectorStore` should not be named `Qdrant*` unless they are Qdrant-specific adapters.

7. **Distinguish query types by layer** in the name when both appear in one flow, e.g. vector-store query vs orchestrator retrieval request vs mutable pipeline state — not two public types both called “search request” without a qualifier.

## What already works well

- Past-tense event nouns (`IngestStarted`, `SearchCompleted`)
- CLI triad: `cli_*_parser.py`, `cli_*.py`, `cli_*_output.py`
- `create_*` provider factories in `search/providers/`
- Frozen dataclass contracts for cross-boundary data
- Lazy `__getattr__` exports on package `__init__.py` surfaces

## Phased rename backlog

Do **not** rename everything in one PR. Order by blast radius and journey gates.

| Priority | Current | Target | Why |
|----------|---------|--------|-----|
| P1 | ~~`use_sidecar` (internal)~~ | `use_lexical_search` (+ legacy JSON key) | Done — one term; traces accept old `use_sidecar` |
| P1 | ~~`QdrantIndexer`~~ | `DocumentIndexer` (alias `QdrantIndexer` retained) | Done — store-agnostic indexer name |
| P2 | ~~`cli_query` / `run_query_command`~~ | `cli_search` / `run_search_command` | Done |
| P2 | `core_prepare` module | `document_prepare` | Exports are `parse_document_bytes`, `prepare_document_bytes` |
| P2 | `parse_bytes` (facade) vs `parse_document_bytes` (impl) | Align on `*_document_*` | Public/private mismatch |
| P3 | `SearchQuery` vs `SearchRequest` | Layer-qualified names | Two “search request” types |
| P3 | `PipelineQuery` | `PipelineState` or `RetrievalPipelineContext` | Mutable stage bundle, not a query |
| P3 | `ModelContextPack` | `ContextPack` or `RetrievalContextPack` | “Model” is ambiguous |
| P3 | `manifest_entry_for_core` | Inline `build_manifest_entry` | No-value alias |
| P3 | `PrepareBytes` / `IngestBytes` (duplicate protocols) | Shared callback types | Duplicated protocol definitions |
| P4 | ~~`local_corpus.py`~~ | `local_ingest.py` | Done |
| P4 | `events/sink.py` vs `events/sinks.py` | `event_sink_protocol.py` / `event_sinks.py` | Easy to import wrong module |
| P4 | `sources.py` barrel | `source_readers` package or explicit module | Filename looks like a domain type |

## Out of scope for rename-only work

- Renames that do not serve a journey acceptance gate (A/B/C/Q)
- Mass renames without deprecation aliases on public `rag_core` exports
- Renaming for style only when the name is accurate and stable in `docs/expectations.md`

When executing a backlog item, update tests, OpenAPI/CLI docs, and any ADR that references the old term in the same slice.
