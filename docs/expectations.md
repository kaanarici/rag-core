# Retrieval expectations

`rag-core` is an embeddable retrieval engine. Applications keep auth, chat, and model
calls; the library owns parse → chunk → index → search → context/citations/manifest/events.

## Retrieval defaults

| Surface | Default search behavior | Rerank |
| --- | --- | --- |
| `RAGCore.search` / `retrieve_context` | capability-aware default query plan; `balanced` when dense+sparse hybrid RRF is supported | off; pass `rerank=True` when a reranker is configured |
| CLI `search` / `retrieve-context` | same default unless `--search-profile` or `--query-plan-preset` is passed | `--rerank` |
| HTTP `POST /v1/search` | same default through the configured runtime providers | `"rerank": true` in JSON |
| Tool contract (`rag_core.contracts`) | same default; `use_lexical_search` only gates configured lexical/exact-match expansion | off unless request sets `rerank` |

Default limits are intentionally surface-specific:

| Surface | Default limit |
| --- | --- |
| `RAGCore.search`, CLI `search`, HTTP `/v1/search` | 10 hits |
| CLI `local-search` | 5 hits |
| `RAGCore.retrieve_context`, CLI `retrieve-context`, HTTP `/v1/retrieve-context` | 8 snippets |
| `search_user_documents` tool contract | 5 snippets |

Rerank runs only when requested **and** a reranker provider is configured; otherwise the
rerank stage is a no-op.

Search profiles are named query-plan presets. `balanced` is the default profile when
the active vector store supports dense+sparse hybrid RRF. `fast` is dense-only,
`lexical` is sparse-only, `coverage` uses hybrid DBSF, and `diverse` uses hybrid
retrieval plus MMR. Explicit query plans win over profile/default behavior.

Provider reranking is fail-open by default: `RerankBudget(fallback_on_error=True)`
returns the original search ordering if the reranker errors or times out, and emits
rerank trace metadata with the fallback reason. Pass `fallback_on_error=False` when a
rerank failure should fail the search request instead.

## Chunk JSON (`SearchResult`)

Familiar `scored_chunks`-style fields for tool and observability adapters:

| App field | `SearchResult` | Observability export | Notes |
| --- | --- | --- | --- |
| `id` | `id` | `id` | Chunk point id |
| `text` | `text` | `content` | Retrieved chunk body |
| `score` | `score` | `score` | Rank score from the active fusion/rerank stage |
| `document_id` | `document_id` | `document_id` | Stable document identity |
| `document_key` | `document_key` | `document_key` | App-facing locator (path, URL key, etc.) |
| `metadata` | `metadata` | `metadata` | Filterable payload fields |
| — | `corpus_id`, `namespace` | `corpus_id`, `namespace` | Tenant scoping on every hit |
| — | `section_path`, `chunk_index`, `title` | `section_path`, `chunk_index`, `title` | Citation locators |

Extraction quality metadata appears as flat `quality_*` hit metadata when available,
and `retrieve_context(...)` carries the same sanitized values under
`retrieval_metadata["quality"]`.

CLI `search --json` emits the app-facing `SearchResult` field names.
`to_retrieval_hits()` in `rag_core.events.export` maps the chunk body from
`SearchResult.text` to an observability-friendly `content` field for LangSmith,
OpenInference, and OTel `gen_ai.retrieval.documents` adapters.
CLI `search` and `retrieve-context` accept `--content-type`, `--document-id`, and
`--metadata-filter` as narrowing filters inside the selected namespace/corpus scope.

## Context pack (`ContextPack`)

Use `retrieve_context(...)` or `rag-core retrieve-context` when the model needs trimmed
text, citations, source previews, and token estimates. Raw `search` hits stay separate
from the context pack by design.

The canonical context artifact is `ContextPack`. It keeps:

- `ContextSnippet` items for ranked context text, token estimates, source references, and
  optional retrieval metadata
- `SourceReference` values for stable source identity
- `SourceLocator` values for page, slide, sheet, row, code line, figure, bbox,
  section, and chunk hints when the parser/indexed payload provides them
- `SourcePreview` values for app-facing citation chips, with prompt-safe preview and
  locator projections in `to_prompt_payload()`

Serializers are intentionally thin: `to_payload()` is app-facing. `to_prompt_payload()`
and `as_prompt_text()` use rank-local citation IDs such as `S1` and omit app-private
source identifiers such as document IDs, document keys, result IDs, corpus IDs, and
source hashes. CLI and HTTP `retrieve-context` use `as_prompt_text()` for
`context_text` while keeping the structured context pack app-facing.

## Manifest vs hosted doc registry

Manifest JSONL records ingest fingerprints and skip-by-hash reconciliation. It is not a
hosted document registry; apps own external doc IDs and map them to `document_id` /
`document_key`.

## Source reconciliation

v0 exposes source reconciliation primitives for app-owned sync jobs, not a connector
sync product.

What core provides:

- Stable source identity via `document_id`, `document_key`, and `content_sha256`
- Status-only `rag_core.sources` reconciliation primitives for unchanged,
  changed, missing, orphaned, and duplicate sources
- Local and URL ingest payloads that include manifest status when a manifest directory
  is configured
- Safe reingest, skip-by-hash, stale chunk deletion, and manifest repair around the
  active ingest operation
- A typed `DeleteDocumentResult` from `RAGCore.delete_document` that reports which
  delete surfaces completed successfully

What the embedding application still owns:

- Connector cursors, polling schedules, webhook handling, and retry queues
- Decisions for deleted, moved, renamed, or access-revoked external sources
- Tenant binding for `namespace` and corpus selection
- Background cleanup jobs that call core deletion primitives

## Non-goals

- Hosted chat, webhooks, billing, teams, admin UI
- Eval or trace-summary CLI commands (use `rag_core.evals`, `examples/retrieval_eval.py`, and `summarize_search_trace` on events JSONL)
- TurboPuffer in the default wheel (use `--extra turbopuffer`)
- Eval HTTP on `rag-core serve`

## Self-host runtime (optional)

Installing the `runtime` extra exposes `rag-core serve` with health, runtime
description, ingest jobs, search, and retrieve-context only. From a checkout,
use `uv sync --extra runtime`.
