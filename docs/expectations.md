# Retrieval expectations

`rag-core` is an embeddable retrieval engine, not a hosted RAG API. Applications keep auth, chat, and model calls; the library owns parse → chunk → index → search → context/citations/manifest/events.

## Chunk JSON (`SearchResult`)

| Ragie `scored_chunks` | `SearchResult` | Notes |
|----------------------|----------------|-------|
| `id` | `id` | Chunk point id |
| `text` | `text` | Model-facing chunk body |
| `score` | `score` | Rank score from the active fusion/rerank stage |
| `document_id` | `document_id` | Stable document identity |
| `document_key` | `document_key` | App-facing locator (path, URL key, etc.) |
| `metadata` | `metadata` | Filterable payload fields |
| — | `corpus_id`, `namespace` | Tenant scoping on every hit |
| — | `section_path`, `chunk_index`, `title` | Citation locators |

CLI `search --json` and `to_retrieval_hits()` in `rag_core.events.export` emit the same logical fields for agents and observability adapters (LangSmith, OpenInference, OTel `gen_ai.retrieval.documents`).

## Context pack (`ModelContextPack`)

Use `retrieve_context(...)` or `rag-core retrieve-context` when the model needs trimmed text, citations, source previews, and token estimates. Raw `search` hits stay separate from the context pack by design.

## Manifest vs vendor doc registry

Manifest JSONL records ingest fingerprints and skip-by-hash reconciliation. It is not a hosted document registry; apps own external doc IDs and map them to `document_id` / `document_key`.

## Non-goals

- Hosted chat, webhooks, billing, teams, admin UI
- `rag-core eval` / `trace-summary` CLI (use `rag_core.evals`, `examples/retrieval_eval.py`, and `summarize_search_trace` on events JSONL)
- TurboPuffer in the default wheel (use `--extra turbopuffer`)
- Eval HTTP on `rag-core serve`

## Self-host runtime (optional)

`pip install 'rag-core[runtime]'` exposes `rag-core serve` with health, runtime description, ingest jobs, search, and retrieve-context only.
