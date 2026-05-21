# API stability (v0 beta)

`0.1.x` is **beta**. Breaking changes are allowed with reason, tests, and doc updates.
This page lists what we treat as the **intentional public contract** today.

## Beta-stable (embed here)

| Surface | Notes |
| --- | --- |
| `RAGCore` | `ingest_bytes`, `search`, `retrieve_context`, `ensure_ready`, `close` |
| `RAGCoreConfig` + `rag_core.config.*` | CLI and library config shapes |
| `SearchResult`, `ModelContextPack`, citation types | Root `rag_core` exports |
| `rag_core.evals` | `load_cases`, `run_eval`, metrics — run in **your** repo |
| `rag_core.contracts` | Tool request parsing for app-owned HTTP endpoints |
| `rag_core.events` | Event types, JSONL sinks, trace summaries |
| `rag_core.search` | `QueryPlan`, `search_profile`, filter types |

## Experimental (may change)

| Surface | Notes |
| --- | --- |
| `rag-core serve` | Thin HTTP wrapper; ingest jobs use server-local `path` |
| Remote / manifest CLI | Maintainer ops; not the core embed API |
| TurboPuffer adapter | Optional extra; query-plan support evolves |
| Provider registry names | New providers and extras may ship without minor bump |

## Not public (do not import for stability)

- `rag_core.demo` — smoke helpers only
- `tests.support` — test package, not shipped in wheel
- Undocumented `cli_*` / `core_*` modules — internal

## Rerank and hybrid defaults

See [expectations.md](expectations.md). Library defaults: hybrid on, rerank off until
you pass `rerank=True` with a configured reranker.
