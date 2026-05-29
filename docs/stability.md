# Public surface stability (v0 beta)

`0.1.x` is **beta**. Breaking changes are allowed with reason, tests, and doc updates.
This page lists what we treat as the **intentional public contract** today.

## Public beta surface

| Surface | Notes |
| --- | --- |
| `RAGCore` | Parse/prepare bytes or files; ingest bytes, one file, local file/directory paths via `ingest_files`, ZIP archives, URLs, or URL lists; delete documents; search; retrieve context; close |
| `RAGCoreConfig` + `rag_core.config` dataclasses/constants | CLI and library config shapes |
| Root `rag_core.__all__` models | First-order data returned by the facade: parsed/prepared/ingested/deleted documents, manifest entries, OCR metadata, `SearchResult`, `ContextPack`, and citation types |
| Curated `rag_core.search` public surface | Query plans, search profiles, profile/catalog description helpers, filters, result/context data types |
| `rag_core.contracts` | Tool request parsing and bound retrieval-scope helpers for app-owned HTTP/tool endpoints |

`ContextPack` intentionally has two projections: `to_payload()` / `as_text()` are
app-facing and may include stable source identity for traces, UI, or debugging;
`to_prompt_payload()` / `as_prompt_text()` are prompt-safe and use rank-local
citations for model/tool responses.

## Experimental (may change)

| Surface | Notes |
| --- | --- |
| `rag-core serve` | Thin HTTP wrapper; ingest jobs use server-local `path` |
| Remote / manifest CLI | Maintainer ops; not the core embed surface |
| `RAGCore` manifest helpers | Available on the facade, but not the main embed/search contract |
| `RAGCore` health/runtime introspection | Useful for diagnostics and `serve`, but may change while runtime hardens |
| `rag_core.sources` | Local, archive, remote, and manifest reconciliation primitives for app-owned sync jobs |
| `rag_core.integrations` | LangChain and OpenAI Agents helper builders |
| `rag_core.evals` | `load_cases`, `run_eval`, metrics - run in your repo |
| `rag_core.events` | Event types, event sinks, JSONL export, trace summaries |
| TurboPuffer adapter | Optional extra; query-plan support evolves |
| Provider registry entries/names | New providers and extras may ship without minor bump |
| Curated `rag_core.search.providers` provider surface | Provider factories, registries, caches, and the default Qdrant vector-store adapter; optional/utility vector stores stay in their owning modules or config/registry selection |
| `rag_core.documents.converters` | Converter base/result types and `get_converter` / `convert_file` lookup |
| `rag_core.documents` extension surface | OCR providers, chunk contextualizers, chunking registry, and local parse helper |
| `rag_core.search.pipeline_runner` | Search pipeline internals, including `SearchRequest` and `SearchExecutionOptions` |
| Low-level `rag_core.search.provider_protocols` protocols | Provider-author contracts; re-exported from `rag_core.search.types` for compatibility; app code should prefer curated imports |

## Not public (do not import for stability)

- `rag_core.demo` — smoke helpers only
- `tests.support` — test package, not shipped in wheel
- `rag_core._engine` — private implementation package behind the facade
- Undocumented `cli_*` modules — internal command wiring

## Rerank and hybrid defaults

See [expectations.md](expectations.md). Library defaults use a capability-aware query
plan: `balanced` hybrid RRF when supported, a narrower dense or sparse plan when not.
Rerank remains off until you pass `rerank=True` with a configured reranker.
`use_lexical_search` is the request flag for configured lexical/exact-match
sidecar path, not a query-plan or search-profile selector.
