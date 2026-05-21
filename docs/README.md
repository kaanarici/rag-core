# Documentation map

**Agents and maintainers:** read this file first, then [../roadmap.md](../roadmap.md) for checklist state.

## Maintainer loop

```bash
./scripts/dx_smoke.sh
uv run rag-core doctor --json
```

Script catalog: [../scripts/README.md](../scripts/README.md).

## Start here (human + agent)

| Need | Doc |
|------|-----|
| First 10 minutes, no keys | [quickstart.md](quickstart.md) |
| Hit JSON, context packs, traces | [expectations.md](expectations.md) |
| Embed `RAGCore` in an app | [embedding/production-guide.md](embedding/production-guide.md) |
| Connector / worker pattern | [embedding/connector-pattern.md](embedding/connector-pattern.md) |
| Self-host HTTP API | [self-host/quickstart.md](self-host/quickstart.md) → [openapi.yaml](self-host/openapi.yaml), [auth.md](self-host/auth.md), [config.md](self-host/config.md) |
| Vector stores (Qdrant, TurboPuffer) | [providers/vector-stores.md](providers/vector-stores.md) |
| Custom providers | [providers/custom-providers.md](providers/custom-providers.md) |
| Provider wire shapes | [providers/provider-output-shapes.md](providers/provider-output-shapes.md) |
| Evals (library only) | [evals/retrieval-quality.md](evals/retrieval-quality.md) |
| Parsing formats | [parsing/formats.md](parsing/formats.md) |
| Vercel AI SDK tools | [integrations/vercel-ai-sdk-tools.md](integrations/vercel-ai-sdk-tools.md) |

## Product planning (substantial work)

| Doc | Use |
|-----|-----|
| [plans/one-repo-retrieval-engine-strategy.md](plans/one-repo-retrieval-engine-strategy.md) | **Active** — journeys A/B/C/Q/R/V, shape gate, out of scope |
| [plans/archive/](plans/archive/) | **Historical** — do not execute |

## Research (comparison only, not implementation orders)

| Doc | Topic |
|-----|--------|
| [research/managed-rag-landscape.md](research/managed-rag-landscape.md) | Ragie-class APIs |
| [research/oss-rag-landscape.md](research/oss-rag-landscape.md) | RAGFlow, Haystack, LlamaIndex |
| [research/turbopuffer-landscape.md](research/turbopuffer-landscape.md) | TurboPuffer optional path |
| [research/convex-landscape.md](research/convex-landscape.md) | Convex RAG — **no integration planned** |
| [research/retrieval-benchmark-corpus.md](research/retrieval-benchmark-corpus.md) | Future Q2a corpus |

## ADRs (architecture truth)

| ADR | Decision |
|-----|----------|
| [adr/0001-vendor-neutral-vector-store.md](adr/0001-vendor-neutral-vector-store.md) | `VectorStore` protocol; Qdrant default, TurboPuffer optional |
| [adr/0002-linear-pipeline-no-dsl.md](adr/0002-linear-pipeline-no-dsl.md) | Linear retrieve → fuse → rerank pipeline |
| [adr/0003-no-platform-provider-model-lock-in.md](adr/0003-no-platform-provider-model-lock-in.md) | Provider injection, no platform lock-in |
| [adr/0004-library-core-with-optional-self-hostable-runtime.md](adr/0004-library-core-with-optional-self-hostable-runtime.md) | Library first; `[runtime]` extra for `serve` |

## Code map (`src/rag_core/`)

| Area | Path | Responsibility |
|------|------|----------------|
| Facade | `core.py`, `core_*.py` | `RAGCore` — ingest, search, context, config assembly |
| CLI | `cli.py`, `cli_*.py` | `rag-core` commands; config flags in `cli_config_parser.py` |
| Quickstart | `quickstart.py` | Wheel-only demo module |
| Local proof | `local_ingest.py` | Folder ingest for `local-search` |
| Runtime HTTP | `runtime/` | Optional `serve` (Starlette), errors, health |
| Search | `search/` | Indexer, searcher, `QueryPlan`, pipeline stages |
| Vector stores | `search/providers/` | Qdrant, TurboPuffer, memory, embeddings, rerankers |
| Documents | `documents/` | Parsers, converters, PDF/OCR routing |
| Events | `events/` | JSONL traces, summaries, retrieval hit export |
| Config | `config/` | Typed `RAGCoreConfig` pieces |
| Integrations | `integrations/` | Optional framework adapters |

Tests mirror contracts: `tests/test_*_contract.py`, `tests/test_runtime_http.py`, `tests/test_turbopuffer_*.py`.

## Naming

Contributor naming principles: [naming.md](naming.md). Local display rebrand only: [../dev/REBRAND.md](../dev/REBRAND.md).

## Doc hygiene rules

1. **Claims must match code and tests** — if a doc says a command exists, grep `src/` and `tests/`.
2. **One active strategy** — `plans/one-repo-retrieval-engine-strategy.md`; never revive archive plans without explicit user request.
3. **No duplicate smoke docs** — quickstart describes steps; `scripts/README.md` owns automation.
4. **v0 pre-release** — avoid “v1.1” labels; default wheel = Qdrant; TurboPuffer = `--extra turbopuffer`.
