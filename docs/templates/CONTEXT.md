# rag-core context

**Read order for agents:** this file → [plans/ROUTING.md](plans/ROUTING.md) → [README.md](README.md) (catalog) → [../roadmap.md](../roadmap.md) (checklist).

**Local only** — generated from [templates/CONTEXT.md](templates/CONTEXT.md). Do not commit.

## Product (one sentence)

**rag-core** is the embeddable **retrieval plane** (parse → chunk → index → hybrid search → rerank → context → traces). Your app owns auth, chat, and connectors. Optional `rag-core serve` is a thin HTTP layer over the same `RAGCore` — not a second product.

## Maturity

**v0 pre-release** (`0.1.x` on PyPI classifiers: Beta). Sole maintainers today; optimize for honest contracts and fast local verification, not platform breadth.

## Three surfaces

| Surface | Entry | Proof |
|---------|--------|-------|
| Library | `RAGCore`, `examples/` | `./scripts/dx_smoke.sh` |
| CLI | `rag-core` (`doctor`, `ingest`, `search`, `local-search`, …) | `dx_smoke.sh` |
| HTTP (optional) | `uv sync --extra runtime` → `rag-core serve` | `./scripts/self_host_smoke.sh` |

## Maintainer loop

```bash
uv sync --group dev
./scripts/dx_smoke.sh
uv run rag-core doctor --json
```

CI parity: [../scripts/README.md](../scripts/README.md).

## Journey status (2026-05)

| Journey | Status | Doc |
|---------|--------|-----|
| A — First 10 minutes | **Done** | [quickstart.md](quickstart.md) |
| B — Embed | Docs done | [embedding/production-guide.md](embedding/production-guide.md) |
| C — Self-host | **Done** (C3+C2) | [self-host/quickstart.md](self-host/quickstart.md) |
| V — TurboPuffer | **Done** (optional extra) | [providers/vector-stores.md](providers/vector-stores.md) |
| Q — Quality proof | **Open** (Q2a on roadmap) | [roadmap.md](../roadmap.md) |

Optional local research notes: `docs/research/` (gitignored).

## Code map (where to edit)

| Change | Look in |
|--------|---------|
| Public API / ingest / search orchestration | `src/rag_core/core.py`, `core_*.py` |
| CLI flags / config | `cli_config_parser.py`, `core_config_cli.py` |
| Retrieval pipeline / `QueryPlan` | `src/rag_core/search/pipeline/`, `searcher.py` |
| Vector store adapter | `src/rag_core/search/providers/` |
| HTTP runtime | `src/rag_core/runtime/` |
| Contracts / hits / context JSON | `docs/expectations.md`, `tests/test_*_contract.py` |

## Hard stops (always)

No connector marketplace, hosted accounts, admin UI, billing, graph DSL, or runtime paths that bypass `RAGCore`. No broad cleanup without a journey gate in [plans/ROUTING.md](plans/ROUTING.md).

## Scope boundaries

Library-first. Qdrant = default wheel. TurboPuffer = `uv sync --extra turbopuffer`. No first-party Convex adapter.
