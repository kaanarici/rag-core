# TurboPuffer landscape

**Accessed:** 2026-05-20  
**Purpose:** Decide whether TurboPuffer should be a first-party managed vector path in rag-core **v0 pre-release** and what “deep” integration means.

Sources: [TurboPuffer query](https://turbopuffer.com/docs/query), [vector search](https://turbopuffer.com/docs/vector), [write](https://turbopuffer.com/docs/write), [pricing](https://turbopuffer.com/pricing), [roadmap](https://turbopuffer.com/docs/roadmap), in-repo [ADR-0001](../adr/0001-vendor-neutral-vector-store.md), [provider-output-shapes](../providers/provider-output-shapes.md).

## What TurboPuffer is

Serverless **vector + full-text + sparse** search over **namespaces** (isolated indexes), backed by object storage economics. One HTTP API covers:

| Capability | TurboPuffer | rag-core need |
|------------|-------------|---------------|
| Dense ANN | Yes (`vector`, ANN) | **Required** — baseline |
| Exact kNN on filtered subset | Yes | Nice for high-recall reruns |
| BM25 full-text | Yes (`text`, BM25) | **Required** for hybrid parity |
| SparseKNN | Yes (2026 GA) | **Required** if SPLADE/BM25 sparse path matters |
| Multi-query / hybrid | Yes (client-side fusion encouraged) | **Required** for `QueryPlan` hybrid |
| Filters on attributes | Yes (SQL-like) | **Required** — namespace/corpus/document scoping |
| Aggregations | Yes | Useful for health / document counts |
| Writes / patches / delete-by-filter | Yes | **Required** for ingest lifecycle |

Performance (vendor claims, 1M docs): warm query p50 ~8ms; cold much higher. Hybrid often implies **multiple queries per user query** — same pattern rag-core already uses with prefetch + fusion on Qdrant.

## Why it is useful for rag-core

| Audience | Benefit |
|----------|---------|
| Teams leaving **managed RAG** who still want **managed vectors** | Run retrieval plane in-process; offload index ops to TurboPuffer instead of operating Qdrant |
| High-scale production | Object-storage pricing model vs self-hosted Qdrant cluster ops |
| Hybrid retrieval | Native BM25 + dense + sparse in one vendor; aligns with rag-core `QueryPlan` direction |

## Why it is not a Journey A default

- Requires **API key** and cloud namespace — breaks no-key first-run promise.
- Cannot replace Qdrant for **local/self-host compose** golden path (`docker compose` + embedded Qdrant).

**Shape lock:** Qdrant remains default; TurboPuffer is **first-party optional** (ADR-0001).

## Fit vs Qdrant in this repo

| Dimension | Qdrant (default) | TurboPuffer (v0 optional) |
|-----------|---------------------|---------------------------|
| Local dev | Excellent (`:memory:`, Docker) | Cloud only |
| Self-host journey C | Yes | Partial (API only; no compose story) |
| Hybrid in `QueryPlan` | RRF, DBSF, weighted RRF, MMR, boost | Multi-query + BM25 + SparseKNN; fusion often client-side |
| Operational cost | You operate the cluster | Pay per query/GB; no minimum in public pricing narrative |
| Adapter history in repo | Full | Optional extra in v0; contract audit in provider-output-shapes |

Provider-output audit (2026-05-20): prior adapter supported **dense ANN + filters + deletes + health** and **failed closed** on sparse, hybrid, MMR, boost. That is the correct starting point — not fake parity.

## Steal / mirror / refuse

| Item | Decision |
|------|----------|
| First-party managed adapter behind `VectorStore` | **Mirror** ADR-0001 |
| Documented query-plan limits per backend | **Steal** — honesty beats feature matrix lies |
| TurboPuffer as only vector store | **Refuse** |
| TurboPuffer-specific shortcuts in core `QueryPlan` DSL | **Refuse** — translate from shared `QueryPlan` |
| Re-implementing TurboPuffer fusion in core when vendor expects client RRF | **Steal** — same as today’s orchestrator fusion |
| Journey A no-key smoke on TurboPuffer | **Refuse** |

## “Deep integration” definition (v0 slices)

Deep does **not** mean “mention TurboPuffer in README.” It means:

1. **Registry + config** — `VectorStoreConfig` accepts `turbopuffer`; CLI/env parity with Qdrant flags where sensible.
2. **Contract tests** — `test_vector_store_contract.py` passes for TurboPuffer alongside Qdrant and memory.
3. **Doctor diagnostics** — namespace, dimensions, declared `StoreCapabilities`, secret-safe errors.
4. **Query-plan stages** — explicit matrix in docs; fail closed with `UnsupportedQueryStage` when not implemented.
5. **Hybrid path** — at minimum BM25 + dense ANN multi-query fused to match `balanced` / `lexical` profiles (live-backed tests behind marker).
6. **Optional extra** — `uv sync --extra turbopuffer` so default wheel stays lean.

### Phased shapes (pick one slice at a time)

| Phase | Shape | User promise | Out of scope |
|-------|-------|--------------|--------------|
| **TP1** | Restore base adapter | I can point rag-core at TurboPuffer for ANN ingest/search with filters | BM25, sparse, MMR |
| **TP2** | Hybrid query-plan | My `search_profile=balanced` works on TurboPuffer for dense+BM25 | Boost, MMR, every Qdrant-only stage |
| **TP3** | SparseKNN + parity matrix | SPLADE/BM25 sparse channels map correctly | TurboPuffer-only ranking hacks |

**Do not** start TP2 before TP1 contract tests are green.

## Risks

| Risk | Mitigation |
|------|------------|
| API drift | Contract tests + pinned client version |
| Cost surprises | Document pricing link; no hiding query fan-out for hybrid |
| Capability lies | `StoreCapabilities` + doctor + provider-output-shapes updates |
| v1 wheel bloat | Optional extra only |
| Diverting from Journey Q | Finish **Q2a** corpus before TP2 unless user explicitly prioritizes TurboPuffer |

## Recommendation

**Yes — TurboPuffer is useful** as the **managed vector** leg of “one retrieval plane, two first-party stores.”

**Default execution order after Phase F:**

1. **Q2a** — public benchmark corpus (belief)
2. **TP1** — restore adapter + contract + doctor (managed vector credibility)
3. **B2.5** — embed production proof
4. **TP2** — hybrid query-plan depth

Parent: [one-repo-retrieval-engine-strategy.md](../plans/one-repo-retrieval-engine-strategy.md).
