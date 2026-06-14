# Design principles

Compressed from former ADRs. For code archaeology, see git history under `docs/adr/`.

## Vector stores

- `VectorStore` protocol stays; **Qdrant** is the default wheel path; **TurboPuffer** is first-party optional (`--extra turbopuffer`).
- Adapters declare `StoreCapabilities` and fail closed on unsupported query-plan stages.
- Honest support matrix beats fake backend parity.

## Retrieval pipeline

- **Linear staged pipeline** (dense, sparse, fusion, rerank, postprocess): no graph DSL as the core model.
- Advanced behavior plugs in as stages/adapters, not orchestrator forks.

## Providers

- Apps inject embeddings, rerankers, OCR, vector stores via registries or constructor args.
- No platform lock-in on a single vendor’s models or hosted index.

## Runtime

- **Library-first:** `import rag_core` stays light; `[runtime]` extra for `rag-core serve`.
- Serve is a thin HTTP layer over the same `RAGCore` contracts, with no eval HTTP, auth, teams, or billing in core.
