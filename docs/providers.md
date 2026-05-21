# Providers

Vector stores, custom extension points, and provider wire-shape notes.

## Vector stores

`rag-core` keeps vector stores behind one contract. The **default wheel** ships **Qdrant** as the first-party vector store.

### Maturity

| Level | Meaning |
| --- | --- |
| first-party default | Default CLI/runtime path, contract tests, diagnostics |
| first-party optional | Managed adapter behind `--extra` |
| utility adapter | Tests/smoke only (e.g. in-memory) |

| Provider | Maturity | Entrypoint |
| --- | --- | --- |
| Qdrant | first-party default | `QdrantConfig`, `--qdrant-*`, `QdrantVectorStore` |
| TurboPuffer | first-party optional | `--vector-store turbopuffer`, `uv sync --extra turbopuffer` |
| In-memory | utility | `InMemoryVectorStore` |

`doctor --json` and `StoreCapabilities` must match real behavior. Contract tests: `tests/test_vector_store_contract.py`, provider-specific suites.

### Qdrant (default)

```bash
uv run rag-core doctor --check-store --qdrant-location :memory: --json
uv run rag-core ingest ./docs --namespace acme --corpus-id help \
  --qdrant-url http://localhost:6333 --embedding-model text-embedding-3-small --embedding-dimensions 1536
```

### TurboPuffer (optional)

Not in the default wheel. Requires `TURBOPUFFER_API_KEY`. Supports dense ANN, hybrid RRF (BM25 + dense when `lexical_query` is set), SparseKNN. Unsupported stages (DBSF, MMR, boost, …) fail closed.

Do not use TurboPuffer for no-key smoke or default `docker compose` (demo embeddings).

### Migration

Re-ingest with the same embedding model/dimensions, re-run `doctor --json` and `examples/retrieval_eval.py` before promoting a new backend.

---

## Custom providers

Register or inject application-owned embeddings, rerankers, OCR, vector stores, caches, and sidecars. A registry entry does not automatically make a category selectable from `RAGCoreConfig`.

| Category | Protocol | Registry | Runtime selection |
| --- | --- | --- | --- |
| Dense embeddings | `rag_core.search.types.EmbeddingProvider` | `EMBEDDING_PROVIDERS` | `RAGCoreConfig.embedding`, CLI, or `RAGCore(embedding_provider=...)` |
| Sparse embeddings | `rag_core.search.types.SparseEmbedder` | `SPARSE_EMBEDDERS` | `RAGCore(sparse_embedder=...)` |
| Rerankers | `rag_core.search.types.RerankerProvider` | `RERANKER_PROVIDERS` | `RAGCoreConfig.reranker`, CLI, or `RAGCore(reranker=...)` |
| Vector stores | `rag_core.search.types.VectorStore` | `VECTOR_STORES` | `RAGCoreConfig.vector_store`, CLI, or `RAGCore(vector_store=...)` |
| OCR | `rag_core.documents.OcrProvider` | `OCR_PROVIDERS` | `RAGCore(ocr_provider=...)`; no `RAGCoreConfig` field |
| Search sidecars | `rag_core.search.types.SearchSidecar` | `SEARCH_SIDECARS` | `RAGCoreConfig.ingest.lexical_search_provider` or `RAGCore(search_sidecar=...)` |
| Embedding cache | `rag_core.search.providers.EmbeddingCache` | `EMBEDDING_CACHES` | `RAGCoreConfig.ingest.embedding_cache_provider` or inject |
| Chunk context cache | `rag_core.search.providers.ChunkContextCache` | `CHUNK_CONTEXT_CACHES` | inject only |
| Chunking | `rag_core.documents.chunking.ChunkingStrategy` | `CHUNKING_STRATEGIES` | prepare paths |
| Converters | `rag_core.documents.converters.BaseConverter` | `get_converter()`, `convert_file()` | `ConversionResult` via converter registry |

Declare `StoreCapabilities` truthfully; unsupported query-plan stages must fail closed.

Install extras before optional first-party providers: `rerank`, `voyage`, `zeroentropy`, `turbopuffer`, `runtime`, `anthropic`, `opentelemetry`, `langchain`, `openai-agents`. `semantic` and `html` add local parsing helpers.

### Support levels and diagnostics

`rag-core doctor --json` reports non-secret readiness:

| Category | Diagnostic surface |
| --- | --- |
| Dense embeddings | provider, registered names, package availability, API-key presence, dimensions |
| Sparse embeddings | registered names, package availability |
| Rerankers | requested/effective provider, fallback reason |
| OCR | registered names, API-key presence |
| Contextualizers | package availability, injection surface |
| Caches | configured provider and registered names |
| Search sidecars | configured provider and registered names |
| Event sinks | package availability, injection surface |
| Vector stores | capabilities, query-plan support, collection/namespace identity |

---

## Provider output shapes

Checked against provider docs on 2026-05-20. Contract audit for first-party adapters — not a benchmark.

| Provider | Notes |
| --- | --- |
| OpenAI embeddings | `data[]` with `index` and `embedding`; dimension overrides for known models |
| Voyage embeddings | Flexible dimensions per model metadata |
| ZeroEntropy embeddings | `results[]` with `embedding`; `float` path supported |
| Cohere rerank | `results[]` with `index`, `relevance_score` |
| Voyage rerank | `index`, `relevance_score` |
| ZeroEntropy rerank | `index`, `relevance_score`; `top_n` |
| FastEmbed sparse | `SparseEmbedding` → `SparseVector` |
| Qdrant Query API | dense, sparse, RRF, DBSF, weighted RRF, MMR, boost |
| TurboPuffer query | dense ANN + filters; hybrid/sparse slices as implemented; else fail closed |
| Mistral OCR | `pages[]` with `markdown` |
| Gemini command OCR | `generateContent` multimodal path; page selection unsupported |

Follow-up: live conformance scripts when API keys are present (shape only). Extend TurboPuffer only with explicit query-plan contract tests.
