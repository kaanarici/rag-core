# Providers

Vector stores, custom extension points, and provider wire-shape notes.

## Vector stores

`rag-core` keeps vector stores behind one contract. The **default wheel** ships **Qdrant** as the first-party vector store.

### Maturity

| Level | Meaning |
| --- | --- |
| first-party default | Default CLI/runtime path, contract tests, diagnostics |
| first-party optional | Managed adapter behind `--extra`; adapter tests by default, optional live smoke when credentials are present |
| utility adapter | Tests/smoke only (e.g. in-memory) |

| Provider | Maturity | Entrypoint |
| --- | --- | --- |
| Qdrant | first-party default | `QdrantConfig`, `--qdrant-*`, `rag_core.search.providers.QdrantVectorStore` |
| TurboPuffer | first-party optional | `--vector-store turbopuffer`, `uv sync --extra turbopuffer`, `rag_core.search.providers.turbopuffer_store.TurboPufferVectorStore` |
| In-memory | utility | `RAGCore(vector_store=...)` with `rag_core.search.providers.memory_store.InMemoryVectorStore` |

`doctor --json` and `StoreCapabilities` must match real behavior. Built-in maturity,
query-plan, and metadata-filter specs live in `vector_store_capabilities.py`; adapters,
runtime descriptions, diagnostics, and docs checks read those typed specs. Contract tests:
`tests/test_vector_store_contract.py`, provider-specific suites.

### Qdrant (default)

```bash
uv run rag-core doctor --check-store --qdrant-location :memory: \
  --embedding-provider demo --embedding-dimensions 64 --json
uv run rag-core ingest ./docs --namespace acme --corpus-id help \
  --qdrant-url http://127.0.0.1:6333 \
  --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536
```

### TurboPuffer (optional)

Not in the default wheel. Requires `TURBOPUFFER_API_KEY`. Upserts serialize `sparse_vector`
when points carry sparse channels. Supports dense ANN, hybrid RRF (BM25 + dense when
`lexical_query` is set), and SparseKNN on the sparse column. Proof:
`tests/test_turbopuffer_store.py`, `tests/test_turbopuffer_query_plan_guard.py`,
`tests/test_turbopuffer_result_shape_validation.py`, and `turbopuffer-fake` in
`tests/test_vector_store_contract.py`; live smoke is `@pytest.mark.live`.
Unsupported stages (DBSF, MMR, boost, …) fail closed.

Do not use TurboPuffer for no-key smoke or default `docker compose` (demo embeddings).

### Migration

Re-ingest with the same embedding model/dimensions, re-run `doctor --json` and `examples/retrieval_eval.py` before promoting a new vector store provider.

---

## Custom providers

Register provider-backed extension points or inject application-owned embeddings, rerankers, OCR, vector stores, caches, sidecars, contextualizers, and event sinks. Registry-backed seams are selected by name; direct-injection seams list their built-in classes. A registry entry does not automatically make a category selectable from `RAGCoreConfig`.

| Category | Protocol | Registry or built-ins | Runtime selection |
| --- | --- | --- | --- |
| Dense embeddings | `rag_core.search.provider_protocols.EmbeddingProvider` | `EMBEDDING_PROVIDERS` | `RAGCoreConfig.embedding`, CLI, or `RAGCore(embedding_provider=...)` |
| Sparse embeddings | `rag_core.search.provider_protocols.SparseEmbedder` | `SPARSE_EMBEDDERS` | `RAGCore(sparse_embedder=...)` |
| Rerankers | `rag_core.search.provider_protocols.RerankerProvider` | `RERANKER_PROVIDERS` | `RAGCoreConfig.reranker`, CLI, or `RAGCore(reranker=...)` |
| Vector stores | `rag_core.search.provider_protocols.VectorStore` | `VECTOR_STORES` | Qdrant/TurboPuffer via `RAGCoreConfig.vector_store` or CLI; any store via `RAGCore(vector_store=...)` |
| OCR | `rag_core.documents.OcrProvider` | `OCR_PROVIDERS` | `RAGCore(ocr_provider=...)`; no `RAGCoreConfig` field |
| Search sidecars | `rag_core.search.provider_protocols.SearchSidecar` | `SEARCH_SIDECARS` | `RAGCoreConfig.ingest.lexical_search_provider` or `RAGCore(search_sidecar=...)` |
| Embedding cache | `rag_core.search.providers.EmbeddingCache` | `EMBEDDING_CACHES` | `RAGCoreConfig.ingest.embedding_cache_provider` or `RAGCore(embedding_cache=...)` |
| Chunk context cache | `rag_core.search.providers.ChunkContextCache` | `CHUNK_CONTEXT_CACHES` | `RAGCore(chunk_context_cache=...)` only |
| Chunk contextualizers | `rag_core.documents.ChunkContextualizer` | Built-ins: `NoOpContextualizer`, `AnthropicChunkContextualizer`, `CachingContextualizer` | `RAGCore(chunk_contextualizer=...)` only |
| Event sinks | `rag_core.events.EventSink` | Built-ins: `NoOpSink`, `LoggingSink`, `JsonlSink`, `EventBuffer`, `MultiSink`, `OpenTelemetrySink` | `RAGCore(event_sink=...)` only |
| Chunking | `rag_core.documents.chunking.ChunkingStrategy` | `CHUNKING_STRATEGIES` | prepare paths |
| Converters | `rag_core.documents.converters.BaseConverter` | Dedicated loader: `get_converter()`, `convert_file()` | `ConversionResult` via converter registry |

Declare `rag_core.search.provider_protocols.StoreCapabilities` truthfully;
unsupported query-plan stages must fail closed. `rag_core.search.types` re-exports
these protocol objects for compatibility alongside shared search value objects, but
`provider_protocols` is the ownership module for provider-author contracts.

Install extras before optional first-party providers: `rerank`, `voyage`, `zeroentropy`, `turbopuffer`, `runtime`, `anthropic`, `opentelemetry`, `langchain`, `openai-agents`. `semantic` and `html` add local parsing helpers.

### Support levels and diagnostics

`rag-core doctor --json` reports non-secret readiness:

Provider diagnostics use these JSON support-level values:

| JSON value | Meaning |
| --- | --- |
| `default` | Default active provider for the category |
| `default_noop` | Built-in no-op provider used when a category is disabled |
| `first_party_optional` | First-party adapter behind an optional package, credential, or explicit configuration |
| `first_party_utility` | First-party local utility such as demo embeddings, the in-memory vector store, caches, event sinks, or lexical sidecars |
| `injected` | Application-owned provider passed directly to `RAGCore` |

| Category | Diagnostic surface |
| --- | --- |
| Dense embeddings | provider, registered names, package availability, API-key presence, dimensions |
| Sparse embeddings | registered names, package availability |
| Rerankers | requested/effective provider, fallback reason |
| OCR | registered names, API-key presence |
| Contextualizers | no-op default, package availability, injection surface |
| Caches | configured provider and registered names |
| Search sidecars | configured provider and registered names |
| Event sinks | package availability, injection surface |
| Vector stores | capabilities, query-plan support, collection/namespace identity |

---

## Provider output shapes

Covered by adapter/parser tests unless a test is marked `live`. This is a contract
audit for first-party adapters, not a benchmark.

| Provider | Notes |
| --- | --- |
| Demo embeddings | Deterministic no-key smoke provider; not semantic retrieval |
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

Provider output-shape rows are adapter/parser tests unless a test is marked `live`.
Extend TurboPuffer only with explicit query-plan contract tests.
