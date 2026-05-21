# Custom Providers

`rag-core` can use registered providers without forking the engine. Use this for application-owned embeddings, rerankers, OCR, vector stores, caches, or search sidecars.

## Extension Points

Prefer the narrow protocol for the category you are replacing. Register named providers for app-selected runtime dependencies, and inject concrete instances directly when the provider is owned by one application boundary. A registry entry does not automatically make a category selectable from `RAGCoreConfig`.

| Category | Protocol or contract | Registry or factory | Runtime selection | Required shape |
| --- | --- | --- | --- | --- |
| Dense embeddings | `rag_core.search.types.EmbeddingProvider` | `EMBEDDING_PROVIDERS`, `create_embedding_provider()` | `RAGCoreConfig.embedding`, CLI flags, or `RAGCore(embedding_provider=...)` | `model_name`, `dimensions`, `embed_texts()`, `embed_query()` |
| Sparse embeddings | `rag_core.search.types.SparseEmbedder` | `SPARSE_EMBEDDERS`, `create_sparse_embedder()` | `RAGCore(sparse_embedder=...)`; the default core uses FastEmbed BM25 | `embed_texts()`, `embed_query()`, `embed_query_multi()` |
| Rerankers | `rag_core.search.types.RerankerProvider` | `RERANKER_PROVIDERS`, `create_reranker()` | `RAGCoreConfig.reranker`, CLI flags, or `RAGCore(reranker=...)` | `rerank(query, documents, top_k)` returning indexed scores |
| Vector stores | `rag_core.search.types.VectorStore` | `VECTOR_STORES` | `RAGCoreConfig.vector_store`, CLI flags, or `RAGCore(vector_store=...)` | `capabilities`, `upsert()`, `search()`, `delete()`, health, close, optional document lookup and point delete |
| OCR | `rag_core.documents.OcrProvider` | `rag_core.documents.OCR_PROVIDERS`, `rag_core.documents.create_ocr_provider()` | `RAGCore(ocr_provider=...)`; no `RAGCoreConfig` field | `provider_name`, `model_name`, page-selection support, `extract_markdown()` |
| Search sidecars | `rag_core.search.types.SearchSidecar` | `SEARCH_SIDECARS`, `create_search_sidecar()` | `RAGCoreConfig.ingest.lexical_search_provider`, `enable_lexical_search`, or `RAGCore(search_sidecar=...)` | upsert/delete records and return lexical/exact-match `SearchResult` rows |
| Embedding cache | `rag_core.search.providers.EmbeddingCache` | `EMBEDDING_CACHES`, `create_embedding_cache()` | `RAGCoreConfig.ingest.embedding_cache_provider` or `RAGCore(embedding_cache=...)` | `get()` and `put()` by `EmbedCacheKey` |
| Chunk context cache | `rag_core.search.providers.ChunkContextCache` | `CHUNK_CONTEXT_CACHES`, `create_chunk_context_cache()` | `RAGCore(chunk_context_cache=...)`; no `RAGCoreConfig` field | `get()` and `put()` by `ChunkContextKey` |
| Chunking | `rag_core.documents.chunking.ChunkingStrategy` | `CHUNKING_STRATEGIES`, `create_chunking_strategy()` | lower-level prepare paths; no top-level `RAGCoreConfig` selector | `chunk(text, config)` returning `PreparedChunk` rows |
| Converters | `rag_core.documents.converters.BaseConverter` | `get_converter()`, `convert_file()` | converter registry and parse helpers, not `RAGCoreConfig` | `convert(file_bytes, filename, mime_type)` returning `ConversionResult` |

Vector stores should declare `StoreCapabilities` truthfully. Query-plan support must fail closed when a backend cannot execute a stage; do not silently downgrade sparse, hybrid, MMR, or boost plans to dense search.

Minimal embedding provider:

```python
from rag_core.search.providers import EMBEDDING_PROVIDERS, create_embedding_provider


class MyEmbeddingProvider:
    model_name = "my-embedding-model"
    dimensions = 384

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] * self.dimensions for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        return [float(len(query))] * self.dimensions


EMBEDDING_PROVIDERS.register("my-embeddings", lambda **_: MyEmbeddingProvider())
provider = create_embedding_provider(provider="my-embeddings")
```

The same registry shape is used for rerankers, sparse embedders, OCR providers, vector stores, search sidecars, and caches. Keep provider names stable in your app config, and pass explicit config into `RAGCoreConfig` instead of relying on hidden environment loading.

Install the matching extra before selecting an optional first-party provider: `rerank` for Cohere, `voyage`, `zeroentropy`, `turbopuffer`, `anthropic`, `opentelemetry`, `langchain`, or `openai-agents`. The `semantic` and `html` extras add local parsing/chunking helpers rather than remote provider clients.

## Support Levels And Diagnostics

First-party support means behavior, tests, docs, and diagnostics, not only a registered factory. `rag-core doctor --json` reports non-secret readiness for model providers and runtime provider categories:

| Category | Default or first-party path | Diagnostic surface |
| --- | --- | --- |
| Dense embeddings | OpenAI default, Voyage and ZeroEntropy optional | configured provider, registered names, package availability, API-key presence, model dimensions |
| Sparse embeddings | FastEmbed BM25 with optional SPLADE | registered names, package availability, sparse model env presence |
| Rerankers | no-op default, Cohere, Voyage, and ZeroEntropy optional | requested provider, effective provider, fallback reason, package availability, API-key presence |
| OCR | Mistral and Gemini command-backed adapters | registered names, API-key presence, page-selection support |
| Contextualizers | no-op default, Anthropic optional | package availability, API-key presence, injection surface |
| Caches | none, in-memory, and SQLite | configured cache provider and registered names |
| Search sidecars | portable lexical sidecar | configured provider and registered names |
| Event sinks | none, logging, JSONL, buffer, and OpenTelemetry | package availability and injection surface |

Vector-store diagnostics are reported separately because they include backend capabilities, query-plan support, collection or namespace identity, and provider-specific runtime settings.
