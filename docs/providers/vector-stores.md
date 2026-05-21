# Vector Store Providers

`rag-core` keeps vector stores behind one base contract so applications can change storage backends without forking the engine. Provider maturity has levels: a first-party adapter is not automatically the default runtime path.

## Provider Maturity

- `first-party default`: repo-owned contract, diagnostics, tests, and the default runtime path
- `first-party optional`: repo-owned contract, diagnostics, and tests, but you opt into it explicitly
- `utility adapter`: useful for tests or local smoke flows, not a production storage recommendation

| Provider | Maturity | Best fit | Current entrypoint | Notes |
| --- | --- | --- | --- | --- |
| Qdrant | first-party default | local development, self-hosting, persistent CLI ingest/search, and self-managed production deployments | `QdrantConfig`, CLI `--qdrant-*` flags, `QdrantVectorStore` | Default CLI and runtime path. Supports dense, sparse, hybrid RRF/DBSF/weighted RRF, MMR, and boost query plans, document lookup, deletes, and collection compatibility checks. |
| TurboPuffer | first-party optional | remote vector storage where the app wants a managed vector-store backend | `VectorStoreConfig(provider="turbopuffer", ...)`, CLI `--vector-store turbopuffer`, `TurboPufferVectorStore`, or `VECTOR_STORES.create("turbopuffer", ...)` | First-party optional path: dense ANN, payload filters, document lookup, deletes, health, and explicit runtime assembly. Sparse and hybrid query planning are unsupported. |
| In-memory | utility adapter | unit tests and no-service smoke flows | `InMemoryVectorStore` | Not a production storage backend. |

## First-Party Support Contract

First-party support means more than adapter registration. A provider is only first-party here when the repo owns the behavior contract, diagnostics, and tests that make the tradeoffs explicit:

- `doctor --json` or `describe_runtime()` must expose enough non-secret runtime shape to debug provider selection, dimensions, collection/namespace identity, and relevant env-backed settings.
- `StoreCapabilities` must truthfully declare required and optional behavior such as document-record lookup, per-point delete, dense-vector dimensions, and query-plan stages.
- Contract tests must prove common vector-store behavior across providers; provider-specific tests must pin backend wire shape, filter translation, health, deletes, and malformed-result handling.
- Docs must state the best fit, entrypoints, known limitations, and migration path without implying backend identity equivalence.

Current validation surfaces:

These paths are source-checkout validation references. The sdist ships docs and examples, but not the repository test suite.

| Surface | What it proves |
| --- | --- |
| `tests/test_vector_store_contract.py` | Cross-provider search, scoping, filters, document records, deletes, point deletes, and health behavior. |
| `tests/test_store_capabilities.py` | Declared capabilities, core assembly requirements, and capability-aware ingest/delete behavior. |
| `tests/test_qdrant_helpers.py` | Qdrant collection, query, dimension, payload, and health edge cases outside the shared contract. |
| `tests/test_turbopuffer_store.py` | TurboPuffer schema, distance conversion, filter tuples, upsert/delete/search materialization, and optional SDK wiring. |
| `rag-core doctor --json` | Runtime selection, dimension shape, collection/namespace identity, query-plan support, and secret-redacted provider env state. |
| CI wheel smokes | Base wheel import/doctor plus TurboPuffer extra install and doctor diagnostics. |

## Qdrant Default Path

Qdrant is the first-party default because it works for local and self-hosted use without requiring a managed control plane. The CLI uses Qdrant for `ingest`, `search`, `eval`, and `doctor --check-store` unless `--vector-store turbopuffer` is selected explicitly.

Use embedded memory storage for quick checks:

```bash
uv run rag-core doctor --check-store --qdrant-location :memory: --json
```

Use a persistent local or remote Qdrant service for real corpora:

```bash
uv run rag-core ingest docs --namespace acme --corpus-id help \
  --qdrant-url http://localhost:6333 \
  --embedding-model text-embedding-3-small --embedding-dimensions 1536
```

`rag-core doctor --fix` can create a missing Qdrant collection and reports dimension drift without mutating an existing mismatched collection.

## TurboPuffer Optional Path

Install the TurboPuffer extra from the GitHub source URL:

```bash
uv add "rag-core[turbopuffer] @ git+https://github.com/kaanarici/rag-core.git"
```

or from a local checkout:

```bash
uv sync --extra turbopuffer
```

The adapter uses one configured TurboPuffer namespace as the physical store, and that physical namespace must match TurboPuffer's `[A-Za-z0-9-_.]{1,128}` namespace rule. The `namespace` and `corpus_id` fields remain ordinary filterable payload fields inside that store, so the same `SearchQuery` scoping model works across the default and optional paths.

Use it through the shared config:

```python
from rag_core import RAGCore, RAGCoreConfig
from rag_core.config import (
    EmbeddingConfig,
    TurboPufferVectorStoreConfig,
    VectorStoreConfig,
)

core = RAGCore(
    RAGCoreConfig(
        vector_store=VectorStoreConfig(
            provider="turbopuffer",
            turbopuffer=TurboPufferVectorStoreConfig(
                namespace="product_docs",
                region="aws-us-west-2",
            ),
        ),
        embedding=EmbeddingConfig(model="text-embedding-3-small", dimensions=1536),
    )
)
```

or from the CLI:

```bash
uv run rag-core doctor --vector-store turbopuffer \
  --turbopuffer-namespace product_docs \
  --embedding-model text-embedding-3-small \
  --embedding-dimensions 1536 \
  --json
```

Construct it directly:

```python
from rag_core.search.providers import TurboPufferVectorStore

store = TurboPufferVectorStore(
    namespace="product_docs",
    dense_dimensions=1536,
    region="aws-us-west-2",
)
```

or through the registry:

```python
from rag_core.search.providers import VECTOR_STORES

store = VECTOR_STORES.create(
    "turbopuffer",
    namespace="product_docs",
    dense_dimensions=1536,
    region="aws-us-west-2",
)
```

The SDK reads `TURBOPUFFER_API_KEY`, `TURBOPUFFER_REGION`, and `TURBOPUFFER_BASE_URL` when constructor values are omitted. `rag-core doctor --json` reports whether those settings are present without printing secrets.

Current limitations:

- Qdrant remains the default; TurboPuffer must be selected explicitly through `VectorStoreConfig.provider` or `--vector-store turbopuffer`.
- TurboPuffer honors explicit dense-only query plans. Sparse, hybrid, MMR, boost, Geo metadata filters, and multi-query plans fail closed with `UnsupportedQueryStage` instead of silently downgrading to dense ANN.
- TurboPuffer document-record lookup uses a lookup query for the sample row plus a separate count aggregation query. Missing or malformed count aggregation is treated as a provider contract failure instead of guessing a one-chunk record.
- Migrate by re-ingesting the source corpus into a TurboPuffer-backed `RAGCore` instance.

## Query Behavior Differences

Qdrant has the broadest first-party query-plan support in `rag-core`: dense, sparse, hybrid RRF, DBSF, weighted RRF, nested prefetches inside fused or MMR-reranked plans, MMR, and boost through the existing query-plan translator.

TurboPuffer satisfies the base `VectorStore` contract. That means dense vector search, top-level namespace/corpus/document/content-type filters, non-Geo metadata filter AST translation, count-backed document record lookup, delete-by-filter, point delete, and health checks. Explicit dense-only query plans are honored. Sparse, hybrid, Geo filters, and multi-query planning require separate contract slices so query behavior is tested and documented instead of inferred from provider SDKs. Unsupported query-plan stages raise `UnsupportedQueryStage`.

The in-memory adapter advertises dense, sparse, and hybrid RRF query-plan support for tests and no-service smoke flows. It intentionally rejects DBSF, MMR, boost, nested prefetches, and unsupported channels so local tests do not imply more provider behavior than the adapter can execute.

When callers do not provide an explicit query plan, `rag-core` uses the documented default search profile, `balanced`, when the active store supports it. If not, it falls back through the strongest supported hybrid plan, then dense-only, then sparse-only. Stores that do not declare query-plan support receive the baseline `SearchQuery` without a plan. Explicit caller-provided plans are never rewritten; if the active store declares query-plan capabilities and the selected plan exceeds them, `rag-core` fails before embedding work or backend calls.

## Migration Notes

Migration is source-of-truth based:

1. Keep source files and JSONL manifests as the authoritative corpus record.
2. Create a TurboPuffer-backed `RAGCore` with the same embedding provider and dimensions.
3. Re-ingest with the same `namespace`, `corpus_id`, document keys, and processing version.
4. Run the vector-store contract tests and a corpus-specific eval before switching application traffic.

Do not point an existing Qdrant collection name at TurboPuffer and assume identity equivalence. The vector point payload shape is portable, but backend storage and query planning are provider-specific.
