# Vector Store Providers

`rag-core` keeps vector stores behind one base contract so applications can change storage backends without forking the engine. The **default wheel** ships **Qdrant** as the first-party vector store.

## Provider Maturity

- `first-party default`: repo-owned contract, diagnostics, tests, and the default runtime path
- `first-party optional`: managed adapter behind `--extra`; same contract tests where applicable
- `utility adapter`: useful for tests or local smoke flows, not a production storage recommendation

| Provider | Maturity | Best fit | Current entrypoint | Notes |
| --- | --- | --- | --- | --- |
| Qdrant | first-party default | local development, self-hosting, persistent CLI ingest/search, and self-managed production deployments | `QdrantConfig`, CLI `--qdrant-*` flags, `QdrantVectorStore` | Default CLI and runtime path. Supports dense, sparse, hybrid RRF/DBSF/weighted RRF, MMR, and boost query plans, document lookup, deletes, and collection compatibility checks. |
| TurboPuffer | first-party optional | managed vector + FTS without operating Qdrant | `--vector-store turbopuffer`, `--turbopuffer-*`, `uv sync --extra turbopuffer` | Not in the default wheel. Dense ANN, BM25 hybrid (RRF), SparseKNN. Fail-closed on unsupported query-plan stages. |
| In-memory | utility adapter | unit tests and no-service smoke flows | `InMemoryVectorStore` | Not a production storage backend. |

## First-Party Support Contract

First-party support means more than adapter registration. A provider is only first-party here when the repo owns the behavior contract, diagnostics, and tests that make the tradeoffs explicit:

- `doctor --json` or `describe_runtime()` must expose enough non-secret runtime shape to debug provider selection, dimensions, collection identity, and relevant env-backed settings.
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
| `tests/test_turbopuffer_*.py` | TurboPuffer wire shape, query-plan guards, and result validation. |
| `rag-core doctor --json` | Runtime selection, dimension shape, collection identity, query-plan support, and secret-redacted provider env state. |
| CI wheel smoke | Base wheel import, doctor, and consumer smoke app. |

## Qdrant Default Path

Qdrant is the first-party default because it works for local and self-hosted use without requiring a managed control plane. The CLI uses Qdrant for `ingest`, `search`, `retrieve-context`, and `doctor --check-store`.

Use embedded memory storage for quick checks:

```bash
uv run rag-core doctor --check-store --qdrant-location :memory: --json
```

Use a persistent local or remote Qdrant service for real corpora:

```bash
uv run rag-core ingest ./docs \
  --namespace acme \
  --corpus-id help \
  --qdrant-url http://localhost:6333 \
  --embedding-model text-embedding-3-small \
  --embedding-dimensions 1536
```

Programmatic assembly:

```python
from rag_core import RAGCore, RAGCoreConfig
from rag_core.config import EmbeddingConfig, QdrantConfig

core = RAGCore(
    RAGCoreConfig(
        qdrant=QdrantConfig(url="http://localhost:6333", collection="product_docs"),
        embedding=EmbeddingConfig(model="text-embedding-3-small", dimensions=1536),
    )
)
```

## TurboPuffer (optional extra)

TurboPuffer is **not** in the default wheel. Install the extra and point the CLI at a namespace:

```bash
uv sync --extra turbopuffer
export TURBOPUFFER_API_KEY=...
uv run rag-core doctor --json \
  --vector-store turbopuffer \
  --turbopuffer-namespace my-docs \
  --embedding-model text-embedding-3-small
```

**Research:** [docs/research/turbopuffer-landscape.md](../research/turbopuffer-landscape.md) — phased slices TP1 (dense) → TP2 (hybrid) → TP3 (sparse).

**Supported today:** dense ANN, namespace/corpus/document filters, upsert/delete, health, doctor diagnostics, hybrid RRF (BM25 + dense when `lexical_query` is set), SparseKNN for sparse channels.

**Not supported:** DBSF, weighted RRF, MMR, boost, nested prefetches — fail closed with `UnsupportedQueryStage`.

Do not use TurboPuffer for no-key Journey A or default `docker compose` smoke (requires API key).

## Migration Notes

When moving between vector stores:

1. Re-ingest the source corpus into the target-backed `RAGCore` instance with the same embedding model and dimensions.
2. Treat collection/namespace identity as provider-specific even when payload fields are portable.
3. Re-run `rag-core doctor --json` and a small eval set (`examples/retrieval_eval.py`) before promoting a new backend.
