# Embed rag-core

Use one `RAGCore` instance per worker process. Your application owns authentication,
connectors, chat, and model calls. rag-core owns parse → chunk → index → search →
context packs and retrieval traces.

## Minimal loop

```python
import asyncio
from rag_core import RAGCore
from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.core_models import RAGCoreConfig

async def main() -> None:
    config = RAGCoreConfig(
        qdrant=QdrantConfig(location="./rag-core-qdrant"),
        embedding=EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
        ),
    )
    async with RAGCore(config) as core:
        await core.ingest_bytes(
            file_bytes=b"Invoices can be paid by card or ACH.",
            filename="billing.txt",
            mime_type="text/plain",
            namespace="tenant:acme",  # from your auth
            corpus_id="help",
        )
        pack = await core.retrieve_context(
            query="How can customers pay?",
            namespace="tenant:acme",
            corpus_ids=["help"],
            limit=8,
            rerank=False,  # pass True when a reranker is configured
        )
        print(pack.as_text())

asyncio.run(main())
```

No API keys yet? Use `from rag_core.demo import build_demo_core` — deterministic
embeddings for wiring checks only, not semantic retrieval. See [quickstart.md](quickstart.md).

## Tenancy

- **`namespace`** — tenant or workspace id from your auth layer.
- **`corpus_id`** — collection within that tenant (product area, KB, project).
- **`document_id` / `document_key`** — stable ids your app assigns; map external
  sources in your sync jobs.

Never accept raw `namespace` / `corpus_id` from model tool calls without binding them
to the authenticated session in your HTTP handler.

## Retrieval defaults

| Surface | Hybrid / lexical | Rerank |
| --- | --- | --- |
| `RAGCore.search` / `retrieve_context` | on (`use_lexical_search=True`) | off unless `rerank=True` and reranker configured |
| CLI `search` / `retrieve-context` | on | `--rerank` to enable |
| HTTP `POST /v1/search` | on | `"rerank": true` in JSON body |

Pass `rerank=True` only when `RAGCoreConfig.reranker` (or CLI flags) point at a real
reranker provider. See [expectations.md](expectations.md).

## Agent tools

`rag_core.contracts` parses model tool payloads; your endpoint enforces scope and calls
`retrieve_context`. Examples: [search_endpoint.py](../examples/search_endpoint.py),
[vercel_ai_sdk_search_tool.ts](../examples/vercel_ai_sdk_search_tool.ts).

## Worker lifecycle

Long-running services should construct `RAGCore` once and reuse it. See
[embedded_service.py](../examples/embedded_service.py).

## Evals in your repo

Keep labeled cases in your application repository and run `rag_core.evals` against
your configured embeddings — not the library’s CI fixture corpus. See
[retrieval_eval.py](../examples/retrieval_eval.py).

## Configured example (credentials required)

```bash
export OPENAI_API_KEY=sk-...
uv run python -m examples.configured_retrieval
```

Not run in CI. Smoke without keys: `python -m rag_core.quickstart` or
`examples/embedded_service`.

## Reference

| Topic | Doc |
| --- | --- |
| Hit and context JSON | [expectations.md](expectations.md) |
| Beta-stable surface | [stability.md](stability.md) |
| Providers and extras | [providers.md](providers.md) |
| Optional HTTP runtime | [self-host.md](self-host.md) |
