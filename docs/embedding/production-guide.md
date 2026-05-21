# Production embed guide

Use `RAGCore` inside your application when you own auth, tenancy, chat, and connectors. This guide covers lifecycle, tenancy binding, and shutdown — not hosted-platform features.

## One `RAGCore` per worker

Create a single `RAGCore` instance per process (or per async worker). Reuse it across requests. Do not construct a new core per HTTP request unless you accept cold-start cost and connection churn.

```python
from rag_core.core import RAGCore
from rag_core.core_models import RAGCoreConfig

core = RAGCore(config)
await core.ensure_ready()
# serve traffic
await core.close()
```

For ASGI apps, initialize on startup and close on shutdown (see `examples/embedded_service.py`).

## Tenancy binding

**Rule:** derive `namespace` and `corpus_id` from the authenticated principal in your app — never trust raw client input.

| Layer | Owns |
|-------|------|
| Your app | User identity, org/tenant ID, connector credentials |
| `RAGCore` | Parse, chunk, index, search, context pack, traces |

Example mapping:

```python
tenant_id = auth.current_tenant_id()  # your code
namespace = f"tenant:{tenant_id}"
corpus_id = request.corpus_slug  # validate against tenant allowlist
```

`retrieve_context` and `search` both require explicit `namespace` + `corpus_ids`.

## Configuration

- Build `RAGCoreConfig` from environment (same variables as `rag-core serve`; see [self-host/config.md](../self-host/config.md)).
- Use dimension-aware Qdrant collection naming when changing embedding models.
- Set `processing_version` when parser/chunk policy changes force reindex.

## Observability

- Attach an `EventSink` or write JSONL from CLI-style flows.
- Summarize traces with `rag_core.events.summarize_search_trace_payload_runs`.
- Export Ragie-shaped hits via `rag_core.events.export.to_retrieval_hits` for your chat layer.

## Shutdown

Always `await core.close()` on process exit. `RAGCore` may hold vector-store clients and caches; skipping close leaks connections in long-running workers.

## Self-host vs embed

| Need | Use |
|------|-----|
| Library control, custom auth | Embed `RAGCore` (this guide) |
| Internal microservice, tools/SDK | `rag-core serve` + [auth middleware](../self-host/auth.md) |

## Next

- [connector-pattern.md](connector-pattern.md) — replace managed connectors without a marketplace
- [../quickstart.md](../quickstart.md) — no-key proof path
- [../expectations.md](../expectations.md) — hit and context contracts
