# Managed RAG landscape

**Accessed:** 2026-05-20  
**Purpose:** Inform rag-core positioning vs hosted retrieval APIs (escape targets).

## Summary

Managed platforms sell a closed retrieval plane: ingest, chunking, search, rerank, and connector sync inside their boundary. rag-core mirrors **hit-layer** ergonomics ([expectations.md](../expectations.md)) while keeping ingest policy, tenancy, and auth in your app.

## Ragie (primary in-repo benchmark)

| Surface | Ragie | rag-core |
|---------|-------|----------|
| Hosted ingest + connectors | Yes | **Refuse** — app-owned sync |
| Semantic search API | `/retrievals`, scored chunks | `search` / HTTP `/v1/search` |
| Context for LLM | SDK helpers | `retrieve_context` |
| Trace/debug | Limited | JSONL + trace summaries (**mirror** inspectability) |
| Auth/tenancy | Hosted accounts | **Refuse** in core — [auth.md](../self-host/auth.md) |

Sources: [Ragie docs](https://docs.ragie.ai/) (product pages, API reference), in-repo `docs/expectations.md`.

## Peer class (same escape bucket)

| Vendor | Notes | rag-core stance |
|--------|-------|-----------------|
| Vectara | API-first semantic search + ingest | **Mirror** hit/context shape only; **refuse** hosted index lock-in |
| Contextual.ai / cloud KB products | Bundled with models | **Defer** — evaluate per customer |
| Generic “knowledge base” APIs | Opaque chunks | **Steal** developer expectation of JSON hits; **refuse** black-box chunk policy |

## Decision matrix

| Capability | Mirror | Steal | Refuse | Defer |
|------------|--------|-------|--------|-------|
| Scored chunk JSON | ✓ | | | |
| retrieve-context ergonomics | ✓ | | | |
| Hosted connectors | | | ✓ | |
| Webhook ingest pipelines | | | ✓ | |
| Billing/admin | | | ✓ | |
| Deep multimodal ingest marketing | | ✓ narrative | | parity |

Parent: [one-repo-retrieval-engine-strategy.md](../plans/one-repo-retrieval-engine-strategy.md).
