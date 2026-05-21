# Convex landscape (rag-core)

**Accessed:** 2026-05-20  
**Purpose:** Decide whether rag-core needs first-party Convex integration, given [Convex RAG](https://www.convex.dev/can-do/rag) and the [`@convex-dev/rag`](https://www.convex.dev/components/rag) component.

## What Convex offers

Convex is a **TypeScript-first reactive backend** (queries, mutations, actions, real-time subscriptions). For RAG it provides:

| Surface | Role |
|---------|------|
| Built-in **vector search** on tables | Store embeddings, filter, paginate |
| **`@convex-dev/rag` component** | Namespaces, chunking helpers, embedding hooks, semantic search, chunk context, filters, migrations |
| App hosting | Auth, scheduling, file storage patterns typical of product backends |

The component is an **in-Convex retrieval layer** for apps already on Convex — not a Python vector-store driver.

## Overlap with rag-core

| Concern | rag-core | Convex RAG |
|---------|----------|------------|
| Language / runtime | Python library + optional `serve` | TypeScript / Convex cloud |
| Ingest + chunking | `RAGCore`, converters, manifests | Component + your actions |
| Vector index | Pluggable `VectorStore` (Qdrant, TurboPuffer, memory) | Convex tables + vector indexes |
| Hybrid / query plans | `QueryPlan`, profiles, provider capability matrix | Component APIs + Convex filters |
| Escape managed black-box RAG | Core mission | Different lane (build *on* Convex) |

**Overlap is conceptual** (namespaces, chunks, semantic search), not architectural. You do not “point” `RAGCore` at Convex like Qdrant without a new adapter that speaks Convex’s mutation/query API from Python — which is awkward compared to running retrieval inside Convex actions in TypeScript.

## Steal / mirror / refuse

| Item | Decision |
|------|----------|
| First-party **Convex `VectorStore` adapter** inside rag-core | **Refuse for v0** — wrong runtime boundary; high coupling to Convex deployment |
| **Integration guide**: embed `RAGCore` from a Convex **action** (HTTP to self-hosted `serve`, or subprocess/CLI) | **Mirror** — honest escape hatch for teams on Convex who want rag-core quality |
| **Example app** pairing Convex auth + rag-core `serve` | **Steal** later — Journey C shape, not core wheel bloat |
| Re-implement `@convex-dev/rag` inside Python | **Refuse** |
| Document “use Convex RAG instead of rag-core when…” | **Steal** — reduces wrong-fit adoption |

## Recommendation

**No first-party Convex backend in rag-core for v0 pre-release.**

- Teams **all-in on Convex** should evaluate [@convex-dev/rag](https://www.convex.dev/components/rag) for retrieval that lives next to their data and auth.
- Teams **leaving managed RAG** who need **Python control**, **Qdrant/TurboPuffer**, and **shared `QueryPlan` semantics** should use rag-core as the retrieval plane and optionally call it from Convex actions via the self-host API.

If demand appears, the right v0+ slice is **documentation + a minimal Convex action example** that calls `rag-core serve`, not a Convex adapter in the default package.

## Out of scope

- Convex Auth / RLS patterns inside rag-core
- Shipping Convex schema or component code in this repo
- Parity matrix claiming feature-equal to `@convex-dev/rag`
