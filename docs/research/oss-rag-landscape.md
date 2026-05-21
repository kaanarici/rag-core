# OSS RAG landscape

**Accessed:** 2026-05-20  
**Purpose:** Steal/refuse matrix for planning — not feature parity mandates (per `AGENTS.md`).

## RAGFlow

| Area | Public claim | rag-core |
|------|--------------|----------|
| Deep PDF/OCR | DeepDoc, layout | **Steal** ambition — in-repo PDF inspector + OCR routing |
| Hybrid + rerank | Yes | **Mirror** — core strength |
| Agent canvas / GraphRAG UI | Yes | **Refuse** |
| Connector marketplace | Drive, Notion, S3, … | **Refuse** — [connector-pattern.md](../embedding/connector-pattern.md) |
| Docker microservices platform | Yes | **Refuse** platform scope; **steal** compose for Qdrant+serve only |

Source: [RAGFlow GitHub](https://github.com/infiniflow/ragflow) README (~80k stars, 2026-05).

## Haystack 2

| Area | Public claim | rag-core |
|------|--------------|----------|
| Pipeline graph DSL | Core model | **Refuse** — ADR linear pipeline |
| Retriever/rerank vocabulary | Rich | **Steal** technique names in docs/profiles |
| Hosted deepset cloud | Optional | **Refuse** |

Source: [Haystack docs](https://docs.haystack.deepset.ai/) (2026-05).

## LlamaIndex

| Area | Public claim | rag-core |
|------|--------------|----------|
| 160+ connectors | Data loaders | **Refuse** marketplace; app sync → ingest |
| Framework + LlamaCloud | Default cloud index | **Refuse** second framework; library embed |
| Query/chat engines | Yes | **Steal** separation of index vs query; app owns chat |

Source: [LlamaIndex docs](https://docs.llamaindex.ai/) (2026-05).

## RAG-Anything / multimodal stacks

**Steal** multimodal ingest ideas when they improve parsing. **Defer** until core retrieval DX is undeniable.

## Master matrix

| Theme | Mirror | Steal | Refuse | Defer |
|-------|--------|-------|--------|-------|
| Hybrid retrieval quality | ✓ | | | |
| Citation/context UX | ✓ | | | |
| Visual agent builder | | | ✓ | |
| Graph orchestration DSL | | | ✓ | |
| Connector marketplace | | | ✓ | |
| OCR/deep doc narrative | | ✓ | | full parity |
| Public benchmark hub | | | | ✓ see retrieval-benchmark-corpus.md |

Parent: [one-repo-retrieval-engine-strategy.md](../plans/one-repo-retrieval-engine-strategy.md).
