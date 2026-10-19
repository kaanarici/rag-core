# Architecture

rag-core is an embedded retrieval library. It does not run an agent, generate
answers, synchronize connectors, or manage accounts. The embedding application
owns those jobs and decides which documents each caller may access.

## Entry points

- `rag_core.easy.index(folder)` opens and refreshes a persistent local folder.
- `RAGCore` binds tenant, collection, and optional document scope to an engine.
- `Engine` owns preparation, ingest, search, context packing, and deletion.
- The CLI and optional MCP entry point call the same core paths.

Use the synchronous folder interface for local tools. Use the async scoped
interface in applications. Keep an async engine and its clients on one event
loop for their lifetime.

## Ingest

A source is read within a byte limit, parsed, and split into prepared chunks.
Converters preserve PDF pages, Office slides, spreadsheet rows, and source
sections when available. Applications can inject OCR or contextualization.

The ingest fingerprint includes bytes, source key, payload metadata, models,
chunk settings, and the processing version. An unchanged document skips work
only when its published chunk count is complete. Changed processing settings
reindex the source on its next ingest.

Storage stages immutable chunk revisions, verifies completeness, then publishes
a conditional document head. Preparation and staging failures preserve the
previous revision. The local manifest records completed storage outcomes. It is
not a second ingest system or a source of truth for publication.

Folder pruning runs only after a complete scan. IDs encode tenant, collection,
and document identity separately. Ingest checks scope before reading or fetching
a source. See [dependability](dependability.md) for concurrency and recovery limits.

## Retrieval

One search runner embeds the query, retrieves candidates, optionally reranks,
and packs context. Qdrant fuses dense and BM25 sparse vectors. TurboPuffer uses
its native full-text and vector search. Both filter by tenant, collection,
metadata, and published document revision.

`Engine.search` chooses `dense`, `hybrid`, or `lexical` mode. The scoped facade
returns evidence and source locators. Prompt projections omit private paths and
storage hashes. Tracing records the effective work and budgets.

There is no query-plan language, stage graph, separate lexical database, or
provider catalogue. Extensions are ordinary Python objects implementing the
[provider protocols](../src/rag_core/search/provider_protocols.py).

## Repository layout

| Directory | Responsibility |
| --- | --- |
| `src/rag_core/` | Public facades and retrieval implementation |
| `src/rag_core/ingest/` | Folder, file, URL, and archive orchestration |
| `src/rag_core/documents/` | Converters, chunking, source locators, and OCR contracts |
| `src/rag_core/search/` | Indexing, publication, retrieval, context, and storage adapters |
| `src/rag_core/config/` | Configuration values and environment decoding |
| `src/rag_core/cli/` | Terminal argument handling and output |
| `src/rag_core/events/` | Correlated events and sinks |
| `tests/` | Contract, parser, storage, recovery, and retrieval checks |
| `examples/` | Runnable consumer examples |
| `scripts/` | Local verification and package checks |
| `docs/` | User and maintainer documentation |

Start at `easy.py`, `rag.py`, or `core.py`. Lower-level `_engine` helpers belong
to `Engine`; applications should use the public entry points.
