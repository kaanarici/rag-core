# VectorStore protocol stays vendor-flexible; Qdrant is the default and TurboPuffer is first-party

## Status

accepted (2026-05-01); updated (2026-05-17)

## Context

`rag-core` defines a `VectorStore` protocol and ships Qdrant as the primary real adapter. The original risk was that Qdrant assumptions could leak across the protocol surface: payload field names (`namespace`, `corpus_id`, `document_id`, `chunk_index`), composite point-ID format, search-filter shape, and query translation behavior.

A "delete the protocol and embrace Qdrant" path was considered. It would simplify internal code and expose Qdrant-specific query features such as server-side MMR, score formulas, and weighted RRF without translation.

The updated product direction keeps Qdrant as the accessible default because it is easy to run locally and self-host without a minimum hosted spend. TurboPuffer is a repo-maintained managed vector-store option for users who want that operating model and accept its cost model.

## Decision

The `VectorStore` protocol stays. Qdrant remains the default local, development, and self-hosted adapter. TurboPuffer is the first-party managed vector-store option. Backend-specific features live behind optional capability interfaces that adapters opt into; the base protocol stays usable on backends that do not have those features.

The protocol is vendor-flexible, not fake-neutral. First-party adapters must be tested, documented, diagnosed by `doctor`, and represented honestly in support matrices. Community or experimental adapters must not be described like first-party surfaces.

## Why

No platform lock-in is a first-class architectural property of `rag-core`. A user choosing this engine must be able to substitute the vector store without forking the engine. Recommending Qdrant and supporting TurboPuffer are both fine; requiring either one is not.

This is hard to reverse: deleting the protocol removes the main extension point for non-Qdrant backends. Restoring it after removal would reshape every caller.

The stronger standard is first-party quality. A vector-store option is not truly supported until it has contract tests, docs, runtime diagnostics, migration guidance, failure behavior, and performance notes.

## Consequences

- Qdrant stays the default and must remain excellent for local and self-hosted usage.
- TurboPuffer is implemented as a first-party adapter, not left as a user exercise.
- Qdrant and TurboPuffer may expose different rich capabilities, but the base contract must remain coherent.
- The protocol must be validated by real adapters; an in-memory adapter for tests is not enough.
- Payload field names, point-ID format, and filter shape become injectable via a `VectorStorePolicy` rather than hardcoded.
- Documentation must distinguish default, first-party, experimental, and community support levels.

## Considered alternatives

- **Embrace Qdrant; delete the protocol.** Rejected: violates no-lock-in.
- **Keep the protocol but never validate it.** Rejected: drifts to Qdrant-shaped behavior over time and traps users.
- **Claim all vector stores are equally supported.** Rejected: misleading and worse for users than an honest first-party support matrix.

## v0 addendum (2026-05-20)

The default wheel ships **Qdrant** on the default CLI and doctor path. **TurboPuffer** is a first-party **optional** adapter (`uv sync --extra turbopuffer`): contract tests, doctor diagnostics, dense + hybrid RRF + SparseKNN with honest unsupported-stage errors. The `VectorStore` protocol and registry are unchanged; only install surface and docs distinguish default vs optional backends.
