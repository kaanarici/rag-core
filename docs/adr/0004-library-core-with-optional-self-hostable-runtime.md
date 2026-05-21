# Library core with optional self-hostable runtime

## Status

accepted (2026-05-17)

## Context

`rag-core` started as an embeddable retrieval library: applications own chat, auth, request routing, UI, and model calls, while `rag-core` owns parsing, chunking, indexing, retrieval, reranking orchestration, scoring, manifests, events, and context handoff.

That center is still correct, and self-hostability is in scope. The design should let users run a private retrieval backend with an API and worker without giving up the library-first architecture or adopting a hosted control plane.

The risk is platform drift. A self-hostable runtime can easily pull the repo toward auth, billing, teams, admin UI, connector marketplaces, job orchestration, and hosted-product assumptions before the retrieval engine itself is strong enough.

## Decision

`rag-core` remains library-first. An optional self-hostable runtime is allowed, but it must be a thin layer over the same core contracts used by embedded library users.

The architecture has three layers:

1. Core library: parsing, conversion, chunking, ingest, indexing, search planning, reranking orchestration, context packs, provider protocols, events, evals, and manifest/state primitives.
2. Optional self-hostable runtime: API, worker entrypoints, health checks, job status, config loading, and deployment examples that call the core library.
3. Product wrappers: managed hosting, hosted accounts, team features, billing, admin UI, hosted connectors, and dashboards. This layer stays outside the engine until there is a separate product decision.

The runtime must not become the source of truth for retrieval behavior. Library mode and runtime mode should share the same parser, indexer, search, provider, event, and eval contracts.

## Why

Self-hostability lets teams keep data inside their own boundary and deploy a retrieval backend without writing the operational shell themselves.

The core must still remain usable as a package because many applications need to embed retrieval directly into their own process. If the server becomes mandatory, `rag-core` stops being a core retrieval engine and becomes a platform.

Keeping runtime optional also protects dependency weight. Library users should not install web-server, worker, object-storage, or deployment dependencies unless they ask for runtime mode.

## Consequences

- Runtime dependencies must be optional extras or isolated modules.
- `import rag_core` must stay cheap and must not import runtime dependencies.
- If `rag-core serve` is added, it must call public core APIs instead of reimplementing pipeline behavior.
- Runtime endpoints should be minimal at first: health, runtime description, ingest, job status, search, and retrieve-context. Eval HTTP stays out of v1; apps use `rag_core.evals` or `examples/retrieval_eval.py`.
- Auth must be externally pluggable or minimal until a separate product layer exists.
- No team, billing, hosted account, admin UI, or connector marketplace concepts belong in the core runtime.
- Runtime state should use explicit contracts such as job backend, corpus state store, blob store, and trace writer only when JSONL manifests and direct library calls are insufficient.

## Considered alternatives

- **Library only.** Rejected: blocks private/self-hosted deployment use cases and leaves too much operational glue to every user.
- **Hosted platform first.** Rejected: creates product and operational scope before the retrieval engine is strong enough.
- **Server as the primary API.** Rejected: makes embedded use worse and turns optional deployment concerns into core dependencies.
