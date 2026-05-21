# No platform, provider, or model lock-in; first-party options must be real

## Status

accepted (2026-05-01); updated (2026-05-17)

## Context

`rag-core` ships opinionated defaults: Qdrant for storage, OpenAI for dense embeddings, FastEmbed BM25 for sparse, optional Cohere/Voyage rerankers. These are the recommended on-ramp. They must not become the only path.

The updated product direction is vendor-flexible but first-party opinionated. Users should be able to choose strong infrastructure without building the retrieval glue themselves. That means the project must not hide behind abstract protocols when the shipped adapter quality is uneven.

## Decision

No platform, provider, or model lock-in is a first-class architectural property of the engine. Name-resolved external dependency categories have typed protocols with registries; users register custom adapters from outside the package without forking. Direct injection remains valid when config lookup would add a shallow interface. Defaults are recommendations, not requirements.

First-party support is a product commitment, not just a registered factory. A first-party provider must have typed config, tests or env-gated integration smokes, docs, examples, `doctor` diagnostics where relevant, failure messages, and limit/performance notes.

The beta categories that must each have a registry or dedicated registry loader: `EmbeddingProvider`, `SparseEmbedder`, `RerankerProvider`, `OcrProvider`, `VectorStore`, `SearchSidecar`, `EmbeddingCache`, `ChunkContextCache`, `ChunkingStrategy`, `Converter`.

Direct-injection categories that remain explicit constructor inputs: `ChunkContextualizer` and `EventSink`.

Provider support levels:

- `default`: the path used by quickstarts and expected to work for new users.
- `first-party`: supported by docs, tests, examples, diagnostics, and maintenance.
- `experimental`: usable but explicitly incomplete or changing.
- `community`: externally maintained or user-provided, not promised by core docs.

## Why

The project is meant to be a dependable retrieval engine, and vendor lock-in works against that. Users must be able to start with the defaults, swap components as their needs grow, and avoid forks for normal provider changes.

This is the principle behind ADR-0001 and the provider contracts generally.

## Consequences

- Name-resolved provider categories get a uniform registry pattern with `register(name, factory)` for users.
- Built-in adapters register themselves on package import in a way that does not pull in optional deps.
- Config strings (`embedding_provider="openai"`) become lookup keys in the registry; the registry validates and instantiates.
- Adding a new provider externally is a 10-30 line adapter file plus one `register()` call, not a fork.
- Tests use lightweight fakes registered the same way real adapters are.
- Documentation must not list a provider as first-party unless it meets the first-party support bar.
- Default provider choices can change only with migration notes and validation coverage.
