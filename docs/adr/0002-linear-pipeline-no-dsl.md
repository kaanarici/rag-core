# Retrieval pipeline is a linear ordered list of stages, not a DSL

## Status

accepted (2026-05-01)

## Context

The retrieval pipeline needs to be composable so advanced and experimental setups (HyDE, multi-query, MMR diversity, reranker cascades, parent-child expansion, query routing, custom postprocessing) plug in as adapters rather than edits to the orchestrator. A graph DSL would add Haystack-style branches, conditional execution, retry policies, and configuration files describing the pipeline shape.

## Decision

The pipeline is a linear ordered list of stages, executed in order, each with one method. Categories are fixed: `QueryTransform`, `Retrieve`, `Fuse`, `Rerank`, `Postprocess`. Stages return data; the runner composes them. No branching, no conditionals, no retry semantics, no declarative pipeline files, no graph runtime.

## Why

`rag-core` is an engine, not a workflow runtime. A graph DSL would put the engine into orchestration scope. Most retrieval pipelines are linear; unusual flows can compose two pipelines in user code instead of adding runtime concepts.

## Consequences

- A user who wants conditional retrieval ("if HyDE confidence < threshold, fall back to plain retrieve") composes two pipelines and chooses between them in their own code. The engine does not provide a switch stage.
- A user who wants retries wraps a stage adapter in their own retry adapter. The engine does not provide retry as a runtime feature.
- The pipeline runner stays a small, linear composition boundary: fixed stage categories, no graph DSL, no declarative runtime, and immutable pipeline definitions.

## Considered alternatives

- **Haystack/LangChain-style graph DSL.** Rejected: orchestration scope, large interface.
- **Single `search()` function with kwargs.** Rejected: every new technique forces an edit to the searcher.
