"""Stage protocols and the mutable query bundle for the retrieval pipeline.

The pipeline is the linear list of stages described in `docs/adr/0002-linear-pipeline-no-dsl.md`:
QueryTransform[] -> Retrieve -> Fuse -> Rerank -> Postprocess[]. Stages return data;
the runner composes them. No branching, no DSL.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from rag_core.search.types import (
    EmbeddingProvider,
    Filter,
    RerankBudget,
    RerankerProvider,
    SearchResult,
    SearchSidecar,
    SparseEmbedder,
    SparseVector,
    VectorStore,
)

if TYPE_CHECKING:
    from rag_core.search.query_plan import QueryPlan

SearchPlanCallback = Callable[["QueryPlan | None", int], None]


@dataclass
class PipelineQuery:
    """The mutable query bundle that flows through QueryTransform stages.

    Stages can rewrite the query string (HyDE), inject pre-computed vectors,
    or stash adjuncts in `extra` (e.g. an awaitable for a parallel sidecar
    fetch). Once Retrieve has run, downstream stages receive results, but the
    same PipelineQuery instance is still passed through so postprocessors can
    look at the resolved limit, filters, or extras.
    """

    query: str
    namespace: str
    corpus_ids: list[str]
    limit: int = 20
    document_ids: Optional[list[str]] = None
    content_types: Optional[list[str]] = None
    rerank: bool = False
    use_sidecar: bool = True
    query_plan: "QueryPlan | None" = None
    query_vector: Optional[list[float]] = None
    query_sparse_vectors: Optional[dict[str, SparseVector]] = None
    metadata_filter: Filter | None = None
    rerank_budget: RerankBudget | None = None
    search_plan_callback: SearchPlanCallback | None = None
    search_plan_emitted: bool = False
    extra: dict[str, object] = field(default_factory=dict)

    def emit_search_plan(self, query_plan: "QueryPlan | None", limit: int) -> None:
        if self.search_plan_callback is None or self.search_plan_emitted:
            return
        self.search_plan_callback(query_plan, limit)
        self.search_plan_emitted = True


@dataclass
class PipelineExecutionSummary:
    attempted_rerank: bool = False
    applied_rerank: bool = False
    attempted_sidecar: bool = False
    applied_sidecar: bool = False


@dataclass(frozen=True)
class PipelineContext:
    """Shared read-only context for stages: providers, store, sidecar, sink.

    Stages emit structured diagnostics to `event_sink` when one is supplied and
    otherwise ignore it.
    """

    embedding_provider: EmbeddingProvider
    sparse_embedder: SparseEmbedder
    vector_store: VectorStore
    reranker: RerankerProvider | None = None
    sidecar: SearchSidecar | None = None
    event_sink: object | None = None
    execution: PipelineExecutionSummary = field(default_factory=PipelineExecutionSummary)


@runtime_checkable
class QueryTransform(Protocol):
    """Pre-retrieval mutation: rewrite the query, expand it, or add hints."""

    async def transform(
        self, query: PipelineQuery, ctx: PipelineContext
    ) -> PipelineQuery: ...


@runtime_checkable
class Retrieve(Protocol):
    """Run the actual hybrid (or other) retrieval against the vector store."""

    async def retrieve(
        self, query: PipelineQuery, ctx: PipelineContext
    ) -> list[SearchResult]: ...


@runtime_checkable
class FuseStage(Protocol):
    """Combine multiple candidate lists into one ranked list.

    The default Retrieve already produces one fused list (Qdrant's server-side
    RRF), so the default FuseStage is identity over `[results]`. Multi-retriever
    pipelines (RAG-Fusion) supply a real fuser that merges parallel result lists.
    """

    async def fuse(
        self,
        results: list[list[SearchResult]],
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> list[SearchResult]: ...


@runtime_checkable
class Rerank(Protocol):
    """Re-order candidates with a stronger model."""

    async def rerank(
        self,
        results: list[SearchResult],
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> list[SearchResult]: ...


@runtime_checkable
class Postprocess(Protocol):
    """Final-stage transforms: sidecar merge, parent-expand, MMR, score boost."""

    async def postprocess(
        self,
        results: list[SearchResult],
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> list[SearchResult]: ...
