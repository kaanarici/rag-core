"""Stage protocols and the mutable query bundle for the retrieval pipeline.

The runner composes a linear stage list:
QueryTransform[] -> Retrieve -> Fuse -> Rerank -> Postprocess[]. Stages return
data; the runner composes them. No branching, no DSL.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from rag_core.retrieval_defaults import (
    DEFAULT_RERANK,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_USE_LEXICAL_SEARCH,
)
from rag_core.search.filters import Filter
from rag_core.search.provider_protocols import (
    EmbeddingProvider,
    RerankerProvider,
    SearchSidecar,
    SparseEmbedder,
    VectorStore,
)
from rag_core.search.request_models import RerankBudget
from rag_core.search.vector_models import SearchResult, SparseVector

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core.search.query_plan import QueryPlan

QueryPlanTraceCallback = Callable[["QueryPlan | None", int], None]


@dataclass(frozen=True)
class PipelineSidecarPrefetch:
    task: asyncio.Task[list[SearchResult]]
    started_ms: float


@dataclass
class PipelineStageState:
    collection_ensured: bool = False
    sidecar_prefetch: PipelineSidecarPrefetch | None = None


@dataclass
class PipelineQuery:
    """The mutable query bundle that flows through QueryTransform stages.

    Stages can rewrite the query string (HyDE), inject pre-computed vectors,
    or update built-in coordination state. Once Retrieve has run, downstream
    stages receive results, but the same PipelineQuery instance is still passed
    through so postprocessors can look at the resolved limit, filters, or
    typed pipeline state.
    """

    query: str
    namespace: str
    collections: list[str]
    limit: int = DEFAULT_SEARCH_LIMIT
    document_ids: Optional[list[str]] = None
    content_types: Optional[list[str]] = None
    rerank: bool = DEFAULT_RERANK
    use_lexical_search: bool = DEFAULT_USE_LEXICAL_SEARCH
    query_plan: "QueryPlan | None" = None
    query_variants: tuple[str, ...] = ()
    dense_query_text: str | None = None
    query_vector: Optional[list[float]] = None
    query_sparse_vectors: Optional[dict[str, SparseVector]] = None
    metadata_filter: Filter | None = None
    rerank_budget: RerankBudget | None = None
    query_plan_trace_callback: QueryPlanTraceCallback | None = None
    query_plan_trace_emitted: bool = False
    state: PipelineStageState = field(default_factory=PipelineStageState)

    def emit_query_plan_trace(self, query_plan: "QueryPlan | None", limit: int) -> None:
        if self.query_plan_trace_callback is None or self.query_plan_trace_emitted:
            return
        self.query_plan_trace_callback(query_plan, limit)
        self.query_plan_trace_emitted = True


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
    event_sink: "EventSink | None" = None
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
