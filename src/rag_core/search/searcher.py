"""Unified search orchestrator. Drives the linear retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Optional, cast
from uuid import uuid4

from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import (
    SearchCompleted,
    SearchStarted,
    StageError,
)
from rag_core.search.pipeline import PipelineContext, RetrievalPipeline
from rag_core.search.pipeline.stages.hybrid_retrieve import (
    COLLECTION_ENSURED_EXTRA_KEY,
    HybridRetrieve,
)
from rag_core.search.planning import QueryPlanPreparer
from rag_core.search.request_models import (
    _require_non_blank_string,
    _require_positive_int,
)
from rag_core.search.search_plan_trace import emit_search_planned
from rag_core.search.searcher_pipeline import (
    default_search_pipeline,
    pipeline_query_from_search_request,
    use_sidecar_for_request,
)
from rag_core.search.types import (
    EmbeddingProvider,
    Filter,
    RerankBudget,
    RerankerProvider,
    SearchSidecar,
    SearchResult,
    SparseEmbedder,
    SparseVector,
    VectorStore,
)

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core.events.types import Event
    from rag_core.search.pipeline import PipelineQuery
    from rag_core.search.query_plan import QueryPlan


@dataclass(frozen=True)
class SearchRequest:
    """Normal retrieval intent for a unified search."""

    query: str
    corpus_ids: list[str]
    namespace: str
    limit: int = 20
    content_types: Optional[list[str]] = None
    document_ids: Optional[list[str]] = None
    rerank: bool = False
    query_vector: list[float] | None = None
    query_sparse_vectors: dict[str, SparseVector] | None = None
    use_sidecar: bool = True
    query_plan: "QueryPlan | None" = None
    metadata_filter: Filter | None = None
    rerank_budget: RerankBudget | None = None

    def __post_init__(self) -> None:
        _require_non_blank_string(self.namespace, "SearchRequest.namespace")
        _require_positive_int(self.limit, "SearchRequest.limit")


@dataclass(frozen=True)
class SearchRunResult:
    results: list[SearchResult]
    search_id: str


class SearchOrchestrator:
    """Drives a linear `RetrievalPipeline` with an injected provider context."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        sparse_embedder: SparseEmbedder,
        vector_store: VectorStore,
        reranker: Optional[RerankerProvider] = None,
        sidecar: SearchSidecar | None = None,
        event_sink: "EventSink | None" = None,
        *,
        pipeline: RetrievalPipeline | None = None,
    ) -> None:
        self._embedding = embedding_provider
        self._sparse = sparse_embedder
        self._store = vector_store
        self._reranker = reranker
        self._sidecar = sidecar
        self._event_sink = event_sink
        self._uses_default_pipeline = pipeline is None
        self._pipeline = pipeline or default_search_pipeline(
            reranker_present=reranker is not None,
            sidecar_present=sidecar is not None,
        )

    async def search(self, req: SearchRequest) -> list[SearchResult]:
        """Execute unified search by running the configured pipeline."""
        return (await self.search_with_trace(req)).results

    async def search_with_trace(self, req: SearchRequest) -> SearchRunResult:
        """Execute unified search and return its correlation identifier."""
        search_id = uuid4().hex
        sink = _SearchCorrelationSink(self._event_sink, search_id) if self._event_sink else None
        effective_use_sidecar = (
            use_sidecar_for_request(req) if self._uses_default_pipeline else req.use_sidecar
        )
        started_ms = now_ms()
        results: list[SearchResult] = []
        attempted_rerank = False
        attempted_sidecar = False
        applied_rerank = False
        applied_sidecar = False
        try:
            pipeline_query = pipeline_query_from_search_request(
                req,
                use_sidecar=effective_use_sidecar,
            )
            emit_event(
                sink,
                SearchStarted(
                    namespace=req.namespace,
                    corpus_ids=tuple(req.corpus_ids),
                    query_length=len(req.query),
                    limit=pipeline_query.limit,
                ),
            )

            def emit_resolved_plan(query_plan: "QueryPlan | None", limit: int) -> None:
                emit_search_planned(
                    sink,
                    namespace=req.namespace,
                    corpus_ids=req.corpus_ids,
                    limit=limit,
                    content_types=req.content_types,
                    document_ids=req.document_ids,
                    metadata_filter=req.metadata_filter,
                    rerank_budget=req.rerank_budget,
                    use_sidecar=effective_use_sidecar,
                    query_plan=query_plan,
                    pipeline=self._pipeline,
                    store=self._store,
                )

            if has_empty_allowlist(req):
                emit_resolved_plan(req.query_plan, pipeline_query.limit)
                emit_event(
                    sink,
                    SearchCompleted(
                        namespace=req.namespace,
                        result_count=0,
                        requested_rerank=req.rerank,
                        requested_sidecar=req.use_sidecar,
                        duration_ms=now_ms() - started_ms,
                    ),
                )
                return SearchRunResult(results=[], search_id=search_id)

            if req.query_plan is None and isinstance(self._store, QueryPlanPreparer):
                await self._store.ensure_collection()
                pipeline_query.extra[COLLECTION_ENSURED_EXTRA_KEY] = True
            ctx = PipelineContext(
                embedding_provider=self._embedding,
                sparse_embedder=self._sparse,
                vector_store=self._store,
                reranker=self._reranker,
                sidecar=self._sidecar,
                event_sink=sink,
            )
            pipeline_query.search_plan_callback = emit_resolved_plan
            if isinstance(self._pipeline.retrieve, HybridRetrieve):
                pipeline = self._pipeline
            else:
                pipeline = replace(
                    self._pipeline,
                    query_transforms=(
                        *self._pipeline.query_transforms,
                        _EmitSearchPlannedTransform(),
                    ),
                )
            results = await pipeline.run(pipeline_query, ctx)
            attempted_rerank = ctx.execution.attempted_rerank
            attempted_sidecar = ctx.execution.attempted_sidecar
            applied_rerank = ctx.execution.applied_rerank
            applied_sidecar = ctx.execution.applied_sidecar
        except Exception as exc:
            emit_event(
                sink,
                StageError(stage="search", error_type=type(exc).__name__),
            )
            emit_event(
                sink,
                SearchCompleted(
                    namespace=req.namespace,
                    result_count=len(results),
                    requested_rerank=req.rerank,
                    requested_sidecar=req.use_sidecar,
                    attempted_rerank=attempted_rerank,
                    attempted_sidecar=attempted_sidecar,
                    applied_rerank=applied_rerank,
                    applied_sidecar=applied_sidecar,
                    succeeded=False,
                    duration_ms=now_ms() - started_ms,
                ),
            )
            raise
        emit_event(
            sink,
            SearchCompleted(
                namespace=req.namespace,
                result_count=len(results),
                used_rerank=applied_rerank,
                used_sidecar=applied_sidecar,
                requested_rerank=req.rerank,
                requested_sidecar=req.use_sidecar,
                attempted_rerank=attempted_rerank,
                attempted_sidecar=attempted_sidecar,
                applied_rerank=applied_rerank,
                applied_sidecar=applied_sidecar,
                duration_ms=now_ms() - started_ms,
            ),
        )
        return SearchRunResult(results=results, search_id=search_id)


def has_empty_allowlist(req: SearchRequest) -> bool:
    return (
        (req.document_ids is not None and not req.document_ids)
        or (req.corpus_ids is not None and not req.corpus_ids)
        or (req.content_types is not None and not req.content_types)
    )


class _SearchCorrelationSink:
    def __init__(self, sink: "EventSink", search_id: str) -> None:
        self._sink = sink
        self._search_id = search_id

    def emit(self, event: "Event") -> None:
        if hasattr(event, "search_id"):
            event = cast("Event", replace(cast(Any, event), search_id=self._search_id))
        self._sink.emit(event)


class _EmitSearchPlannedTransform:
    async def transform(
        self,
        query: "PipelineQuery",
        ctx: PipelineContext,
    ) -> "PipelineQuery":
        del ctx
        if query.query_plan is not None:
            query.limit = query.query_plan.final_limit
        query.emit_search_plan(query.query_plan, query.limit)
        return query
