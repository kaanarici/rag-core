"""Unified search pipeline runner. Drives the linear retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional, cast
from uuid import uuid4

from rag_core.events.emit import emit_event, now_ms
from rag_core.events.search_events import RETURNED_DOCUMENT_IDS_CAP
from rag_core.events.trace_payload_fields import SEARCH_ERROR_STAGE
from rag_core.events.types import (
    AuditContext,
    SearchCompleted,
    SearchStarted,
    StageError,
)
from rag_core.retrieval_defaults import (
    DEFAULT_RERANK,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_USE_LEXICAL_SEARCH,
)
from rag_core.search.pipeline import PipelineContext, RetrievalPipeline
from rag_core.search.pipeline.stages.hybrid_retrieve import HybridRetrieve
from rag_core.search.planning import QueryPlanPreparer
from rag_core.search.policy import CollectionPolicy
from rag_core.search.request_models import (
    RerankBudget,
    _require_non_blank_string,
    _require_non_blank_string_items,
    _require_positive_int,
)
from rag_core.search.vector_models import (
    SearchResult,
    SparseVector,
    _validate_dense_vector,
)
from rag_core.search.query_plan_trace import emit_query_plan_trace_event
from rag_core.search.pipeline_runner_defaults import (
    default_search_pipeline,
    pipeline_query_from_search_request,
    use_lexical_search_for_request,
)
from rag_core.search.filters import Filter
from rag_core.search.provider_protocols import (
    EmbeddingProvider,
    RerankerProvider,
    SearchSidecar,
    SparseEmbedder,
    VectorStore,
)

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core.events.types import Event
    from rag_core.search.pipeline import PipelineQuery
    from rag_core.search.query_plan import QueryPlan


@dataclass(frozen=True)
class SearchExecutionOptions:
    """Advanced runner controls outside normal retrieval intent."""

    query_vector: list[float] | None = None
    query_sparse_vectors: dict[str, SparseVector] | None = None
    use_lexical_search: bool = DEFAULT_USE_LEXICAL_SEARCH
    query_plan: "QueryPlan | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.use_lexical_search, bool):
            raise ValueError(
                "SearchExecutionOptions.use_lexical_search must be a boolean"
            )
        if self.query_vector is not None:
            _validate_dense_vector(
                self.query_vector,
                "SearchExecutionOptions.query_vector",
            )
        if self.query_sparse_vectors is not None:
            _validate_query_sparse_vectors(self.query_sparse_vectors)


@dataclass(frozen=True)
class SearchRequest:
    """Engine-level retrieval intent before pipeline planning and provider calls."""

    query: str
    collections: list[str]
    namespace: str
    limit: int = DEFAULT_SEARCH_LIMIT
    content_types: Optional[list[str]] = None
    document_ids: Optional[list[str]] = None
    rerank: bool = DEFAULT_RERANK
    metadata_filter: Filter | None = None
    rerank_budget: RerankBudget | None = None
    execution: SearchExecutionOptions = field(default_factory=SearchExecutionOptions)
    # Caller-supplied correlation. ``search_id`` is still minted internally
    # by ``search_with_trace`` so existing consumers keep working; the other
    # fields (actor, request_id, ingest_id) are pass-through audit context
    # the gateway already authenticated.
    audit_context: AuditContext | None = None

    def __post_init__(self) -> None:
        _require_non_blank_string(self.query, "SearchRequest.query")
        _require_non_blank_string(self.namespace, "SearchRequest.namespace")
        _require_non_blank_string_items(self.collections, "SearchRequest.collections")
        _require_non_blank_string_items(self.content_types, "SearchRequest.content_types")
        _require_non_blank_string_items(self.document_ids, "SearchRequest.document_ids")
        _require_positive_int(self.limit, "SearchRequest.limit")


@dataclass(frozen=True)
class SearchRunResult:
    results: list[SearchResult]
    search_id: str


def _validate_query_sparse_vectors(value: object) -> None:
    message = (
        "SearchExecutionOptions.query_sparse_vectors must map non-empty "
        "channel names to SparseVector values"
    )
    if not isinstance(value, dict):
        raise ValueError(message)
    for channel_name, sparse_vector in value.items():
        if (
            not isinstance(channel_name, str)
            or not channel_name.strip()
            or not isinstance(sparse_vector, SparseVector)
        ):
            raise ValueError(message)


class SearchPipelineRunner:
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
        collection_policy: CollectionPolicy | None = None,
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
        self._collection_policy = collection_policy

    async def search(self, req: SearchRequest) -> list[SearchResult]:
        """Execute unified search by running the configured pipeline."""
        return (await self.search_with_trace(req)).results

    async def search_with_trace(self, req: SearchRequest) -> SearchRunResult:
        """Execute unified search and return its correlation identifier."""
        search_id = uuid4().hex
        sink = (
            _SearchCorrelationSink(self._event_sink, search_id, req.audit_context)
            if self._event_sink
            else None
        )
        effective_use_lexical_search = (
            use_lexical_search_for_request(req)
            if self._uses_default_pipeline
            else req.execution.use_lexical_search
        )
        if self._collection_policy is not None:
            # Tier fence: refuse cross-namespace, cross-collection, and disallowed
            # capability requests at the seam before any provider call so a
            # restricted-tier process cannot leak through wider tiers' code paths.
            # An explicit caller-supplied query plan is checked by its
            # search-profile name; internally resolved default plans follow the
            # store's declared capabilities.
            explicit_plan = req.execution.query_plan
            self._collection_policy.validate_search(
                namespace=req.namespace,
                collections=req.collections,
                rerank=req.rerank,
                use_lexical_search=effective_use_lexical_search,
                query_plan_preset=(
                    explicit_plan.search_profile if explicit_plan is not None else None
                ),
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
                use_lexical_search=effective_use_lexical_search,
            )
            emit_event(
                sink,
                SearchStarted(
                    namespace=req.namespace,
                    collections=tuple(req.collections),
                    query_length=len(req.query),
                    limit=pipeline_query.limit,
                ),
            )

            def emit_resolved_plan(query_plan: "QueryPlan | None", limit: int) -> None:
                emit_query_plan_trace_event(
                    sink,
                    namespace=req.namespace,
                    collections=req.collections,
                    limit=limit,
                    content_types=req.content_types,
                    document_ids=req.document_ids,
                    metadata_filter=req.metadata_filter,
                    rerank_budget=req.rerank_budget,
                    use_lexical_search=effective_use_lexical_search,
                    query_plan=query_plan,
                    pipeline=self._pipeline,
                    store=self._store,
                )

            if has_empty_allowlist(req):
                emit_resolved_plan(req.execution.query_plan, pipeline_query.limit)
                emit_event(
                    sink,
                    SearchCompleted(
                        namespace=req.namespace,
                        result_count=0,
                        requested_rerank=req.rerank,
                        requested_sidecar=effective_use_lexical_search,
                        duration_ms=now_ms() - started_ms,
                        collections=tuple(req.collections),
                        returned_document_ids=(),
                    ),
                )
                return SearchRunResult(results=[], search_id=search_id)

            if req.execution.query_plan is None and isinstance(
                self._store, QueryPlanPreparer
            ):
                await self._store.ensure_collection()
                pipeline_query.state.collection_ensured = True
            ctx = PipelineContext(
                embedding_provider=self._embedding,
                sparse_embedder=self._sparse,
                vector_store=self._store,
                reranker=self._reranker,
                sidecar=self._sidecar,
                event_sink=sink,
            )
            pipeline_query.query_plan_trace_callback = emit_resolved_plan
            # Canonicalize the result limit against an explicit caller plan
            # here, at the request seam. Stages and trace emitters must not
            # change retrieval semantics as a side effect.
            if req.execution.query_plan is not None:
                pipeline_query.limit = req.execution.query_plan.final_limit
            if isinstance(self._pipeline.retrieve, HybridRetrieve):
                pipeline = self._pipeline
            else:
                # Custom pipelines: after user transforms (which may inject a
                # plan), canonicalize the limit, then emit the resolved-plan
                # trace. Two transforms so the trace emitter stays pure.
                pipeline = replace(
                    self._pipeline,
                    query_transforms=(
                        *self._pipeline.query_transforms,
                        _CanonicalizePlanLimitTransform(),
                        _EmitQueryPlanTraceTransform(),
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
                StageError(stage=SEARCH_ERROR_STAGE, error_type=type(exc).__name__),
            )
            emit_event(
                sink,
                SearchCompleted(
                    namespace=req.namespace,
                    result_count=len(results),
                    requested_rerank=req.rerank,
                    requested_sidecar=effective_use_lexical_search,
                    attempted_rerank=attempted_rerank,
                    attempted_sidecar=attempted_sidecar,
                    applied_rerank=applied_rerank,
                    applied_sidecar=applied_sidecar,
                    succeeded=False,
                    duration_ms=now_ms() - started_ms,
                    collections=tuple(req.collections),
                    returned_document_ids=_capped_returned_document_ids(results),
                ),
            )
            raise
        emit_event(
            sink,
            SearchCompleted(
                namespace=req.namespace,
                result_count=len(results),
                requested_rerank=req.rerank,
                requested_sidecar=effective_use_lexical_search,
                attempted_rerank=attempted_rerank,
                attempted_sidecar=attempted_sidecar,
                applied_rerank=applied_rerank,
                applied_sidecar=applied_sidecar,
                duration_ms=now_ms() - started_ms,
                collections=tuple(req.collections),
                returned_document_ids=_capped_returned_document_ids(results),
            ),
        )
        return SearchRunResult(results=results, search_id=search_id)


def _capped_returned_document_ids(results: list[SearchResult]) -> tuple[str, ...]:
    # Distinct document_ids, in result order, up to RETURNED_DOCUMENT_IDS_CAP.
    # Skips blanks so providers that don't carry document_id don't fill the
    # audit row with empty strings.
    seen: list[str] = []
    seen_set: set[str] = set()
    for hit in results:
        document_id = getattr(hit, "document_id", "") or ""
        if not document_id or document_id in seen_set:
            continue
        seen.append(document_id)
        seen_set.add(document_id)
        if len(seen) >= RETURNED_DOCUMENT_IDS_CAP:
            break
    return tuple(seen)


def has_empty_allowlist(req: SearchRequest) -> bool:
    return (
        (req.document_ids is not None and not req.document_ids)
        or (req.collections is not None and not req.collections)
        or (req.content_types is not None and not req.content_types)
    )


class _SearchCorrelationSink:
    """Stamp ``search_id`` and caller-supplied audit context onto every event.

    The runner mints ``search_id`` (no external caller can pass one. That
    keeps the trace identifier unforgeable). ``actor``, ``request_id``, and
    ``ingest_id`` come from ``AuditContext`` (typically populated from
    gateway headers). All four are written via ``dataclasses.replace`` so
    the underlying event is still a frozen dataclass.
    """

    def __init__(
        self,
        sink: "EventSink",
        search_id: str,
        audit_context: AuditContext | None,
    ) -> None:
        self._sink = sink
        self._search_id = search_id
        self._audit_context = audit_context

    def emit(self, event: "Event") -> None:
        updates: dict[str, Any] = {}
        if hasattr(event, "search_id"):
            updates["search_id"] = self._search_id
        ctx = self._audit_context
        if ctx is not None:
            if ctx.actor and hasattr(event, "actor") and not getattr(event, "actor"):
                updates["actor"] = ctx.actor
            if (
                ctx.request_id
                and hasattr(event, "request_id")
                and not getattr(event, "request_id")
            ):
                updates["request_id"] = ctx.request_id
            if (
                ctx.ingest_id
                and hasattr(event, "ingest_id")
                and not getattr(event, "ingest_id")
            ):
                updates["ingest_id"] = ctx.ingest_id
        if updates:
            event = cast("Event", replace(cast(Any, event), **updates))
        self._sink.emit(event)


class _CanonicalizePlanLimitTransform:
    """Align ``query.limit`` with a transform-injected plan's ``final_limit``."""

    async def transform(
        self,
        query: "PipelineQuery",
        ctx: PipelineContext,
    ) -> "PipelineQuery":
        del ctx
        if query.query_plan is not None:
            query.limit = query.query_plan.final_limit
        return query


class _EmitQueryPlanTraceTransform:
    """Trace-only transform for custom pipelines; must not mutate the query."""

    async def transform(
        self,
        query: "PipelineQuery",
        ctx: PipelineContext,
    ) -> "PipelineQuery":
        del ctx
        query.emit_query_plan_trace(query.query_plan, query.limit)
        return query
