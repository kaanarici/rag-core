"""Query-plan trace event assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.events.emit import emit_event
from rag_core.events.trace_payload_fields import TRACE_ABSENT_LABEL, TRACE_EMPTY_LABEL
from rag_core.events.types import SearchPlanned
from rag_core.retrieval_channels import (
    DENSE_RETRIEVAL_CHANNEL,
    SPARSE_RETRIEVAL_CHANNEL,
)
from rag_core.search.pipeline import RetrievalPipeline, SidecarPostprocess
from rag_core.search.planning import default_query_plan_for_store
from rag_core.search.query_plan import (
    PRIMARY_DENSE_QUERY_VECTOR,
    DenseChannel,
    Prefetch,
    QueryPlan,
)
from rag_core.search.filters import Filter
from rag_core.search.provider_protocols import VectorStore
from rag_core.search.request_models import RerankBudget

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink

QUERY_PLAN_TRACE_STORE_DEFAULT_LABEL = "store_default"


def emit_query_plan_trace_event(
    sink: "EventSink | None",
    *,
    namespace: str,
    corpus_ids: list[str],
    limit: int,
    content_types: list[str] | None,
    document_ids: list[str] | None,
    metadata_filter: Filter | None,
    rerank_budget: RerankBudget | None,
    use_lexical_search: bool,
    query_plan: QueryPlan | None,
    pipeline: RetrievalPipeline,
    store: VectorStore,
) -> None:
    plan = query_plan or default_query_plan_for_store(
        store=store,
        capabilities=store.capabilities.query_plan,
        result_limit=limit,
    )
    budget = rerank_budget or RerankBudget()
    prefetches = tuple(_flatten_prefetches(plan.prefetches)) if plan else ()
    emit_event(
        sink,
        SearchPlanned(
            namespace=namespace,
            corpus_ids=tuple(corpus_ids),
            limit=limit,
            final_limit=plan.final_limit if plan else limit,
            channels=tuple(_channel_name(prefetch) for prefetch in prefetches),
            prefetch_limits=tuple(prefetch.limit for prefetch in prefetches),
            search_profile=_plan_search_profile_name(plan),
            fusion=_plan_fusion_name(plan),
            plan_rerank=_plan_rerank_name(plan),
            boost=_plan_boost_name(plan),
            metadata_filter=_filter_name(metadata_filter),
            content_type_count=len(content_types or []),
            document_id_count=len(document_ids or []),
            rerank_candidate_count=budget.candidate_count or 0,
            rerank_timeout_ms=_timeout_ms(budget.timeout_seconds),
            rerank_max_output=budget.max_output or 0,
            rerank_fallback_on_error=budget.fallback_on_error,
            use_lexical_search=use_lexical_search and _has_sidecar_postprocess(pipeline),
            query_transforms=tuple(
                _stage_name(stage) for stage in pipeline.query_transforms
            ),
            retrieve_stage=_stage_name(pipeline.retrieve),
            fuse_stage=_stage_name(pipeline.fuse),
            rerank_stage=_stage_name(pipeline.rerank),
            postprocesses=tuple(_stage_name(stage) for stage in pipeline.postprocesses),
        ),
    )


def _flatten_prefetches(prefetches: tuple[Prefetch, ...]) -> list[Prefetch]:
    flattened: list[Prefetch] = []
    for prefetch in prefetches:
        flattened.append(prefetch)
        flattened.extend(_flatten_prefetches(prefetch.nested))
    return flattened


def _channel_name(prefetch: Prefetch) -> str:
    channel = prefetch.channel
    if isinstance(channel, DenseChannel):
        query_vector = channel.using_query_vector or PRIMARY_DENSE_QUERY_VECTOR
        field = channel.vector_field or DENSE_RETRIEVAL_CHANNEL
        return f"{DENSE_RETRIEVAL_CHANNEL}:{field}:{query_vector}"
    return (
        f"{SPARSE_RETRIEVAL_CHANNEL}:"
        f"{channel.vector_field}:{channel.using_query_vector}"
    )


def _plan_fusion_name(plan: QueryPlan | None) -> str:
    if plan is None:
        return QUERY_PLAN_TRACE_STORE_DEFAULT_LABEL
    if plan.fuse is None:
        return TRACE_ABSENT_LABEL
    return plan.fuse.kind


def _plan_search_profile_name(plan: QueryPlan | None) -> str:
    if plan is None or plan.search_profile is None:
        return TRACE_EMPTY_LABEL
    return plan.search_profile


def _plan_rerank_name(plan: QueryPlan | None) -> str:
    if plan is None or plan.rerank is None:
        return TRACE_ABSENT_LABEL
    return type(plan.rerank).__name__.lower()


def _plan_boost_name(plan: QueryPlan | None) -> str:
    if plan is None or plan.boost is None:
        return TRACE_ABSENT_LABEL
    return plan.boost.kind


def _filter_name(filter_value: Filter | None) -> str:
    if filter_value is None:
        return TRACE_ABSENT_LABEL
    return type(filter_value).__name__


def _timeout_ms(timeout_seconds: float | None) -> float:
    if timeout_seconds is None:
        return 0.0
    return timeout_seconds * 1000.0


def _stage_name(stage: object) -> str:
    return type(stage).__name__


def _has_sidecar_postprocess(pipeline: RetrievalPipeline) -> bool:
    return any(
        isinstance(stage, SidecarPostprocess) for stage in pipeline.postprocesses
    )
