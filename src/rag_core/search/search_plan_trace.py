"""Search-plan trace event assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.events.emit import emit_event
from rag_core.events.types import SearchPlanned
from rag_core.search.pipeline import RetrievalPipeline, SidecarPostprocess
from rag_core.search.planning import default_query_plan_for_store
from rag_core.search.query_plan import DenseChannel, Prefetch, QueryPlan
from rag_core.search.types import Filter, RerankBudget, VectorStore

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


def emit_search_planned(
    sink: "EventSink | None",
    *,
    namespace: str,
    corpus_ids: list[str],
    limit: int,
    content_types: list[str] | None,
    document_ids: list[str] | None,
    metadata_filter: Filter | None,
    rerank_budget: RerankBudget | None,
    use_sidecar: bool,
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
            use_sidecar=use_sidecar and _has_sidecar_postprocess(pipeline),
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
        query_vector = channel.using_query_vector or "primary"
        field = channel.vector_field or "dense"
        return f"dense:{field}:{query_vector}"
    return f"sparse:{channel.vector_field}:{channel.using_query_vector}"


def _plan_fusion_name(plan: QueryPlan | None) -> str:
    if plan is None:
        return "store_default"
    if plan.fuse is None:
        return "none"
    return plan.fuse.kind


def _plan_rerank_name(plan: QueryPlan | None) -> str:
    if plan is None or plan.rerank is None:
        return "none"
    return type(plan.rerank).__name__.lower()


def _plan_boost_name(plan: QueryPlan | None) -> str:
    if plan is None or plan.boost is None:
        return "none"
    return plan.boost.kind


def _filter_name(filter_value: Filter | None) -> str:
    if filter_value is None:
        return "none"
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
