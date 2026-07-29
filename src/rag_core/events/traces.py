"""Trace payload value types and the trace builders/summaries over them."""

from __future__ import annotations

from typing import Mapping
from rag_core.events.trace_payload_fields import (
    bool_field,
    float_field,
    int_field,
    int_tuple_field,
    optional_float_field,
    safe_label_field,
    safe_optional_label_field,
    safe_label_tuple_field,
    search_id_field,
    safe_stage_label_field,
    safe_stage_label_tuple_field,
    stage_field,
    str_tuple_field,
)
from rag_core.events.event_types import (
    RERANK_APPLIED_EVENT,
    SEARCH_COMPLETED_EVENT,
    SEARCH_PLANNED_EVENT,
    SEARCH_STAGE_COMPLETED_EVENT,
    SEARCH_STARTED_EVENT,
    SIDECAR_APPLIED_EVENT,
    STAGE_ERROR_EVENT,
)
from rag_core.events.types import (
    Event,
    RerankApplied,
    SearchCompleted,
    SearchPlanned,
    SearchStarted,
    SearchStageCompleted,
    SidecarApplied,
    StageError,
)
from dataclasses import replace
from typing import Iterable
from rag_core.events.trace_payload_fields import (
    safe_trace_label,
    safe_trace_label_sequence,
)
from rag_core.events.trace_summary_models import (
    SearchStageTraceSummary,
    SearchTraceSummary,
    safe_search_id,
    search_stage_trace_summary_from_event,
)


def _lexical_search_flag(payload: Mapping[str, object]) -> bool:
    return bool_field(payload, "use_lexical_search", default=False)

def search_event_from_payload(payload: Mapping[str, object]) -> Event | None:
    event_type = payload.get("event_type")
    if event_type == SEARCH_STARTED_EVENT:
        return SearchStarted(
            search_id=search_id_field(payload, "search_id"),
            collections=str_tuple_field(payload, "collections"),
            query_length=int_field(payload, "query_length"),
            limit=int_field(payload, "limit"),
            collection_count=int_field(payload, "collection_count"),
        )
    if event_type == SEARCH_PLANNED_EVENT:
        return SearchPlanned(
            search_id=search_id_field(payload, "search_id"),
            collections=str_tuple_field(payload, "collections"),
            limit=int_field(payload, "limit"),
            final_limit=int_field(payload, "final_limit"),
            collection_count=int_field(payload, "collection_count"),
            channels=safe_label_tuple_field(payload, "channels"),
            prefetch_limits=int_tuple_field(payload, "prefetch_limits"),
            search_profile=safe_optional_label_field(payload, "search_profile"),
            fusion=safe_optional_label_field(payload, "fusion"),
            plan_rerank=safe_optional_label_field(payload, "plan_rerank"),
            boost=safe_optional_label_field(payload, "boost"),
            metadata_filter=safe_optional_label_field(payload, "metadata_filter"),
            content_type_count=int_field(payload, "content_type_count"),
            document_id_count=int_field(payload, "document_id_count"),
            rerank_candidate_count=int_field(payload, "rerank_candidate_count"),
            rerank_timeout_ms=float_field(payload, "rerank_timeout_ms"),
            rerank_max_output=int_field(payload, "rerank_max_output"),
            rerank_fallback_on_error=bool_field(
                payload,
                "rerank_fallback_on_error",
                default=True,
            ),
            use_lexical_search=_lexical_search_flag(payload),
            query_transforms=safe_stage_label_tuple_field(payload, "query_transforms"),
            retrieve_stage=safe_stage_label_field(payload, "retrieve_stage"),
            fuse_stage=safe_stage_label_field(payload, "fuse_stage"),
            rerank_stage=safe_stage_label_field(payload, "rerank_stage"),
            postprocesses=safe_stage_label_tuple_field(payload, "postprocesses"),
        )
    if event_type == SEARCH_STAGE_COMPLETED_EVENT:
        return SearchStageCompleted(
            search_id=search_id_field(payload, "search_id"),
            stage=stage_field(payload, "stage"),
            stage_name=safe_stage_label_field(payload, "stage_name"),
            candidate_count=int_field(payload, "candidate_count"),
            result_count=int_field(payload, "result_count"),
            dropped_count=int_field(payload, "dropped_count"),
            truncated=bool_field(payload, "truncated"),
            max_chars=int_field(payload, "max_chars"),
            max_tokens=int_field(payload, "max_tokens"),
            token_estimate=int_field(payload, "token_estimate"),
            char_count=int_field(payload, "char_count"),
            citation_count=int_field(payload, "citation_count"),
            source_preview_count=int_field(payload, "source_preview_count"),
            duration_ms=float_field(payload, "duration_ms"),
        )
    if event_type == SEARCH_COMPLETED_EVENT:
        return SearchCompleted(
            search_id=search_id_field(payload, "search_id"),
            result_count=int_field(payload, "result_count"),
            requested_rerank=bool_field(payload, "requested_rerank"),
            requested_sidecar=bool_field(payload, "requested_sidecar"),
            attempted_rerank=bool_field(payload, "attempted_rerank"),
            attempted_sidecar=bool_field(payload, "attempted_sidecar"),
            applied_rerank=bool_field(payload, "applied_rerank"),
            applied_sidecar=bool_field(payload, "applied_sidecar"),
            succeeded=bool_field(payload, "succeeded", default=True),
            duration_ms=float_field(payload, "duration_ms"),
            collections=str_tuple_field(payload, "collections"),
            returned_document_ids=str_tuple_field(payload, "returned_document_ids"),
        )
    if event_type == RERANK_APPLIED_EVENT:
        return RerankApplied(
            search_id=search_id_field(payload, "search_id"),
            provider=safe_optional_label_field(payload, "provider"),
            model=safe_optional_label_field(payload, "model"),
            input_count=int_field(payload, "input_count"),
            candidate_count=int_field(payload, "candidate_count"),
            result_count=int_field(payload, "result_count"),
            top_k=int_field(payload, "top_k"),
            fallback_reason=safe_optional_label_field(payload, "fallback_reason"),
            truncation_reason=safe_optional_label_field(payload, "truncation_reason"),
            duration_ms=float_field(payload, "duration_ms"),
            succeeded=bool_field(payload, "succeeded", default=True),
            provider_result_count=int_field(payload, "provider_result_count"),
            accepted_count=int_field(payload, "accepted_count"),
            dropped_count=int_field(payload, "dropped_count"),
            rank_changed_count=int_field(payload, "rank_changed_count"),
            rank_promoted_count=int_field(payload, "rank_promoted_count"),
            rank_demoted_count=int_field(payload, "rank_demoted_count"),
            max_rank_gain=int_field(payload, "max_rank_gain"),
            max_rank_loss=int_field(payload, "max_rank_loss"),
            provider_score_min=optional_float_field(payload, "provider_score_min"),
            provider_score_max=optional_float_field(payload, "provider_score_max"),
            search_score_min=optional_float_field(payload, "search_score_min"),
            search_score_max=optional_float_field(payload, "search_score_max"),
        )
    if event_type == SIDECAR_APPLIED_EVENT:
        return SidecarApplied(
            search_id=search_id_field(payload, "search_id"),
            provider=safe_optional_label_field(payload, "provider"),
            input_count=int_field(payload, "input_count"),
            provider_result_count=int_field(payload, "provider_result_count"),
            accepted_count=int_field(payload, "accepted_count"),
            dropped_count=int_field(payload, "dropped_count"),
            result_count=int_field(payload, "result_count"),
            duration_ms=float_field(payload, "duration_ms"),
            succeeded=bool_field(payload, "succeeded", default=True),
            fallback_reason=safe_optional_label_field(payload, "fallback_reason"),
        )
    if event_type == STAGE_ERROR_EVENT:
        return StageError(
            search_id=search_id_field(payload, "search_id"),
            stage=safe_label_field(payload, "stage"),
            error_type=safe_label_field(payload, "error_type"),
        )
    return None


def summarize_search_trace(events: Iterable[Event]) -> SearchTraceSummary:
    summary = SearchTraceSummary()

    for event in events:
        summary = _apply_search_event(summary, event)

    return summary

def summarize_search_trace_runs(events: Iterable[Event]) -> tuple[SearchTraceSummary, ...]:
    buffered = [event for event in events if _is_search_trace_event(event)]
    if any(_search_id(event) for event in buffered):
        return _summarize_correlated_search_trace_runs(buffered)
    return _summarize_uncorrelated_search_trace_runs(buffered)

def _summarize_uncorrelated_search_trace_runs(
    events: Iterable[Event],
) -> tuple[SearchTraceSummary, ...]:
    summaries: list[SearchTraceSummary] = []
    summary = SearchTraceSummary()
    seen_search = False

    for event in events:
        if isinstance(event, SearchStarted):
            if seen_search:
                summaries.append(summary)
                summary = SearchTraceSummary()
            seen_search = True
            summary = _apply_search_event(summary, event)
        elif _is_search_trace_event(event):
            if seen_search:
                summary = _apply_search_event(summary, event)

    if seen_search:
        summaries.append(summary)
    return tuple(summaries)

def _summarize_correlated_search_trace_runs(
    events: Iterable[Event],
) -> tuple[SearchTraceSummary, ...]:
    summaries: dict[str, SearchTraceSummary] = {}
    ordered_search_ids: list[str] = []
    uncorrelated_lifecycle_events = 0
    uncorrelated_auxiliary_events: list[Event] = []
    for event in events:
        search_id = _search_id(event)
        if not search_id:
            if isinstance(event, (SearchStarted, SearchPlanned, SearchCompleted)):
                uncorrelated_lifecycle_events += 1
            else:
                uncorrelated_auxiliary_events.append(event)
            continue
        if search_id not in summaries:
            summaries[search_id] = SearchTraceSummary(search_id=search_id)
            ordered_search_ids.append(search_id)
        summaries[search_id] = _apply_search_event(summaries[search_id], event)
    if uncorrelated_lifecycle_events:
        raise ValueError("trace contains mixed correlated and uncorrelated search events")
    if uncorrelated_auxiliary_events:
        if len(ordered_search_ids) != 1:
            raise ValueError("trace contains mixed correlated and uncorrelated search events")
        correlated_search_id = ordered_search_ids[0]
        summary = summaries[correlated_search_id]
        for event in uncorrelated_auxiliary_events:
            summary = _apply_search_event(summary, event)
        summaries[correlated_search_id] = summary
    return tuple(summaries[search_id] for search_id in ordered_search_ids)

def summarize_search_trace_payloads(
    payloads: Iterable[Mapping[str, object]],
) -> SearchTraceSummary:
    events = [
        event
        for payload in payloads
        if (event := search_event_from_payload(payload)) is not None
    ]
    search_ids = {search_id for event in events if (search_id := _search_id(event))}
    if len(search_ids) > 1:
        raise ValueError("trace contains multiple search_ids")
    return summarize_search_trace(events)

def summarize_search_trace_payload_runs(
    payloads: Iterable[Mapping[str, object]],
) -> tuple[SearchTraceSummary, ...]:
    events = (
        event
        for payload in payloads
        if (event := search_event_from_payload(payload)) is not None
    )
    return summarize_search_trace_runs(events)

def _apply_search_event(
    summary: SearchTraceSummary,
    event: Event,
) -> SearchTraceSummary:
    event_search_id = _search_id(event)
    summary_search_id = safe_search_id(summary.search_id)
    effective_search_id = event_search_id or summary_search_id

    if isinstance(event, SearchStarted):
        return replace(
            summary,
            search_id=effective_search_id,
            query_length=event.query_length,
            limit=event.limit,
            collection_count=event.collection_count,
        )
    if isinstance(event, SearchPlanned):
        return replace(
            summary,
            search_id=effective_search_id,
            collection_count=event.collection_count,
            limit=event.limit,
            final_limit=event.final_limit,
            channels=tuple(_safe_labels(event.channels)),
            prefetch_limits=event.prefetch_limits,
            search_profile=_safe_optional_label(event.search_profile),
            fusion=_safe_optional_label(event.fusion),
            plan_rerank=_safe_optional_label(event.plan_rerank),
            boost=_safe_optional_label(event.boost),
            metadata_filter=_safe_optional_label(event.metadata_filter),
            content_type_count=event.content_type_count,
            document_id_count=event.document_id_count,
            rerank_candidate_count=event.rerank_candidate_count,
            rerank_timeout_ms=event.rerank_timeout_ms,
            rerank_max_output=event.rerank_max_output,
            rerank_fallback_on_error=event.rerank_fallback_on_error,
            use_lexical_search=event.use_lexical_search,
            query_transforms=tuple(_safe_stage_labels(event.query_transforms)),
            retrieve_stage=_safe_stage_label(event.retrieve_stage),
            fuse_stage=_safe_stage_label(event.fuse_stage),
            rerank_stage=_safe_stage_label(event.rerank_stage),
            postprocesses=tuple(_safe_stage_labels(event.postprocesses)),
        )
    if isinstance(event, SearchStageCompleted):
        stage_summary = search_stage_trace_summary_from_event(event)
        return replace(summary, stages=(*summary.stages, stage_summary))
    if isinstance(event, SearchCompleted):
        return replace(
            summary,
            search_id=effective_search_id,
            result_count=event.result_count,
            requested_rerank=event.requested_rerank,
            requested_sidecar=event.requested_sidecar,
            attempted_rerank=summary.attempted_rerank or event.attempted_rerank,
            attempted_sidecar=summary.attempted_sidecar or event.attempted_sidecar,
            applied_rerank=summary.applied_rerank or event.applied_rerank,
            applied_sidecar=summary.applied_sidecar or event.applied_sidecar,
            succeeded=event.succeeded,
            duration_ms=event.duration_ms,
            completed=True,
        )
    if isinstance(event, RerankApplied):
        return replace(
            summary,
            search_id=effective_search_id,
            attempted_rerank=True,
            applied_rerank=event.succeeded and event.accepted_count > 0,
            rerank_provider=_safe_optional_label(event.provider),
            rerank_model=_safe_optional_label(event.model),
            rerank_input_count=event.input_count,
            rerank_applied_candidate_count=event.candidate_count,
            rerank_provider_result_count=event.provider_result_count,
            rerank_accepted_count=event.accepted_count,
            rerank_dropped_count=event.dropped_count,
            rerank_rank_changed_count=event.rank_changed_count,
            rerank_rank_promoted_count=event.rank_promoted_count,
            rerank_rank_demoted_count=event.rank_demoted_count,
            rerank_max_rank_gain=event.max_rank_gain,
            rerank_max_rank_loss=event.max_rank_loss,
            rerank_provider_score_min=event.provider_score_min,
            rerank_provider_score_max=event.provider_score_max,
            rerank_search_score_min=event.search_score_min,
            rerank_search_score_max=event.search_score_max,
            rerank_result_count=event.result_count,
            rerank_top_k=event.top_k,
            rerank_fallback_reason=_safe_optional_label(event.fallback_reason),
            rerank_truncation_reason=_safe_optional_label(event.truncation_reason),
            rerank_duration_ms=event.duration_ms,
            rerank_succeeded=event.succeeded,
        )
    if isinstance(event, SidecarApplied):
        return replace(
            summary,
            search_id=effective_search_id,
            attempted_sidecar=True,
            applied_sidecar=event.succeeded and event.accepted_count > 0,
            sidecar_provider=_safe_optional_label(event.provider),
            sidecar_input_count=event.input_count,
            sidecar_provider_result_count=event.provider_result_count,
            sidecar_accepted_count=event.accepted_count,
            sidecar_dropped_count=event.dropped_count,
            sidecar_result_count=event.result_count,
            sidecar_duration_ms=event.duration_ms,
            sidecar_succeeded=event.succeeded,
            sidecar_fallback_reason=_safe_optional_label(event.fallback_reason),
        )
    if isinstance(event, StageError):
        return _append_stage_error(summary, event)
    return summary

def _append_stage_error(summary: SearchTraceSummary, event: StageError) -> SearchTraceSummary:
    stage = _safe_optional_label(event.stage)
    error_type = _safe_optional_label(event.error_type)
    if stage == "search":
        if summary.error_types:
            return summary
        return replace(
            summary,
            error_stages=(*summary.error_stages, stage),
            error_types=(*summary.error_types, error_type),
        )
    if "search" in summary.error_stages:
        concrete_pairs = [
            pair
            for pair in zip(summary.error_stages, summary.error_types, strict=False)
            if pair[0] != "search"
        ]
        return replace(
            summary,
            error_stages=(*[pair[0] for pair in concrete_pairs], stage),
            error_types=(*[pair[1] for pair in concrete_pairs], error_type),
        )
    return replace(
        summary,
        error_stages=(*summary.error_stages, stage),
        error_types=(*summary.error_types, error_type),
    )

def _safe_optional_label(value: str) -> str:
    if not value:
        return ""
    return safe_trace_label(value, stage=False)

def _safe_stage_label(value: str) -> str:
    if not value:
        return ""
    return safe_trace_label(value, stage=True)

def _safe_stage_labels(values: tuple[str, ...]) -> list[str]:
    return safe_trace_label_sequence(values, stage=True)

def _safe_labels(values: tuple[str, ...]) -> list[str]:
    return safe_trace_label_sequence(values, stage=False)

def _is_search_trace_event(event: Event) -> bool:
    return isinstance(
        event,
        (
            SearchStarted,
            SearchPlanned,
            SearchStageCompleted,
            SearchCompleted,
            RerankApplied,
            SidecarApplied,
            StageError,
        ),
    )

def _search_id(event: Event) -> str:
    return safe_search_id(getattr(event, "search_id", ""))


__all__ = [
    "SearchStageTraceSummary",
    "SearchTraceSummary",
    "summarize_search_trace",
    "summarize_search_trace_payloads",
    "summarize_search_trace_payload_runs",
    "summarize_search_trace_runs",
]
