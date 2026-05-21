from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping

from rag_core.events.trace_payload_fields import (
    safe_trace_label,
    safe_trace_label_sequence,
)
from rag_core.events.trace_payloads import search_event_from_payload
from rag_core.events.trace_summary_models import (
    SearchStageTraceSummary,
    SearchTraceSummary,
    safe_search_id,
    search_stage_trace_summary_from_event,
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
            corpus_count=event.corpus_count,
        )
    if isinstance(event, SearchPlanned):
        return replace(
            summary,
            search_id=effective_search_id,
            corpus_count=event.corpus_count,
            limit=event.limit,
            final_limit=event.final_limit,
            channels=tuple(_safe_labels(event.channels)),
            prefetch_limits=event.prefetch_limits,
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
            use_sidecar=event.use_sidecar,
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
            used_rerank=event.used_rerank,
            used_sidecar=event.used_sidecar,
            requested_rerank=event.requested_rerank,
            requested_sidecar=event.requested_sidecar,
            attempted_rerank=event.attempted_rerank,
            attempted_sidecar=event.attempted_sidecar,
            applied_rerank=event.applied_rerank,
            applied_sidecar=event.applied_sidecar,
            rerank_attempted=summary.rerank_attempted or event.attempted_rerank,
            rerank_applied=summary.rerank_applied or event.applied_rerank,
            sidecar_attempted=summary.sidecar_attempted or event.attempted_sidecar,
            sidecar_applied=summary.sidecar_applied or event.applied_sidecar,
            succeeded=event.succeeded,
            duration_ms=event.duration_ms,
            completed=True,
        )
    if isinstance(event, RerankApplied):
        return replace(
            summary,
            search_id=effective_search_id,
            rerank_attempted=True,
            rerank_applied=event.succeeded and event.accepted_count > 0,
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
            sidecar_attempted=True,
            sidecar_applied=event.succeeded and event.accepted_count > 0,
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
