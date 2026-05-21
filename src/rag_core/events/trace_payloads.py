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


def _lexical_search_flag(payload: Mapping[str, object]) -> bool:
    """Read lexical flag from trace JSON; accept legacy ``use_sidecar`` key."""
    if "use_lexical_search" in payload:
        return bool_field(payload, "use_lexical_search")
    if "use_sidecar" in payload:
        return bool_field(payload, "use_sidecar")
    return False


def search_event_from_payload(payload: Mapping[str, object]) -> Event | None:
    event_type = payload.get("event_type")
    if event_type == "search.started":
        return SearchStarted(
            search_id=search_id_field(payload, "search_id"),
            corpus_ids=str_tuple_field(payload, "corpus_ids"),
            query_length=int_field(payload, "query_length"),
            limit=int_field(payload, "limit"),
            corpus_count=int_field(payload, "corpus_count"),
        )
    if event_type == "search.planned":
        return SearchPlanned(
            search_id=search_id_field(payload, "search_id"),
            corpus_ids=str_tuple_field(payload, "corpus_ids"),
            limit=int_field(payload, "limit"),
            final_limit=int_field(payload, "final_limit"),
            corpus_count=int_field(payload, "corpus_count"),
            channels=safe_label_tuple_field(payload, "channels"),
            prefetch_limits=int_tuple_field(payload, "prefetch_limits"),
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
    if event_type == "search.stage.completed":
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
    if event_type == "search.completed":
        return SearchCompleted(
            search_id=search_id_field(payload, "search_id"),
            result_count=int_field(payload, "result_count"),
            used_rerank=bool_field(payload, "used_rerank"),
            used_sidecar=bool_field(payload, "used_sidecar"),
            requested_rerank=bool_field(payload, "requested_rerank"),
            requested_sidecar=bool_field(payload, "requested_sidecar"),
            attempted_rerank=bool_field(payload, "attempted_rerank"),
            attempted_sidecar=bool_field(payload, "attempted_sidecar"),
            applied_rerank=bool_field(payload, "applied_rerank"),
            applied_sidecar=bool_field(payload, "applied_sidecar"),
            succeeded=bool_field(payload, "succeeded", default=True),
            duration_ms=float_field(payload, "duration_ms"),
        )
    if event_type == "rerank.applied":
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
    if event_type == "sidecar.applied":
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
    if event_type == "stage.error":
        return StageError(
            search_id=search_id_field(payload, "search_id"),
            stage=safe_label_field(payload, "stage"),
            error_type=safe_label_field(payload, "error_type"),
        )
    return None
