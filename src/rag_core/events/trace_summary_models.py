from __future__ import annotations

from dataclasses import asdict, dataclass

from rag_core.events.trace_payload_fields import safe_trace_label
from rag_core.events.types import SearchStageCompleted


def safe_search_id(value: object) -> str:
    if not isinstance(value, str):
        return ""
    if not value:
        return ""
    if safe_trace_label(value, stage=False) != value:
        return ""
    return value


@dataclass(frozen=True)
class SearchStageTraceSummary:
    stage: str
    stage_name: str
    candidate_count: int = 0
    result_count: int = 0
    dropped_count: int = 0
    truncated: bool = False
    max_chars: int = 0
    max_tokens: int = 0
    token_estimate: int = 0
    char_count: int = 0
    citation_count: int = 0
    source_preview_count: int = 0
    duration_ms: float = 0.0

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SearchTraceSummary:
    search_id: str = ""
    query_length: int = 0
    limit: int = 0
    corpus_count: int = 0
    final_limit: int = 0
    channels: tuple[str, ...] = ()
    prefetch_limits: tuple[int, ...] = ()
    search_profile: str = ""
    fusion: str = ""
    plan_rerank: str = ""
    boost: str = ""
    metadata_filter: str = ""
    content_type_count: int = 0
    document_id_count: int = 0
    rerank_candidate_count: int = 0
    rerank_timeout_ms: float = 0.0
    rerank_max_output: int = 0
    rerank_fallback_on_error: bool = True
    rerank_provider: str = ""
    rerank_model: str = ""
    rerank_input_count: int = 0
    rerank_applied_candidate_count: int = 0
    rerank_provider_result_count: int = 0
    rerank_accepted_count: int = 0
    rerank_dropped_count: int = 0
    rerank_rank_changed_count: int = 0
    rerank_rank_promoted_count: int = 0
    rerank_rank_demoted_count: int = 0
    rerank_max_rank_gain: int = 0
    rerank_max_rank_loss: int = 0
    rerank_provider_score_min: float | None = None
    rerank_provider_score_max: float | None = None
    rerank_search_score_min: float | None = None
    rerank_search_score_max: float | None = None
    rerank_result_count: int = 0
    rerank_top_k: int = 0
    rerank_fallback_reason: str = ""
    rerank_truncation_reason: str = ""
    rerank_duration_ms: float = 0.0
    rerank_succeeded: bool = False
    use_lexical_search: bool = False
    sidecar_provider: str = ""
    sidecar_input_count: int = 0
    sidecar_provider_result_count: int = 0
    sidecar_accepted_count: int = 0
    sidecar_dropped_count: int = 0
    sidecar_result_count: int = 0
    sidecar_duration_ms: float = 0.0
    sidecar_succeeded: bool = False
    sidecar_fallback_reason: str = ""
    query_transforms: tuple[str, ...] = ()
    retrieve_stage: str = ""
    fuse_stage: str = ""
    rerank_stage: str = ""
    postprocesses: tuple[str, ...] = ()
    stages: tuple[SearchStageTraceSummary, ...] = ()
    result_count: int = 0
    requested_rerank: bool = False
    requested_sidecar: bool = False
    attempted_rerank: bool = False
    attempted_sidecar: bool = False
    applied_rerank: bool = False
    applied_sidecar: bool = False
    succeeded: bool = False
    duration_ms: float = 0.0
    completed: bool = False
    error_stages: tuple[str, ...] = ()
    error_types: tuple[str, ...] = ()

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    @property
    def error_count(self) -> int:
        return len(self.error_types)

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["search_id"] = safe_search_id(self.search_id)
        payload["schema_version"] = 1
        for key in _SEQUENCE_PAYLOAD_FIELDS:
            payload[key] = list(getattr(self, key))
        payload["stage_count"] = self.stage_count
        payload["error_count"] = self.error_count
        payload["stages"] = [stage.to_payload() for stage in self.stages]
        return payload


def search_stage_trace_summary_from_event(
    event: SearchStageCompleted,
) -> SearchStageTraceSummary:
    return SearchStageTraceSummary(
        stage=event.stage,
        stage_name=safe_trace_label(event.stage_name, stage=True),
        candidate_count=event.candidate_count,
        result_count=event.result_count,
        dropped_count=event.dropped_count,
        truncated=event.truncated,
        max_chars=event.max_chars,
        max_tokens=event.max_tokens,
        token_estimate=event.token_estimate,
        char_count=event.char_count,
        citation_count=event.citation_count,
        source_preview_count=event.source_preview_count,
        duration_ms=event.duration_ms,
    )


_SEQUENCE_PAYLOAD_FIELDS = (
    "channels",
    "prefetch_limits",
    "query_transforms",
    "postprocesses",
    "error_stages",
    "error_types",
)


__all__ = [
    "SearchStageTraceSummary",
    "SearchTraceSummary",
    "safe_search_id",
    "search_stage_trace_summary_from_event",
]
