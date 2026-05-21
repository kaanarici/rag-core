"""Search, rerank, sidecar, and stage-error event records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SearchStarted:
    namespace: str = ""
    corpus_ids: tuple[str, ...] = ()
    query_length: int = 0
    limit: int = 0
    corpus_count: int = 0
    search_id: str = ""
    event_type: Literal["search.started"] = "search.started"

    def __post_init__(self) -> None:
        if self.corpus_count == 0 and self.corpus_ids:
            object.__setattr__(self, "corpus_count", len(self.corpus_ids))


@dataclass(frozen=True)
class SearchPlanned:
    namespace: str = ""
    corpus_ids: tuple[str, ...] = ()
    limit: int = 0
    final_limit: int = 0
    corpus_count: int = 0
    channels: tuple[str, ...] = ()
    prefetch_limits: tuple[int, ...] = ()
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
    use_lexical_search: bool = False
    query_transforms: tuple[str, ...] = ()
    retrieve_stage: str = ""
    fuse_stage: str = ""
    rerank_stage: str = ""
    postprocesses: tuple[str, ...] = ()
    search_id: str = ""
    event_type: Literal["search.planned"] = "search.planned"

    def __post_init__(self) -> None:
        if self.corpus_count == 0 and self.corpus_ids:
            object.__setattr__(self, "corpus_count", len(self.corpus_ids))


@dataclass(frozen=True)
class SearchStageCompleted:
    stage: Literal[
        "query_transform",
        "retrieve",
        "fuse",
        "rerank",
        "postprocess",
        "context_pack",
    ] = "retrieve"
    stage_name: str = ""
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
    search_id: str = ""
    event_type: Literal["search.stage.completed"] = "search.stage.completed"


@dataclass(frozen=True)
class SearchCompleted:
    namespace: str = ""
    result_count: int = 0
    used_rerank: bool = False
    used_sidecar: bool = False
    requested_rerank: bool = False
    requested_sidecar: bool = False
    attempted_rerank: bool = False
    attempted_sidecar: bool = False
    applied_rerank: bool = False
    applied_sidecar: bool = False
    succeeded: bool = True
    duration_ms: float = 0.0
    search_id: str = ""
    event_type: Literal["search.completed"] = "search.completed"


@dataclass(frozen=True)
class RerankApplied:
    provider: str = ""
    model: str = ""
    input_count: int = 0
    candidate_count: int = 0
    result_count: int = 0
    top_k: int = 0
    fallback_reason: str = ""
    truncation_reason: str = "none"
    duration_ms: float = 0.0
    succeeded: bool = True
    event_type: Literal["rerank.applied"] = "rerank.applied"
    provider_result_count: int = 0
    accepted_count: int = 0
    dropped_count: int = 0
    rank_changed_count: int = 0
    rank_promoted_count: int = 0
    rank_demoted_count: int = 0
    max_rank_gain: int = 0
    max_rank_loss: int = 0
    provider_score_min: float | None = 0.0
    provider_score_max: float | None = 0.0
    search_score_min: float | None = 0.0
    search_score_max: float | None = 0.0
    search_id: str = ""


@dataclass(frozen=True)
class SidecarApplied:
    provider: str = ""
    input_count: int = 0
    provider_result_count: int = 0
    accepted_count: int = 0
    dropped_count: int = 0
    result_count: int = 0
    duration_ms: float = 0.0
    succeeded: bool = True
    fallback_reason: str = ""
    search_id: str = ""
    event_type: Literal["sidecar.applied"] = "sidecar.applied"


@dataclass(frozen=True)
class StageError:
    stage: str = ""
    error_type: str = ""
    message: str = ""
    search_id: str = ""
    event_type: Literal["stage.error"] = "stage.error"
