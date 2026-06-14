"""Search, rerank, sidecar, and stage-error event records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rag_core.events.event_types import (
    LEXICAL_SIDECAR_BOUND_EXCEEDED_EVENT,
    NEIGHBOR_EXPAND_SKIPPED_EVENT,
    RERANK_APPLIED_EVENT,
    SEARCH_COMPLETED_EVENT,
    SEARCH_PLANNED_EVENT,
    SEARCH_STAGE_COMPLETED_EVENT,
    SEARCH_STARTED_EVENT,
    SIDECAR_APPLIED_EVENT,
    STAGE_ERROR_EVENT,
)
from rag_core.events.trace_payload_fields import TRACE_ABSENT_LABEL
from rag_core.events.trace_payload_fields import RETRIEVE_SEARCH_STAGE

# Audit lines should not be unbounded. SearchCompleted carries up to this many
# document_ids inline; use SearchStageCompleted (retrieve stage) or the OTel
# sink for the full list when callers need it.
RETURNED_DOCUMENT_IDS_CAP: int = 64


@dataclass(frozen=True)
class SearchStarted:
    namespace: str = ""
    corpus_ids: tuple[str, ...] = ()
    query_length: int = 0
    limit: int = 0
    corpus_count: int = 0
    search_id: str = ""
    event_type: Literal["search.started"] = SEARCH_STARTED_EVENT
    emitted_at_ns: int = 0
    wall_clock_ns: int = 0
    actor: str = ""
    request_id: str = ""
    ingest_id: str = ""

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
    use_lexical_search: bool = False
    query_transforms: tuple[str, ...] = ()
    retrieve_stage: str = ""
    fuse_stage: str = ""
    rerank_stage: str = ""
    postprocesses: tuple[str, ...] = ()
    search_id: str = ""
    event_type: Literal["search.planned"] = SEARCH_PLANNED_EVENT
    emitted_at_ns: int = 0
    wall_clock_ns: int = 0
    actor: str = ""
    request_id: str = ""
    ingest_id: str = ""

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
    ] = RETRIEVE_SEARCH_STAGE
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
    event_type: Literal["search.stage.completed"] = SEARCH_STAGE_COMPLETED_EVENT
    emitted_at_ns: int = 0
    wall_clock_ns: int = 0
    actor: str = ""
    request_id: str = ""
    ingest_id: str = ""


@dataclass(frozen=True)
class SearchCompleted:
    namespace: str = ""
    result_count: int = 0
    requested_rerank: bool = False
    requested_sidecar: bool = False
    attempted_rerank: bool = False
    attempted_sidecar: bool = False
    applied_rerank: bool = False
    applied_sidecar: bool = False
    succeeded: bool = True
    duration_ms: float = 0.0
    search_id: str = ""
    # Tier scope that was searched. Required for audit ("which tier was hit")
    # Kept on the same event as the duration/result_count so a single audit
    # row captures the answer to "who searched what tier and got how many
    # results in how long".
    corpus_ids: tuple[str, ...] = ()
    # Document IDs returned, capped to avoid unbounded audit lines. Callers
    # needing the full list should subscribe to SearchStageCompleted on the
    # retrieve stage or the OTel sink. Capped to RETURNED_DOCUMENT_IDS_CAP.
    returned_document_ids: tuple[str, ...] = ()
    event_type: Literal["search.completed"] = SEARCH_COMPLETED_EVENT
    emitted_at_ns: int = 0
    wall_clock_ns: int = 0
    actor: str = ""
    request_id: str = ""
    ingest_id: str = ""


@dataclass(frozen=True)
class RerankApplied:
    provider: str = ""
    model: str = ""
    input_count: int = 0
    candidate_count: int = 0
    result_count: int = 0
    top_k: int = 0
    fallback_reason: str = ""
    truncation_reason: str = TRACE_ABSENT_LABEL
    duration_ms: float = 0.0
    succeeded: bool = True
    event_type: Literal["rerank.applied"] = RERANK_APPLIED_EVENT
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
    emitted_at_ns: int = 0
    wall_clock_ns: int = 0
    actor: str = ""
    request_id: str = ""
    ingest_id: str = ""


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
    event_type: Literal["sidecar.applied"] = SIDECAR_APPLIED_EVENT
    emitted_at_ns: int = 0
    wall_clock_ns: int = 0
    actor: str = ""
    request_id: str = ""
    ingest_id: str = ""


@dataclass(frozen=True)
class NeighborExpandSkipped:
    reason: str = ""
    input_count: int = 0
    result_count: int = 0
    search_id: str = ""
    event_type: Literal[
        "neighbor_expand.skipped"
    ] = NEIGHBOR_EXPAND_SKIPPED_EVENT
    emitted_at_ns: int = 0
    wall_clock_ns: int = 0
    actor: str = ""
    request_id: str = ""
    ingest_id: str = ""


@dataclass(frozen=True)
class LexicalSidecarBoundExceeded:
    """Emitted when an upsert would exceed the lexical sidecar's bounds.

    The sidecar refuses the overflow rather than silently OOMing. ``reason``
    is one of ``"max_entries"`` or ``"max_bytes"``; ``rejected_count`` is how
    many records in the incoming batch were dropped.
    """

    provider: str = ""
    reason: str = ""
    rejected_count: int = 0
    current_entries: int = 0
    max_entries: int = 0
    current_bytes: int = 0
    max_bytes: int = 0
    event_type: Literal[
        "lexical_sidecar.bound_exceeded"
    ] = LEXICAL_SIDECAR_BOUND_EXCEEDED_EVENT
    emitted_at_ns: int = 0
    wall_clock_ns: int = 0
    actor: str = ""
    request_id: str = ""
    ingest_id: str = ""


@dataclass(frozen=True)
class StageError:
    stage: str = ""
    error_type: str = ""
    message: str = ""
    search_id: str = ""
    event_type: Literal["stage.error"] = STAGE_ERROR_EVENT
    emitted_at_ns: int = 0
    wall_clock_ns: int = 0
    actor: str = ""
    request_id: str = ""
    ingest_id: str = ""
