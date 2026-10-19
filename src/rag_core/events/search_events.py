"""Search, rerank, and stage-error event records."""

from __future__ import annotations

from dataclasses import dataclass
from rag_core.events.record import EventRecord
from typing import Literal

from rag_core.events.event_types import (
    RERANK_APPLIED_EVENT,
    SEARCH_COMPLETED_EVENT,
    SEARCH_PLANNED_EVENT,
    SEARCH_STAGE_COMPLETED_EVENT,
    SEARCH_STARTED_EVENT,
    STAGE_ERROR_EVENT,
)
from rag_core.events.trace_payload_fields import TRACE_ABSENT_LABEL
from rag_core.events.trace_payload_fields import RETRIEVE_SEARCH_STAGE

# Audit lines should not be unbounded. SearchCompleted carries up to this many
# document_ids inline. Applications needing all IDs should use search results.
RETURNED_DOCUMENT_IDS_CAP: int = 64


@dataclass(frozen=True)
class SearchStarted(EventRecord):
    tenant_id: str = ""
    collections: tuple[str, ...] = ()
    query_length: int = 0
    limit: int = 0
    collection_count: int = 0
    event_type: Literal["search.started"] = SEARCH_STARTED_EVENT

    def __post_init__(self) -> None:
        if self.collection_count == 0 and self.collections:
            object.__setattr__(self, "collection_count", len(self.collections))


@dataclass(frozen=True)
class SearchPlanned(EventRecord):
    tenant_id: str = ""
    collections: tuple[str, ...] = ()
    limit: int = 0
    collection_count: int = 0
    channels: tuple[str, ...] = ()
    mode: str = ""
    candidate_limit: int = 0
    rerank_enabled: bool = False
    fusion: str = ""
    metadata_filter: str = ""
    content_type_count: int = 0
    document_id_count: int = 0
    rerank_timeout_ms: float = 0.0
    rerank_max_output: int = 0
    rerank_fallback_on_error: bool = False
    event_type: Literal["search.planned"] = SEARCH_PLANNED_EVENT

    def __post_init__(self) -> None:
        if self.collection_count == 0 and self.collections:
            object.__setattr__(self, "collection_count", len(self.collections))


@dataclass(frozen=True)
class SearchStageCompleted(EventRecord):
    stage: Literal[
        "retrieve",
        "rerank",
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
    event_type: Literal["search.stage.completed"] = SEARCH_STAGE_COMPLETED_EVENT


@dataclass(frozen=True)
class SearchCompleted(EventRecord):
    tenant_id: str = ""
    result_count: int = 0
    requested_rerank: bool = False
    attempted_rerank: bool = False
    applied_rerank: bool = False
    succeeded: bool = True
    duration_ms: float = 0.0
    # Tier scope that was searched. Required for audit ("which tier was hit")
    # Kept on the same event as the duration/result_count so a single audit
    # row captures the answer to "who searched what tier and got how many
    # results in how long".
    collections: tuple[str, ...] = ()
    # Document IDs returned, capped to RETURNED_DOCUMENT_IDS_CAP.
    returned_document_ids: tuple[str, ...] = ()
    event_type: Literal["search.completed"] = SEARCH_COMPLETED_EVENT


@dataclass(frozen=True)
class RerankApplied(EventRecord):
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


@dataclass(frozen=True)
class StageError(EventRecord):
    stage: str = ""
    error_type: str = ""
    message: str = ""
    event_type: Literal["stage.error"] = STAGE_ERROR_EVENT
