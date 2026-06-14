"""Sidecar result filtering and trace payload helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_core.events.emit import emit_event
from rag_core.events.types import SidecarApplied
from rag_core.search.pipeline.types import PipelineQuery
from rag_core.search.result_filters import result_matches_sidecar_query
from rag_core.search.request_models import SearchSidecarQuery
from rag_core.search.vector_models import SearchResult

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


@dataclass(frozen=True)
class SidecarApplication:
    provider_results: list[SearchResult]
    accepted_results: list[SearchResult]
    dropped_count: int


def build_sidecar_query(query: PipelineQuery) -> SearchSidecarQuery:
    return SearchSidecarQuery(
        query=query.query,
        namespace=query.namespace,
        corpus_ids=query.corpus_ids,
        limit=query.limit,
        content_types=query.content_types,
        document_ids=query.document_ids,
        metadata_filter=query.metadata_filter,
    )


def prepare_sidecar_application(
    query: PipelineQuery,
    provider_results: list[SearchResult],
    vector_results: list[SearchResult],
) -> SidecarApplication:
    sidecar_query = build_sidecar_query(query)
    vector_ids = {result.id for result in vector_results}
    sidecar_only_slots = max(query.limit - len(vector_results), 1)
    accepted_results: list[SearchResult] = []
    for result in provider_results:
        if not result_matches_sidecar_query(
            result,
            sidecar_query,
        ):
            continue
        if result.id in vector_ids:
            accepted_results.append(result)
            continue
        if sidecar_only_slots <= 0:
            continue
        accepted_results.append(result)
        sidecar_only_slots -= 1
    return SidecarApplication(
        provider_results=provider_results,
        accepted_results=accepted_results,
        dropped_count=len(provider_results) - len(accepted_results),
    )


def emit_sidecar_failure(
    sink: "EventSink | None",
    *,
    provider: str,
    input_count: int,
    result_count: int,
    duration_ms: float,
    fallback_reason: str,
) -> None:
    emit_event(
        sink,
        SidecarApplied(
            provider=provider,
            input_count=input_count,
            result_count=result_count,
            duration_ms=duration_ms,
            succeeded=False,
            fallback_reason=fallback_reason,
        ),
    )


def emit_sidecar_success(
    sink: "EventSink | None",
    *,
    provider: str,
    input_count: int,
    application: SidecarApplication,
    result_count: int,
    duration_ms: float,
) -> None:
    emit_event(
        sink,
        SidecarApplied(
            provider=provider,
            input_count=input_count,
            provider_result_count=len(application.provider_results),
            accepted_count=len(application.accepted_results),
            dropped_count=application.dropped_count,
            result_count=result_count,
            duration_ms=duration_ms,
        ),
    )
