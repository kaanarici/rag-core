"""Consolidated checks that event types stay under rag_core.events."""

from __future__ import annotations

import pytest

import rag_core
from rag_core.events import (
    EmbedCompleted,
    EmbeddingTraceSummary,
    OpenTelemetrySink,
    SearchStageTraceSummary,
    SearchTraceSummary,
    SidecarApplied,
    summarize_embedding_trace,
    summarize_search_trace,
)
from rag_core.events import EmbeddingTraceSummary as EventsEmbeddingTraceSummary
from rag_core.events import SearchTraceSummary as EventsSearchTraceSummary


@pytest.mark.meta
@pytest.mark.parametrize(
    ("symbol_name", "expected"),
    [
        ("SearchTraceSummary", SearchTraceSummary),
        ("EmbeddingTraceSummary", EmbeddingTraceSummary),
        ("SidecarApplied", SidecarApplied),
        ("OpenTelemetrySink", OpenTelemetrySink),
    ],
)
def test_events_symbols_live_under_events_namespace(symbol_name: str, expected: object) -> None:
    assert symbol_name not in rag_core.__all__
    assert getattr(rag_core.events, symbol_name) is expected


@pytest.mark.meta
def test_trace_summarizers_live_under_events_namespace() -> None:
    assert SearchTraceSummary is EventsSearchTraceSummary
    assert EmbeddingTraceSummary is EventsEmbeddingTraceSummary
    assert summarize_search_trace is rag_core.events.summarize_search_trace
    assert summarize_embedding_trace is rag_core.events.summarize_embedding_trace


def test_search_trace_summary_payload_round_trip() -> None:
    assert (
        SearchStageTraceSummary(stage="retrieve", stage_name="HybridRetrieve").stage
        == "retrieve"
    )
    summary = summarize_search_trace([])
    assert summary.to_payload()["result_count"] == 0


def test_event_types_are_not_root_exports() -> None:
    expected = {
        "EventSink",
        "EventBuffer",
        "JsonlSink",
        "IngestStarted",
        "IngestBatchStarted",
        "IngestBatchProgress",
        "IngestBatchCompleted",
        "IngestBatchFailed",
        "SearchCompleted",
        "SearchPlanned",
        "SearchStageCompleted",
        "StageError",
    }
    assert expected.isdisjoint(set(rag_core.__all__))


def test_embedding_trace_summary_payload_round_trip() -> None:
    summary = summarize_embedding_trace(
        [
            EmbedCompleted(
                provider="openai",
                model="text-embedding-3-small",
                text_count=2,
                role="dense",
                cache_hits=1,
                cache_misses=1,
            )
        ]
    )

    assert summary.to_payload()["cache_hits"] == 1
