from __future__ import annotations

import rag_core
from rag_core.events import (
    EmbedCompleted,
    EmbeddingTraceSummary,
    SearchStageTraceSummary,
    SearchTraceSummary,
    summarize_embedding_trace,
    summarize_embedding_trace_payloads,
    summarize_search_trace,
    summarize_search_trace_payloads,
)
from rag_core.events import EmbeddingTraceSummary as EventsEmbeddingTraceSummary
from rag_core.events import summarize_embedding_trace as events_summarize_embedding_trace
from rag_core.events import (
    summarize_embedding_trace_payloads as events_summarize_embedding_trace_payloads,
)
from rag_core.events import SearchTraceSummary as EventsSearchTraceSummary
from rag_core.events import summarize_search_trace as events_summarize_search_trace
from rag_core.events import (
    summarize_search_trace_payloads as events_summarize_search_trace_payloads,
)


def test_search_trace_summary_lives_under_events_namespace() -> None:
    assert SearchTraceSummary is EventsSearchTraceSummary
    assert summarize_search_trace is events_summarize_search_trace
    assert summarize_search_trace_payloads is events_summarize_search_trace_payloads
    assert (
        SearchStageTraceSummary(stage="retrieve", stage_name="HybridRetrieve").stage
        == "retrieve"
    )
    assert {
        "SearchStageTraceSummary",
        "SearchTraceSummary",
        "summarize_search_trace",
        "summarize_search_trace_payloads",
    }.isdisjoint(set(rag_core.__all__))


def test_embedding_trace_summary_lives_under_events_namespace() -> None:
    assert EmbeddingTraceSummary is EventsEmbeddingTraceSummary
    assert summarize_embedding_trace is events_summarize_embedding_trace
    assert (
        summarize_embedding_trace_payloads
        is events_summarize_embedding_trace_payloads
    )
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
    assert {
        "EmbeddingTraceSummary",
        "summarize_embedding_trace",
        "summarize_embedding_trace_payloads",
    }.isdisjoint(set(rag_core.__all__))
