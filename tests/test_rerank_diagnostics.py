from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

from rag_core.events.sinks import EventBuffer
from rag_core.events.sinks import JsonlSink
from rag_core.events.types import RerankApplied
from rag_core.search.pipeline.stages.reranker_stage import ProviderRerankStage
from rag_core.search.pipeline.types import PipelineContext, PipelineQuery
from rag_core.search.providers.rerank_results import ValidatedRerankResults
from rag_core.search.types import RerankerProvider, RerankResult, SearchResult


class _StaticReranker:
    provider_name = "test-reranker"
    model_name = "test-model"

    def __init__(self, results: list[RerankResult]) -> None:
        self._results = results

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RerankResult]:
        return list(self._results)


class _ValidatedReranker:
    provider_name = "validated-reranker"
    model_name = "validated-model"

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RerankResult]:
        return ValidatedRerankResults(
            [RerankResult(index=0, score=0.9, text=documents[0])],
            provider_result_count=3,
        )


def test_rerank_trace_counts_provider_results_and_drops() -> None:
    hits = [_result("a"), _result("b")]
    result, applied = _run_rerank(
        hits,
        _StaticReranker(
            [
                RerankResult(index=99, score=0.99, text="invalid"),
                RerankResult(index=1, score=0.98, text=hits[1].text),
                RerankResult(index=1, score=0.97, text=hits[1].text),
            ]
        ),
    )

    assert [hit.id for hit in result] == ["b", "a"]
    assert applied.provider_result_count == 3
    assert applied.accepted_count == 1
    assert applied.dropped_count == 2
    assert applied.succeeded is False
    assert "private query text" not in str(applied)


def test_rerank_trace_preserves_adapter_raw_provider_result_count() -> None:
    hits = [_result("a"), _result("b")]
    result, applied = _run_rerank(hits, _ValidatedReranker())

    assert [hit.id for hit in result] == ["a", "b"]
    assert applied.provider_result_count == 3
    assert applied.accepted_count == 1
    assert applied.dropped_count == 2
    assert applied.succeeded is False


def test_rerank_trace_counts_empty_provider_success() -> None:
    hits = [_result("a"), _result("b")]
    result, applied = _run_rerank(hits, _StaticReranker([]))

    assert [hit.id for hit in result] == ["a", "b"]
    assert applied.provider_result_count == 0
    assert applied.accepted_count == 0
    assert applied.dropped_count == 0
    assert applied.succeeded is True


def test_rerank_trace_counts_all_invalid_provider_results() -> None:
    hits = [_result("a"), _result("b")]
    result, applied = _run_rerank(
        hits,
        _StaticReranker(
            [
                RerankResult(index=99, score=0.99, text="invalid"),
                RerankResult(index=-1, score=0.98, text="invalid"),
            ]
        ),
    )

    assert [hit.id for hit in result] == ["a", "b"]
    assert applied.provider_result_count == 2
    assert applied.accepted_count == 0
    assert applied.dropped_count == 2
    assert applied.succeeded is False


def test_rerank_trace_reports_rank_movement_and_score_ranges() -> None:
    hits = [
        _result("a", score=0.2),
        _result("b", score=0.5),
        _result("c", score=0.9),
    ]
    result, applied = _run_rerank(
        hits,
        _StaticReranker(
            [
                RerankResult(index=2, score=0.91, text=hits[2].text),
                RerankResult(index=0, score=0.72, text=hits[0].text),
                RerankResult(index=1, score=0.12, text=hits[1].text),
            ]
        ),
    )

    assert [hit.id for hit in result] == ["c", "a", "b"]
    assert [hit.score for hit in result] == [0.9, 0.2, 0.5]
    assert applied.provider_result_count == 3
    assert applied.accepted_count == 3
    assert applied.dropped_count == 0
    assert applied.succeeded is True
    assert applied.rank_changed_count == 3
    assert applied.rank_promoted_count == 1
    assert applied.rank_demoted_count == 2
    assert applied.max_rank_gain == 2
    assert applied.max_rank_loss == 1
    assert applied.provider_score_min == 0.12
    assert applied.provider_score_max == 0.91
    assert applied.search_score_min == 0.2
    assert applied.search_score_max == 0.9
    rerank = result[0].metadata["rerank"]
    assert isinstance(rerank, dict)
    assert rerank["rank_delta"] == 2
    assert rerank["provider_score"] == 0.91
    assert rerank["search_score"] == 0.9


def test_rerank_trace_serializes_provider_result_counters(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    hits = [_result("a"), _result("b")]
    result = asyncio.run(
        ProviderRerankStage().rerank(
            hits,
            _query(),
            _context(
                reranker=_StaticReranker(
                    [
                        RerankResult(index=99, score=0.99, text="text a"),
                        RerankResult(index=0, score=0.98, text="text a"),
                    ]
                ),
                event_sink=JsonlSink(path),
            ),
        )
    )

    assert [hit.id for hit in result] == ["a", "b"]
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["event_type"] == "rerank.applied"
    assert payload["provider_result_count"] == 2
    assert payload["accepted_count"] == 1
    assert payload["dropped_count"] == 1
    assert payload["rank_changed_count"] == 0
    assert payload["rank_promoted_count"] == 0
    assert payload["rank_demoted_count"] == 0
    assert payload["max_rank_gain"] == 0
    assert payload["max_rank_loss"] == 0
    assert payload["provider_score_min"] == 0.98
    assert payload["provider_score_max"] == 0.98
    assert payload["search_score_min"] == 1.0
    assert payload["search_score_max"] == 1.0
    serialized = json.dumps(payload)
    assert "private query text" not in serialized
    assert "text a" not in serialized


def test_rerank_trace_preserves_existing_positional_constructor_shape() -> None:
    event = RerankApplied(
        "provider",
        "model",
        2,
        1,
        2,
        10,
        "fallback",
        "candidate_budget",
        12.5,
        False,
    )

    assert event.provider == "provider"
    assert event.model == "model"
    assert event.input_count == 2
    assert event.candidate_count == 1
    assert event.result_count == 2
    assert event.top_k == 10
    assert event.fallback_reason == "fallback"
    assert event.truncation_reason == "candidate_budget"
    assert event.duration_ms == 12.5
    assert event.succeeded is False
    assert event.provider_result_count == 0
    assert event.accepted_count == 0
    assert event.dropped_count == 0
    assert event.rank_changed_count == 0
    assert event.rank_promoted_count == 0
    assert event.rank_demoted_count == 0
    assert event.max_rank_gain == 0
    assert event.max_rank_loss == 0
    assert event.provider_score_min == 0.0
    assert event.provider_score_max == 0.0
    assert event.search_score_min == 0.0
    assert event.search_score_max == 0.0

    event_with_type = RerankApplied(
        "provider",
        "model",
        2,
        1,
        2,
        10,
        "fallback",
        "candidate_budget",
        12.5,
        False,
        "rerank.applied",
    )
    assert event_with_type.event_type == "rerank.applied"
    assert event_with_type.provider_result_count == 0
    assert event_with_type.accepted_count == 0
    assert event_with_type.dropped_count == 0
    assert event_with_type.rank_changed_count == 0
    assert event_with_type.rank_promoted_count == 0
    assert event_with_type.rank_demoted_count == 0
    assert event_with_type.max_rank_gain == 0
    assert event_with_type.max_rank_loss == 0
    assert event_with_type.provider_score_min == 0.0
    assert event_with_type.provider_score_max == 0.0
    assert event_with_type.search_score_min == 0.0
    assert event_with_type.search_score_max == 0.0


def _run_rerank(
    hits: list[SearchResult],
    reranker: RerankerProvider,
) -> tuple[list[SearchResult], RerankApplied]:
    events = EventBuffer()
    result = asyncio.run(
        ProviderRerankStage().rerank(
            hits,
            _query(),
            _context(reranker=reranker, event_sink=events),
        )
    )
    applied = [event for event in events.events if isinstance(event, RerankApplied)]
    assert len(applied) == 1
    return result, applied[0]


def _query() -> PipelineQuery:
    return PipelineQuery(
        query="private query text",
        namespace="ns",
        corpus_ids=["corpus"],
        rerank=True,
    )


def _context(
    *,
    reranker: RerankerProvider,
    event_sink: object,
) -> PipelineContext:
    return PipelineContext(
        embedding_provider=cast(Any, object()),
        sparse_embedder=cast(Any, object()),
        vector_store=cast(Any, object()),
        reranker=reranker,
        event_sink=event_sink,
    )


def _result(result_id: str, *, score: float = 1.0) -> SearchResult:
    return SearchResult(
        id=result_id,
        text=f"text {result_id}",
        score=score,
        content_type="document",
        source_type="file",
    )
