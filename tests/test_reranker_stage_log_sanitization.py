from __future__ import annotations

import asyncio
import logging

import pytest

from rag_core.events.sinks import EventBuffer
from rag_core.events.types import RerankApplied
from rag_core.search.pipeline import PipelineContext, PipelineQuery, ProviderRerankStage
from rag_core.search.types import RerankBudget
from tests.support import (
    FakeEmbeddingProvider,
    FakeReranker,
    FakeSparseEmbedder,
    RecordingVectorStore,
    assert_caplog_omits_private,
    make_search_result,
)


class _NamedFailingReranker(FakeReranker):
    provider_name = "fake-reranker"


def test_rerank_fallback_warning_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def run() -> tuple[EventBuffer, list[str]]:
        hits = [
            make_search_result(id="a", text="alpha"),
            make_search_result(id="b", text="beta"),
        ]
        events = EventBuffer()
        ctx = PipelineContext(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(),
            reranker=_NamedFailingReranker(
                error=RuntimeError("raw provider detail with api key sk-test-secret")
            ),
            event_sink=events,
        )
        result = await ProviderRerankStage().rerank(
            hits,
            PipelineQuery(
                query="private rerank query",
                namespace="space-1",
                corpus_ids=["corpus-1"],
                limit=2,
                rerank_budget=RerankBudget(fallback_on_error=True),
            ),
            ctx,
        )
        return events, [hit.id for hit in result]

    with caplog.at_level(
        logging.WARNING, logger="rag_core.search.pipeline.stages.reranker_stage"
    ):
        events, result_ids = asyncio.run(run())

    applied = [event for event in events.events if isinstance(event, RerankApplied)]
    assert result_ids == ["a", "b"]
    assert len(applied) == 1
    assert applied[0].succeeded is False
    assert applied[0].fallback_reason == "RuntimeError"
    assert "fake-reranker" in caplog.text
    assert "RuntimeError" in caplog.text
    assert_caplog_omits_private(caplog, "raw provider detail")
