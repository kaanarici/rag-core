from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

import pytest

from rag_core import ContextPack, RAGCore
from rag_core.events import EventBuffer, summarize_search_trace_payload_runs
from rag_core._engine.core_retrieval import retrieve_context_with_core
from rag_core.events.types import SearchStageCompleted, StageError
from rag_core.search.pipeline_runner import SearchRequest, SearchRunResult

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
    make_test_config,
)


def test_retrieve_context_trace_includes_safe_context_pack_summary() -> None:
    async def run() -> tuple[SearchStageCompleted, ContextPack, list[Any]]:
        events = EventBuffer()
        core = RAGCore(
            make_test_config(
                qdrant_collection="rag_core_context_pack_trace_summary",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(
                search_results=[
                    make_search_result(id="hit-1", document_id="doc-1", text="billing " * 18),
                    make_search_result(id="hit-2", document_id="doc-2", text="refund workflow"),
                ]
            ),
            event_sink=events,
        )
        try:
            pack = await core.retrieve_context(
                query="billing refund",
                namespace="acme",
                corpus_ids=["help"],
                limit=2,
                rerank=False,
                max_chars=72,
                max_tokens=100,
            )
        finally:
            await core.close()
        context_events = [
            event
            for event in events.events
            if isinstance(event, SearchStageCompleted) and event.stage == "context_pack"
        ]
        [event] = context_events
        return event, pack, list(events.events)

    event, pack, trace_events = asyncio.run(run())

    assert event.stage_name == "build_context_pack"
    assert event.candidate_count == 2
    assert event.result_count == len(pack.snippets)
    assert event.dropped_count == pack.dropped_count
    assert event.truncated is pack.truncated
    assert event.max_chars == pack.max_chars
    assert event.max_tokens == pack.max_tokens
    assert event.token_estimate == pack.token_estimate
    assert event.char_count == pack.char_count
    assert event.citation_count == len(pack.citations)
    assert event.source_preview_count == len(pack.source_previews)
    assert event.duration_ms >= 0.0
    assert event.search_id

    event_payload = asdict(event)
    assert "query" not in event_payload
    assert "text" not in event_payload
    assert "snippets" not in event_payload
    assert "citations" not in event_payload
    rendered_event = repr(event_payload)
    assert "billing refund" not in rendered_event
    assert "refund workflow" not in rendered_event
    assert "doc-1" not in rendered_event
    payloads = [asdict(event) for event in trace_events if hasattr(event, "search_id")]
    summaries = summarize_search_trace_payload_runs(payloads)
    [summary] = summaries
    assert summary.search_id == event.search_id
    assert any(stage.stage == "context_pack" for stage in summary.stages)


def test_context_pack_failure_emits_sanitized_terminal_stage_error() -> None:
    class _SearchWithTrace:
        async def search(self, req: SearchRequest) -> list[Any]:
            del req
            return []

        async def search_with_trace(self, req: SearchRequest) -> SearchRunResult:
            del req
            return SearchRunResult(results=[], search_id="tenant/a")

    async def run() -> list[Any]:
        events = EventBuffer()
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "rag_core._engine.core_retrieval.build_context_pack",
                lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
            )
            with pytest.raises(RuntimeError, match="boom"):
                await retrieve_context_with_core(
                    search=_SearchWithTrace(),
                    event_sink=events,
                    query="billing refund",
                    namespace="acme",
                    corpus_ids=["help"],
                )
        return list(events.events)

    trace_events = asyncio.run(run())
    stage_errors = [event for event in trace_events if isinstance(event, StageError)]
    [event] = stage_errors
    assert event.stage == "context_pack"
    assert event.error_type == "RuntimeError"
    assert event.message == ""
    assert event.search_id == ""
