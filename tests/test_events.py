"""Tests for the event sink seam.

Covers:
- Default Engine behavior (NoOpSink) is observably silent.
- EventBuffer captures expected events on a basic ingest+search cycle.
- Event records carry expected fields.
- MultiSink fans out; JsonlSink round-trips; LoggingSink logs.
- Sink errors do NOT break ingest or search.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import pytest

import rag_core.documents.local_parse as local_parse_module
from rag_core.core import Engine
from rag_core.core_models import IngestedDocument
from rag_core.events import (
    EventBuffer,
    EventSink,
    JsonlSink,
    LoggingSink,
    MultiSink,
    NoOpSink,
)
from rag_core.events.types import (
    ChunkProduced,
    EmbedCompleted,
    EmbedRequested,
    Event,
    IndexDeleted,
    IndexUpserted,
    IngestBatchCompleted,
    IngestBatchFailed,
    IngestBatchProgress,
    IngestBatchStarted,
    IngestCompleted,
    IngestSkipped,
    IngestStarted,
    ParseCompleted,
    SearchCompleted,
    SearchStarted,
    StageError,
)
from rag_core._engine.core_prepare import parse_document_bytes
from rag_core.search.runner import (
    SearchRunner,
    SearchRequest,
)
from rag_core.search.request_models import RerankBudget

from tests.support import (
    make_test_config,
    FakeEmbeddingProvider,
    FakeReranker,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
)


_DENSE_PRIMARY_CHANNEL = "dense:dense:primary"


async def _ingest_hello(
    core: Engine,
    *,
    document_id: str | None = None,
) -> IngestedDocument:
    """Drive a basic ingest used by most lifecycle tests."""
    try:
        return await core.add_bytes(
            file_bytes=b"hello world",
            filename="a.txt",
            mime_type="text/plain",
            tenant_id="ns",
            collection="corp",
            document_id=document_id,
        )
    finally:
        await core.close()


def test_default_core_uses_no_op_sink() -> None:
    core = _make_core(event_sink=None)
    assert core._event_sink is None


def test_no_op_sink_does_nothing() -> None:
    NoOpSink().emit(IngestStarted(filename="a.txt"))


def test_event_buffer_collects_filters_and_clears() -> None:
    buffer = EventBuffer()
    buffer.emit(IngestStarted(filename="a.txt"))
    buffer.emit(IngestStarted(filename="b.txt"))
    buffer.emit(SearchStarted(query_length=4))

    assert [event.event_type for event in buffer.events] == [
        "ingest.started",
        "ingest.started",
        "search.started",
    ]
    assert [event.event_type for event in buffer.by_type("ingest.started")] == [
        "ingest.started",
        "ingest.started",
    ]

    buffer.clear()
    assert buffer.events == []


def test_multi_sink_fans_out_and_swallows_errors() -> None:
    class ExplodingSink:
        def emit(self, event: Event) -> None:
            raise RuntimeError("boom")

    a = EventBuffer()
    b = EventBuffer()
    multi = MultiSink(ExplodingSink(), a, b)
    multi.emit(IngestStarted(filename="a.txt"))

    assert len(a.events) == 1
    assert len(b.events) == 1
    assert multi.failure_count == 1
    assert a.failure_count == 0


def test_multi_sink_counts_failures_swallowed_by_builtin_child_sinks() -> None:
    class ExplodingLogger(logging.Logger):
        def handle(self, record: logging.LogRecord) -> None:
            raise RuntimeError("log write failed")

    sink = LoggingSink(logger=ExplodingLogger("test"))
    multi = MultiSink(sink)

    multi.emit(IngestStarted(filename="a.txt"))

    assert sink.failure_count == 1
    assert multi.failure_count == 1


def test_jsonl_sink_round_trips_ingest_and_batch_events(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    sink = JsonlSink(log)
    sink.emit(
        IngestStarted(
            tenant_id="ns", collection="corp", document_id="doc-1", filename="a.txt"
        )
    )
    sink.emit(
        EmbedCompleted(
            provider="fake",
            model="fake-embedding",
            text_count=2,
            role="dense",
        )
    )
    sink.emit(IngestCompleted(document_id="doc-1", chunk_count=3))
    sink.emit(IngestBatchStarted(tenant_id="ns", collection="corp", planned_count=2))
    sink.emit(
        IngestBatchProgress(
            tenant_id="ns",
            collection="corp",
            planned_count=2,
            completed_count=1,
            succeeded_count=1,
            failed_count=0,
            current_index=1,
            filename="a.txt",
            document_key="/docs/a.txt",
            content_sha256="sha-a",
            status="succeeded",
            ingest_state="created",
        )
    )
    sink.emit(
        IngestBatchCompleted(
            tenant_id="ns",
            collection="corp",
            planned_count=2,
            succeeded_count=1,
            failed_count=1,
            duration_ms=12.5,
        )
    )
    sink.emit(
        IngestBatchFailed(
            tenant_id="ns",
            collection="corp",
            planned_count=2,
            completed_count=1,
            succeeded_count=1,
            failed_count=0,
            duration_ms=13.0,
            error="vector store unavailable",
        )
    )

    payloads = [
        json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
    ]
    assert [payload["event_type"] for payload in payloads] == [
        "ingest.started",
        "embed.completed",
        "ingest.completed",
        "ingest.batch.started",
        "ingest.batch.progress",
        "ingest.batch.completed",
        "ingest.batch.failed",
    ]
    assert "document_id" not in payloads[0]
    assert "filename" not in payloads[0]
    assert payloads[1]["text_count"] == 2
    assert payloads[2]["chunk_count"] == 3
    progress = payloads[4]
    assert progress["status"] == "succeeded"
    assert "content_sha256" not in progress
    assert "document_key" not in progress
    assert payloads[5]["failed_count"] == 1
    assert "error" not in payloads[6]


def test_jsonl_sink_creates_parent_directories(tmp_path: Path) -> None:
    log = tmp_path / "traces" / "events.jsonl"
    JsonlSink(log).emit(IngestStarted(filename="a.txt"))

    assert json.loads(log.read_text(encoding="utf-8"))["event_type"] == "ingest.started"
    if os.name != "nt":
        assert log.stat().st_mode & 0o777 == 0o600
        assert (tmp_path / "traces").stat().st_mode & 0o777 == 0o700


def test_jsonl_sink_rejects_symlink_path(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are unavailable on this platform")
    target = tmp_path / "target.jsonl"
    target.write_text("", encoding="utf-8")
    link = tmp_path / "events.jsonl"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        JsonlSink(link)


def test_jsonl_sink_serializes_tuple_collections_as_list(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    JsonlSink(log).emit(SearchStarted(collections=("a", "b")))

    assert "collections" not in json.loads(log.read_text(encoding="utf-8"))


def test_jsonl_sink_counts_emit_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_payload(event: Event) -> dict[str, object]:
        raise RuntimeError("payload failed")

    sink = JsonlSink(tmp_path / "events.jsonl")
    monkeypatch.setattr("rag_core.events.sinks.event_to_jsonl_dict", fail_payload)

    sink.emit(IngestStarted(filename="a.txt"))

    assert sink.failure_count == 1
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8") == ""


def test_logging_sink_emits_event_type_at_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = LoggingSink()
    with caplog.at_level(logging.INFO, logger="rag_core.events"):
        sink.emit(IngestStarted(filename="a.txt"))

    assert any(record.message.startswith("ingest.started") for record in caplog.records)


def test_logging_sink_swallows_logger_errors() -> None:
    class BrokenLogger:
        def log(self, level: int, msg: str, *args: object) -> None:
            raise RuntimeError("logging failed")

    sink = LoggingSink(logger=cast(Any, BrokenLogger()))
    sink.emit(IngestStarted())

    assert sink.failure_count == 1


def test_failing_sink_does_not_break_ingest() -> None:
    class ExplodingSink:
        def emit(self, event: Event) -> None:
            raise RuntimeError("boom")

    async def scenario() -> str:
        core = _make_core(event_sink=ExplodingSink())
        try:
            document = await core.add_bytes(
                file_bytes=b"hello",
                filename="a.txt",
                mime_type="text/plain",
                tenant_id="ns",
                collection="corp",
            )
        finally:
            await core.close()
        return document.document_id

    assert asyncio.run(scenario())


def test_basic_ingest_emits_full_lifecycle_with_identifying_fields() -> None:
    async def scenario() -> tuple[EventBuffer, IngestedDocument]:
        buffer = EventBuffer()
        core = _make_core(event_sink=buffer)
        result = await _ingest_hello(core)
        return buffer, result

    buffer, result = asyncio.run(scenario())
    types = [event.event_type for event in buffer.events]
    assert {
        "ingest.started",
        "parse.completed",
        "chunk.produced",
        "embed.requested",
        "embed.completed",
        "index.upserted",
        "ingest.completed",
    }.issubset(types)

    [started] = [event for event in buffer.events if isinstance(event, IngestStarted)]
    assert started.tenant_id == "ns"
    assert started.collection == "corp"
    assert started.filename == "a.txt"
    assert started.mime_type == "text/plain"
    assert started.content_sha256

    [completed] = [
        event for event in buffer.events if isinstance(event, IngestCompleted)
    ]
    assert completed.chunk_count == result.chunk_count
    assert completed.duration_ms >= 0.0

    [chunk_event] = [
        event for event in buffer.events if isinstance(event, ChunkProduced)
    ]
    assert chunk_event.chunk_count >= 1
    assert chunk_event.chunking_strategy

    [upsert] = [event for event in buffer.events if isinstance(event, IndexUpserted)]
    assert upsert.point_count == result.chunk_count
    assert upsert.tenant_id == "ns"


def test_parse_file_bytes_records_structured_quality_metadata() -> None:
    _, metadata = asyncio.run(
        local_parse_module.parse_file_bytes(
            file_bytes=b"hello world",
            filename="a.txt",
            mime_type="text/plain",
        )
    )

    assert "quality_verdict" not in metadata
    assert metadata["quality"] == {
        "verdict": "poor",
        "details": "below minimum char count (11 < 50)",
        "char_count": 11,
        "meaningful_ratio": 1.0,
        "mojibake_ratio": 0.0,
        "text_to_page_ratio": 11.0,
        "page_count": 1,
    }


def test_parse_completed_records_parser_and_quality_diagnostics() -> None:
    async def scenario() -> EventBuffer:
        buffer = EventBuffer()
        await _ingest_hello(_make_core(event_sink=buffer))
        return buffer

    buffer = asyncio.run(scenario())
    [event] = [e for e in buffer.events if isinstance(e, ParseCompleted)]
    assert event.filename == "a.txt"
    assert event.mime_type == "text/plain"
    assert event.parser
    assert event.quality_verdict == "poor"
    assert event.quality_details == "below minimum char count (11 < 50)"
    assert event.char_count == 11
    assert event.page_count == 1
    assert event.ocr_page_count == 0
    assert event.ocr_page_indices == ()
    assert event.extraction_ratio is None


def test_parse_completed_records_ocr_routing_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_parse_file_bytes(
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> tuple[str, dict[str, object]]:
        return (
            "mixed pdf text",
            {
                "parser": "local:pdf_inspector",
                "needs_ocr": True,
                "page_count": 4,
                "ocr_page_count": 2,
                "ocr_page_indices": [2, 0, 2],
                "extraction_ratio": 0.5,
                "quality": {
                    "verdict": "poor",
                    "details": "mixed extraction needs OCR",
                    "char_count": 14,
                    "meaningful_ratio": 1.0,
                    "mojibake_ratio": 0.0,
                    "text_to_page_ratio": 3.5,
                    "page_count": 4,
                },
            },
        )

    monkeypatch.setattr(local_parse_module, "parse_file_bytes", fake_parse_file_bytes)

    async def scenario() -> EventBuffer:
        buffer = EventBuffer()
        await parse_document_bytes(
            file_bytes=b"%PDF-1.7",
            filename="mixed.pdf",
            mime_type="application/pdf",
            event_sink=buffer,
        )
        return buffer

    buffer = asyncio.run(scenario())
    [event] = [e for e in buffer.events if isinstance(e, ParseCompleted)]
    assert event.parser == "local:pdf_inspector"
    assert event.needs_ocr is True
    assert event.ocr_page_count == 2
    # Sorted, deduplicated indices: routing trace must be canonical.
    assert event.ocr_page_indices == (0, 2)
    assert event.extraction_ratio == 0.5


def test_parse_completed_uses_capped_ocr_page_indices_for_event_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_parse_file_bytes(
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> tuple[str, dict[str, object]]:
        return (
            "image-only pdf",
            {
                "parser": "local:pdf_inspector",
                "needs_ocr": True,
                "page_count": 450,
                "ocr_page_count": 450,
                "ocr_page_indices": list(range(450)),
                "ocr_page_indices_telemetry": list(range(400)),
            },
        )

    monkeypatch.setattr(local_parse_module, "parse_file_bytes", fake_parse_file_bytes)

    async def scenario() -> EventBuffer:
        buffer = EventBuffer()
        await parse_document_bytes(
            file_bytes=b"%PDF-1.7",
            filename="scan.pdf",
            mime_type="application/pdf",
            event_sink=buffer,
        )
        return buffer

    buffer = asyncio.run(scenario())
    [event] = [e for e in buffer.events if isinstance(e, ParseCompleted)]
    assert event.ocr_page_count == 450
    assert event.ocr_page_indices == tuple(range(400))


def test_embed_events_pair_request_and_complete_per_role() -> None:
    async def scenario() -> EventBuffer:
        buffer = EventBuffer()
        await _ingest_hello(_make_core(event_sink=buffer))
        return buffer

    buffer = asyncio.run(scenario())
    requested = [e for e in buffer.events if isinstance(e, EmbedRequested)]
    completed = [e for e in buffer.events if isinstance(e, EmbedCompleted)]

    assert len(requested) == len(completed)
    roles = {event.role for event in requested}
    assert {"dense", "sparse"}.issubset(roles)


def test_skipped_ingest_emits_ingest_skipped_and_no_completed() -> None:
    async def scenario() -> EventBuffer:
        buffer = EventBuffer()
        core = _make_core(event_sink=buffer, store=RecordingVectorStore())
        try:
            await core.add_bytes(
                file_bytes=b"hello world",
                filename="a.txt",
                mime_type="text/plain",
                tenant_id="ns",
                collection="corp",
            )
            buffer.clear()
            await core.add_bytes(
                file_bytes=b"hello world",
                filename="a.txt",
                mime_type="text/plain",
                tenant_id="ns",
                collection="corp",
            )
        finally:
            await core.close()
        return buffer

    buffer = asyncio.run(scenario())
    [skipped] = [e for e in buffer.events if isinstance(e, IngestSkipped)]
    assert skipped.reason == "content_unchanged"
    assert [e for e in buffer.events if isinstance(e, IngestCompleted)] == []


def test_delete_document_emits_index_deleted() -> None:
    async def scenario() -> tuple[EventBuffer, str]:
        buffer = EventBuffer()
        core = _make_core(event_sink=buffer)
        try:
            result = await core.add_bytes(
                file_bytes=b"hello world",
                filename="a.txt",
                mime_type="text/plain",
                tenant_id="ns",
                collection="corp",
            )
            buffer.clear()
            await core.delete_document(
                document_id=result.document_id,
                tenant_id="ns",
                collection="corp",
            )
        finally:
            await core.close()
        return buffer, result.document_id

    buffer, document_id = asyncio.run(scenario())
    [deleted] = [e for e in buffer.events if isinstance(e, IndexDeleted)]
    assert deleted.document_id == document_id


def test_search_emits_started_and_completed_with_scope() -> None:
    async def scenario() -> EventBuffer:
        buffer = EventBuffer()
        store = RecordingVectorStore(search_results=[make_search_result(id="hit-1")])
        core = _make_core(event_sink=buffer, store=store)
        try:
            await core.search(
                query="fox",
                tenant_id="ns",
                collections=["corp"],
                limit=5,
                rerank=False,
            )
        finally:
            await core.close()
        return buffer

    buffer = asyncio.run(scenario())
    [started] = [e for e in buffer.events if isinstance(e, SearchStarted)]
    [completed] = [e for e in buffer.events if isinstance(e, SearchCompleted)]
    assert started.tenant_id == "ns"
    assert started.collections == ("corp",)
    assert started.limit == 5
    assert completed.result_count == 1
    assert completed.applied_rerank is False


def test_search_completed_distinguishes_requested_attempted_and_applied_rerank() -> (
    None
):
    async def scenario() -> SearchCompleted:
        buffer = EventBuffer()
        pipeline_runner = SearchRunner(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(
                search_results=[make_search_result(id="hit-1")]
            ),
            reranker=FakeReranker(error=RuntimeError("rerank failed")),
            event_sink=buffer,
        )
        await pipeline_runner.search(
            SearchRequest(
                query="fox",
                tenant_id="ns",
                collections=["corp"],
                limit=5,
                rerank=True,
                rerank_budget=RerankBudget(fallback_on_error=True),
            )
        )
        [completed] = [
            event for event in buffer.events if isinstance(event, SearchCompleted)
        ]
        return completed

    completed = asyncio.run(scenario())
    assert completed.requested_rerank is True
    assert completed.attempted_rerank is True
    assert completed.applied_rerank is False


def test_stage_error_event_emitted_on_failure() -> None:
    from typing import Sequence

    from rag_core.search.vector_models import VectorPoint

    class FailingStore(RecordingVectorStore):
        async def upsert(self, points: Sequence[VectorPoint]) -> None:
            raise RuntimeError("disk full")

    async def scenario() -> EventBuffer:
        buffer = EventBuffer()
        core = _make_core(event_sink=buffer, store=FailingStore())
        try:
            with pytest.raises(RuntimeError):
                await core.add_bytes(
                    file_bytes=b"hello world",
                    filename="a.txt",
                    mime_type="text/plain",
                    tenant_id="ns",
                    collection="corp",
                )
        finally:
            await core.close()
        return buffer

    buffer = asyncio.run(scenario())
    errors = [e for e in buffer.events if isinstance(e, StageError)]
    assert [error.stage for error in errors] == ["index", "ingest"]
    assert [error.error_type for error in errors] == ["RuntimeError", "RuntimeError"]
    assert all(error.message == "" for error in errors)
    assert "disk full" not in str(errors)


def test_event_sink_protocol_is_runtime_checkable() -> None:
    assert isinstance(NoOpSink(), EventSink)
    assert isinstance(EventBuffer(), EventSink)


def _make_core(
    *,
    event_sink: EventSink | None,
    store: RecordingVectorStore | None = None,
) -> Engine:
    return Engine(
        make_test_config(
            qdrant_collection="rag_core_events_tests",
            embedding_dimensions=4,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store or RecordingVectorStore(),
        event_sink=event_sink,
    )
