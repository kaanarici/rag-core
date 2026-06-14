"""Audit correlation contract for the event surface.

What this proves:
- Every event union member carries the audit fields (``emitted_at_ns``,
  ``wall_clock_ns``, ``actor``, ``request_id``, ``ingest_id``).
- ``emit_event`` stamps ``emitted_at_ns``/``wall_clock_ns`` if the emitter
  left them at the default ``0``.
- ``AUDIT_EVENT_TYPES`` is the canonical tier-crossing subset; non-tier
  pipeline-internal events stay out so audit sinks can filter cleanly.
- ``SearchCompleted`` surfaces ``corpus_ids`` and a capped
  ``returned_document_ids`` for the audit trail.
- The pipeline runner threads caller-supplied ``AuditContext`` onto every
  emitted event via ``_SearchCorrelationSink``.
- The ingest correlation sink mirrors the search pattern.
- The HTTP runtime pulls ``X-Request-Id``/``X-Actor``/``X-Ingest-Id`` off the
  request headers and threads them into search and ingest events.

Label: contract.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import get_args

from rag_core.events.emit import emit_event
from rag_core.events.event_types import (
    AUDIT_EVENT_TYPES,
    FETCH_COMPLETED_EVENT,
    FETCH_FAILED_EVENT,
    FETCH_STARTED_EVENT,
    INDEX_DELETED_EVENT,
    INDEX_UPSERTED_EVENT,
    INGEST_COMPLETED_EVENT,
    INGEST_STARTED_EVENT,
    SEARCH_COMPLETED_EVENT,
    SEARCH_STARTED_EVENT,
)
from rag_core.events.search_events import RETURNED_DOCUMENT_IDS_CAP
from rag_core.events.sinks import EventBuffer
from rag_core.events.types import (
    AuditContext,
    Event,
    IngestCompleted,
    IngestStarted,
    SearchCompleted,
    SearchStarted,
)
from rag_core._engine.core_ingest_events import (
    _IngestCorrelationSink,
    maybe_wrap_with_ingest_correlation,
)
from rag_core.search.pipeline_runner import SearchPipelineRunner, SearchRequest
from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
)


_AUDIT_FIELDS = ("emitted_at_ns", "wall_clock_ns", "actor", "request_id", "ingest_id")


def _event_classes() -> tuple[type, ...]:
    classes: list[type] = []
    for arg in get_args(Event):
        if isinstance(arg, type) and dataclasses.is_dataclass(arg):
            classes.append(arg)
    return tuple(classes)


def test_every_event_union_member_carries_audit_fields() -> None:
    """Every event payload must carry the audit-correlation field set.

    Catches accidental schema drift if a new event class is added without the
    audit fields. The caller's audit log binds to this shape.
    """
    missing: dict[str, list[str]] = {}
    for klass in _event_classes():
        field_names = {f.name for f in dataclasses.fields(klass)}
        absent = [name for name in _AUDIT_FIELDS if name not in field_names]
        if absent:
            missing[klass.__name__] = absent
    assert missing == {}, f"event classes missing audit fields: {missing}"


def test_audit_event_types_covers_tier_crossing_subset() -> None:
    """AUDIT_EVENT_TYPES is the minimum a compliance audit consumer needs.

    Pipeline-internal events (chunk.produced, embed.requested, parse.completed)
    are explicitly excluded. They're debug telemetry, not audit material.
    """
    assert AUDIT_EVENT_TYPES == frozenset(
        {
            INGEST_STARTED_EVENT,
            INGEST_COMPLETED_EVENT,
            INDEX_UPSERTED_EVENT,
            INDEX_DELETED_EVENT,
            SEARCH_STARTED_EVENT,
            SEARCH_COMPLETED_EVENT,
            FETCH_STARTED_EVENT,
            FETCH_COMPLETED_EVENT,
            FETCH_FAILED_EVENT,
        }
    )
    # Pipeline-internal events explicitly excluded.
    assert "chunk.produced" not in AUDIT_EVENT_TYPES
    assert "embed.requested" not in AUDIT_EVENT_TYPES
    assert "parse.completed" not in AUDIT_EVENT_TYPES


def test_emit_event_stamps_emitted_at_ns_and_wall_clock_ns() -> None:
    """``emit_event`` fills in timestamps when the emitter passes 0.

    The dataclass default is ``0``; the emitter helper owns the clock so a
    caller cannot accidentally ship an event with a stale timestamp.
    """
    buffer = EventBuffer()
    emit_event(buffer, IngestStarted(namespace="ns", corpus_id="c", document_id="d"))
    [event] = buffer.events
    assert isinstance(event, IngestStarted)
    assert event.emitted_at_ns > 0
    assert event.wall_clock_ns > 0


def test_emit_event_preserves_caller_supplied_timestamps() -> None:
    """If an emitter set a non-zero timestamp (replay tests), don't clobber it."""
    buffer = EventBuffer()
    emit_event(
        buffer,
        IngestStarted(
            namespace="ns",
            corpus_id="c",
            document_id="d",
            emitted_at_ns=12345,
            wall_clock_ns=67890,
        ),
    )
    [event] = buffer.events
    assert isinstance(event, IngestStarted)
    assert event.emitted_at_ns == 12345
    assert event.wall_clock_ns == 67890


def test_search_completed_carries_corpus_ids_and_returned_document_ids() -> None:
    async def scenario() -> SearchCompleted:
        buffer = EventBuffer()
        store = RecordingVectorStore(
            search_results=[
                make_search_result(id=f"hit-{i}", document_id=f"doc-{i}")
                for i in range(3)
            ]
        )
        runner = SearchPipelineRunner(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            event_sink=buffer,
        )
        await runner.search(
            SearchRequest(
                query="hello",
                namespace="ns",
                corpus_ids=["public", "licensed"],
                limit=5,
            )
        )
        [completed] = [e for e in buffer.events if isinstance(e, SearchCompleted)]
        return completed

    completed = asyncio.run(scenario())
    assert completed.corpus_ids == ("public", "licensed")
    assert completed.returned_document_ids == ("doc-0", "doc-1", "doc-2")
    assert completed.result_count == 3


def test_search_completed_returned_document_ids_is_capped() -> None:
    """Audit lines must not be unbounded; ``returned_document_ids`` is capped."""
    async def scenario() -> SearchCompleted:
        buffer = EventBuffer()
        store = RecordingVectorStore(
            search_results=[
                make_search_result(id=f"hit-{i}", document_id=f"doc-{i}")
                for i in range(RETURNED_DOCUMENT_IDS_CAP + 5)
            ]
        )
        runner = SearchPipelineRunner(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            event_sink=buffer,
        )
        await runner.search(
            SearchRequest(
                query="hello",
                namespace="ns",
                corpus_ids=["public"],
                limit=RETURNED_DOCUMENT_IDS_CAP + 5,
            )
        )
        [completed] = [e for e in buffer.events if isinstance(e, SearchCompleted)]
        return completed

    completed = asyncio.run(scenario())
    assert len(completed.returned_document_ids) == RETURNED_DOCUMENT_IDS_CAP


def test_search_pipeline_threads_audit_context_onto_every_event() -> None:
    """``AuditContext`` on the request stamps actor/request_id/ingest_id on
    every event the runner emits."""
    async def scenario() -> list[Event]:
        buffer = EventBuffer()
        runner = SearchPipelineRunner(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(
                search_results=[make_search_result(id="hit-1", document_id="doc-1")]
            ),
            event_sink=buffer,
        )
        await runner.search(
            SearchRequest(
                query="hello",
                namespace="ns",
                corpus_ids=["public"],
                limit=5,
                audit_context=AuditContext(
                    actor="user-42",
                    request_id="req-7777",
                    ingest_id="ing-1234",
                ),
            )
        )
        return list(buffer.events)

    events = asyncio.run(scenario())
    [started] = [e for e in events if isinstance(e, SearchStarted)]
    [completed] = [e for e in events if isinstance(e, SearchCompleted)]
    for event in (started, completed):
        assert event.actor == "user-42"
        assert event.request_id == "req-7777"
        assert event.ingest_id == "ing-1234"
        # search_id is engine-minted, not caller-supplied.
        assert event.search_id != ""


def test_ingest_correlation_sink_stamps_ingest_id_and_audit_context() -> None:
    """Mirror of ``_SearchCorrelationSink`` for ingest events."""
    buffer = EventBuffer()
    sink = _IngestCorrelationSink(
        buffer,
        ingest_id="ing-9001",
        audit_context=AuditContext(actor="svc-loader", request_id="req-abc"),
    )
    sink.emit(IngestStarted(namespace="ns", corpus_id="public", document_id="d-1"))
    [event] = buffer.events
    assert isinstance(event, IngestStarted)
    assert event.ingest_id == "ing-9001"
    assert event.actor == "svc-loader"
    assert event.request_id == "req-abc"


def test_ingest_correlation_sink_does_not_overwrite_existing_values() -> None:
    """A nested wrap must not clobber correlation fields the inner sink already
    set. Supports composing search-within-ingest pipelines later."""
    buffer = EventBuffer()
    sink = _IngestCorrelationSink(
        buffer,
        ingest_id="ing-OUTER",
        audit_context=AuditContext(actor="outer-actor"),
    )
    sink.emit(
        IngestStarted(
            namespace="ns",
            corpus_id="public",
            document_id="d-1",
            ingest_id="ing-INNER",
            actor="inner-actor",
        )
    )
    [event] = buffer.events
    assert isinstance(event, IngestStarted)
    assert event.ingest_id == "ing-INNER"
    assert event.actor == "inner-actor"


def test_maybe_wrap_with_ingest_correlation_skips_wrap_when_no_context() -> None:
    """No audit context → return the raw sink so the hot path stays allocation-free."""
    buffer = EventBuffer()
    assert (
        maybe_wrap_with_ingest_correlation(buffer, ingest_id=None, audit_context=None)
        is buffer
    )


def test_core_ingest_bytes_threads_audit_context_onto_index_upserted() -> None:
    """The CoreIngestor wrap propagates audit context all the way through the
    indexer's IndexUpserted emit, not just the local ingest.started event."""
    from rag_core.core import RAGCore
    from rag_core.events.types import IndexUpserted, IngestCompleted
    from tests.support import make_test_config

    async def scenario() -> tuple[IngestStarted, IndexUpserted, IngestCompleted]:
        buffer = EventBuffer()
        core = RAGCore(
            make_test_config(
                qdrant_collection="rag_core_audit_ingest_tests",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(),
            event_sink=buffer,
        )
        try:
            await core.ingest_bytes(
                file_bytes=b"hello world",
                filename="a.txt",
                mime_type="text/plain",
                namespace="ns",
                corpus_id="public",
                ingest_id="batch-42-doc-3",
                audit_context=AuditContext(
                    actor="gateway",
                    request_id="req-abc",
                ),
            )
        finally:
            await core.close()
        [started] = [e for e in buffer.events if isinstance(e, IngestStarted)]
        [upsert] = [e for e in buffer.events if isinstance(e, IndexUpserted)]
        [completed] = [e for e in buffer.events if isinstance(e, IngestCompleted)]
        return started, upsert, completed

    started, upsert, completed = asyncio.run(scenario())
    for event in (started, upsert, completed):
        assert event.ingest_id == "batch-42-doc-3", event
        assert event.actor == "gateway", event
        assert event.request_id == "req-abc", event


def test_runtime_search_route_threads_x_request_id_header() -> None:
    """X-Request-Id flows through to ``SearchCompleted.request_id``."""
    from pathlib import Path
    import tempfile

    from starlette.testclient import TestClient

    from rag_core.core import RAGCore
    from rag_core.core_models import RAGCoreConfig
    from rag_core.events import EventBuffer
    from rag_core.runtime.app import create_app
    from tests.support import make_test_config

    captured_buffer: list[EventBuffer] = []

    def _core_factory(config: RAGCoreConfig) -> RAGCore:
        buffer = EventBuffer()
        captured_buffer.append(buffer)
        return RAGCore(
            config,
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(
                search_results=[make_search_result(id="hit-1", document_id="doc-1")]
            ),
            event_sink=buffer,
        )

    config = make_test_config(qdrant_collection="rag_core_audit_tests", embedding_dimensions=4)
    with tempfile.TemporaryDirectory() as job_dir:
        app = create_app(
            config=config,
            core_factory=_core_factory,
            job_db_path=Path(job_dir) / "jobs.sqlite",
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/search",
                headers={
                    "X-Request-Id": "req-from-gateway",
                    "X-Actor": "analyst@example.com",
                },
                json={
                    "query": "hello",
                    "namespace": "ns",
                    "corpus_ids": ["public"],
                },
            )
            assert response.status_code == 200, response.text

    [buffer] = captured_buffer
    [completed] = [e for e in buffer.events if isinstance(e, SearchCompleted)]
    assert completed.request_id == "req-from-gateway"
    assert completed.actor == "analyst@example.com"


def test_runtime_ingest_route_threads_audit_headers_into_ingest_events() -> None:
    from pathlib import Path
    import tempfile

    from starlette.testclient import TestClient

    from rag_core.core import RAGCore
    from rag_core.core_models import RAGCoreConfig
    from rag_core.events import EventBuffer
    from rag_core.runtime.app import create_app
    from tests.support import make_test_config

    captured_buffer: list[EventBuffer] = []

    def _core_factory(config: RAGCoreConfig) -> RAGCore:
        buffer = EventBuffer()
        captured_buffer.append(buffer)
        return RAGCore(
            config,
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(),
            event_sink=buffer,
        )

    config = make_test_config(
        qdrant_collection="rag_core_audit_runtime_ingest_tests",
        embedding_dimensions=4,
    )
    with tempfile.TemporaryDirectory() as job_dir:
        root = Path(job_dir).resolve()
        doc = root / "note.md"
        doc.write_text("hello from runtime ingest\n", encoding="utf-8")
        app = create_app(
            config=config,
            core_factory=_core_factory,
            job_db_path=root / "jobs.sqlite",
            ingest_roots=(root,),
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/ingest",
                headers={
                    "X-Request-Id": "req-ingest-gateway",
                    "X-Actor": "loader@example.com",
                    "X-Ingest-Id": "batch-7-doc-1",
                },
                json={
                    "path": str(doc),
                    "namespace": "ns",
                    "corpus_id": "public",
                },
            )
            assert response.status_code == 202, response.text
            status = client.get(f"/v1/ingest/{response.json()['job_id']}")
            assert status.status_code == 200
            assert status.json()["status"] == "completed"

    [buffer] = captured_buffer
    [completed] = [e for e in buffer.events if isinstance(e, IngestCompleted)]
    assert completed.request_id == "req-ingest-gateway"
    assert completed.actor == "loader@example.com"
    assert completed.ingest_id == "batch-7-doc-1"
