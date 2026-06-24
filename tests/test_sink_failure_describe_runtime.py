"""Sink kind + failure_count surface on describe_runtime and readiness.

Rank 12 surface contract:

* ``describe_runtime()`` carries an ``event_sink`` block with
  ``{"kind": str, "failure_count": int}`` regardless of how the sink was
  wired (``None`` / a builtin / a ``MultiSink`` / a user-supplied sink).
* ``describe_event_sink_status`` aggregates failures across ``MultiSink``
  children via the existing ``_FailureCounter`` rollup.
* Readiness payload mirrors the same field and sets ``degraded=True`` when
  ``failure_count > 0`` without flipping ``ready=False`` (observable, not
  blocking).
"""

from __future__ import annotations

import asyncio
import logging

from rag_core import Engine
from rag_core.events import (
    EventBuffer,
    EventSink,
    LoggingSink,
    MultiSink,
    NoOpSink,
)
from rag_core.events.sinks import (
    BUFFER_EVENT_SINK_PROVIDER,
    MULTI_EVENT_SINK_PROVIDER,
    NOOP_EVENT_SINK_PROVIDER,
    describe_event_sink_status,
)
from rag_core.events.types import Event, IngestStarted
from rag_core.runtime.health import readiness_payload
from rag_core.search.providers.memory_store import InMemoryVectorStore

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    make_test_config,
)


def test_describe_event_sink_status_handles_none_sink() -> None:
    status = describe_event_sink_status(None)

    assert status == {"kind": NOOP_EVENT_SINK_PROVIDER, "failure_count": 0}


def test_describe_event_sink_status_returns_provider_name_and_failures() -> None:
    class ExplodingSink:
        def emit(self, event: Event) -> None:
            raise RuntimeError("boom")

    multi = MultiSink(ExplodingSink(), EventBuffer())
    multi.emit(IngestStarted(filename="a.txt"))

    status = describe_event_sink_status(multi)

    assert status["kind"] == MULTI_EVENT_SINK_PROVIDER
    assert isinstance(status["failure_count"], int)
    assert status["failure_count"] >= 1


def test_describe_event_sink_status_falls_back_to_class_name() -> None:
    class _UserSink:
        def emit(self, event: Event) -> None:
            return None

    status = describe_event_sink_status(_UserSink())

    assert status == {"kind": "_UserSink", "failure_count": 0}


def test_describe_runtime_includes_event_sink_block_for_noop() -> None:
    core = Engine(
        make_test_config(embedding_dimensions=4),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=InMemoryVectorStore(),
    )
    try:
        payload = core.describe_runtime()
    finally:
        asyncio.run(core.close())

    event_sink = payload["event_sink"]
    assert isinstance(event_sink, dict)
    assert event_sink == {"kind": NOOP_EVENT_SINK_PROVIDER, "failure_count": 0}


def test_describe_runtime_event_sink_reports_buffer_kind() -> None:
    buffer = EventBuffer()
    core = Engine(
        make_test_config(embedding_dimensions=4),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=InMemoryVectorStore(),
        event_sink=buffer,
    )
    try:
        payload = core.describe_runtime()
    finally:
        asyncio.run(core.close())

    event_sink = payload["event_sink"]
    assert event_sink == {
        "kind": BUFFER_EVENT_SINK_PROVIDER,
        "failure_count": 0,
    }


def test_describe_runtime_event_sink_reports_multi_sink_failures() -> None:
    class ExplodingLogger(logging.Logger):
        def handle(self, record: logging.LogRecord) -> None:
            raise RuntimeError("log write failed")

    exploding = LoggingSink(logger=ExplodingLogger("rag_core.tests.sink_failure"))
    multi = MultiSink(exploding, EventBuffer())

    core = Engine(
        make_test_config(embedding_dimensions=4),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=InMemoryVectorStore(),
        event_sink=multi,
    )
    try:
        # Drive an ingest path so the sink receives real events and fails.
        asyncio.run(
            core.add_bytes(
                file_bytes=b"hello world",
                filename="a.txt",
                mime_type="text/plain",
                namespace="ns",
                collection="corp",
                document_id="doc-1",
            )
        )
        payload = core.describe_runtime()
    finally:
        asyncio.run(core.close())

    event_sink = payload["event_sink"]
    assert isinstance(event_sink, dict)
    assert event_sink["kind"] == MULTI_EVENT_SINK_PROVIDER
    assert isinstance(event_sink["failure_count"], int)
    assert event_sink["failure_count"] >= 1


def test_describe_event_sink_status_method_on_core() -> None:
    buffer = EventBuffer()
    core = Engine(
        make_test_config(embedding_dimensions=4),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=InMemoryVectorStore(),
        event_sink=buffer,
    )
    try:
        assert core.describe_event_sink_status() == {
            "kind": BUFFER_EVENT_SINK_PROVIDER,
            "failure_count": 0,
        }
    finally:
        asyncio.run(core.close())


def test_readiness_payload_surfaces_sink_status_without_flipping_ready() -> None:
    status_clean = {"kind": BUFFER_EVENT_SINK_PROVIDER, "failure_count": 0}
    payload = readiness_payload(
        ready=True,
        checks={"core": {"status": "ok"}},
        event_sink_status=status_clean,
    )

    assert payload["ready"] is True
    assert payload["event_sink"] == status_clean
    assert "degraded" not in payload


def test_readiness_payload_marks_degraded_when_sink_has_failures() -> None:
    status_degraded = {"kind": MULTI_EVENT_SINK_PROVIDER, "failure_count": 3}
    payload = readiness_payload(
        ready=True,
        checks={"core": {"status": "ok"}},
        event_sink_status=status_degraded,
    )

    # Sink failures are observable, not blocking: ``ready`` stays True but
    # the top-level ``degraded`` flag and the ``event_sink.failure_count``
    # give the operator a clear signal.
    assert payload["ready"] is True
    assert payload["degraded"] is True
    assert payload["event_sink"] == status_degraded


def test_readiness_payload_back_compat_without_sink_status() -> None:
    """Callers that don't pass ``event_sink_status`` still get the legacy shape."""
    payload = readiness_payload(ready=True, checks={"core": {"status": "ok"}})

    assert payload["ready"] is True
    assert "event_sink" not in payload
    assert "degraded" not in payload


def test_noop_sink_provider_name_round_trips() -> None:
    # Sanity check that the helper is wired against the canonical constants.
    status = describe_event_sink_status(NoOpSink())

    assert status == {"kind": NOOP_EVENT_SINK_PROVIDER, "failure_count": 0}


# Ensure EventSink protocol satisfaction is still observable (kept for
# regression: the helper accepts any ``EventSink``-shaped object).
def test_describe_event_sink_status_accepts_event_sink_protocol() -> None:
    class _Sink:
        def emit(self, event: Event) -> None:
            return None

    sink: EventSink = _Sink()
    status = describe_event_sink_status(sink)

    assert status["kind"] == "_Sink"
    assert status["failure_count"] == 0
