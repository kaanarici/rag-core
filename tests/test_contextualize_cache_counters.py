"""ContextualizeCompleted exposes cache_hits/cache_misses/cache_writes.

Mirrors the EmbedCompleted contract so the caller's audit consumer can read
contextualizer cache effectiveness off the same event shape as embed-cache
effectiveness.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from rag_core import RAGCore, RAGCoreConfig
from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.documents.contextualizer import ChunkContextRequest
from rag_core.events import EventBuffer
from rag_core.events.types import ContextualizeCompleted
from rag_core.search.providers.chunk_context_cache import InMemoryChunkContextCache

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
)

pytestmark = [pytest.mark.plumbing]


class _StubContextualizer:
    def __init__(self) -> None:
        self.contextualizer_id = "test:stub"
        self.requests: list[ChunkContextRequest] = []

    async def contextualize(self, request: ChunkContextRequest) -> str:
        self.requests.append(request)
        return f"ctx-{request.chunk_index}"


_DOCUMENT = (
    "# Title\n\nIntro paragraph one.\n\n"
    "## Section\n\nSecond paragraph with more body text. "
    "More content for chunking diversity.\n\n"
    "## Another\n\nThird paragraph keeps content flowing for the chunker."
)


def _build_core(
    *,
    chunk_contextualizer: Any,
    chunk_context_cache: Any,
    event_sink: EventBuffer,
) -> RAGCore:
    return RAGCore(
        RAGCoreConfig(
            qdrant=QdrantConfig(
                location=":memory:",
                collection="contextual_counters",
            ),
            embedding=EmbeddingConfig(dimensions=4),
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
        chunk_contextualizer=chunk_contextualizer,
        chunk_context_cache=chunk_context_cache,
        event_sink=event_sink,
    )


def test_contextualize_completed_carries_default_zero_cache_counters() -> None:
    completed = ContextualizeCompleted(chunk_count=3, model="m", duration_ms=1.0)

    assert completed.cache_hits == 0
    assert completed.cache_misses == 0
    assert completed.cache_writes == 0


def test_contextualize_completed_reports_misses_and_writes_on_first_pass() -> None:
    contextualizer = _StubContextualizer()
    cache = InMemoryChunkContextCache()
    buffer = EventBuffer()
    core = _build_core(
        chunk_contextualizer=contextualizer,
        chunk_context_cache=cache,
        event_sink=buffer,
    )
    try:
        asyncio.run(
            core.prepare_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
            )
        )
    finally:
        asyncio.run(core.close())

    [completed] = [
        event for event in buffer.events if isinstance(event, ContextualizeCompleted)
    ]
    assert completed.succeeded is True
    assert completed.chunk_count > 0
    # First pass through the cache: every call is a miss, every miss writes.
    assert completed.cache_misses == completed.chunk_count
    assert completed.cache_hits == 0
    assert completed.cache_writes == completed.chunk_count


def test_contextualize_completed_reports_hits_on_second_pass() -> None:
    contextualizer = _StubContextualizer()
    cache = InMemoryChunkContextCache()
    buffer = EventBuffer()
    core = _build_core(
        chunk_contextualizer=contextualizer,
        chunk_context_cache=cache,
        event_sink=buffer,
    )
    try:
        asyncio.run(
            core.prepare_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
            )
        )
        buffer.clear()
        asyncio.run(
            core.prepare_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
            )
        )
    finally:
        asyncio.run(core.close())

    [completed] = [
        event for event in buffer.events if isinstance(event, ContextualizeCompleted)
    ]
    assert completed.cache_hits == completed.chunk_count
    assert completed.cache_misses == 0
    assert completed.cache_writes == 0


def test_contextualize_completed_reports_zero_counters_when_cache_absent() -> None:
    contextualizer = _StubContextualizer()
    buffer = EventBuffer()
    core = _build_core(
        chunk_contextualizer=contextualizer,
        chunk_context_cache=None,
        event_sink=buffer,
    )
    try:
        asyncio.run(
            core.prepare_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
            )
        )
    finally:
        asyncio.run(core.close())

    [completed] = [
        event for event in buffer.events if isinstance(event, ContextualizeCompleted)
    ]
    assert (
        completed.cache_hits,
        completed.cache_misses,
        completed.cache_writes,
    ) == (0, 0, 0)
