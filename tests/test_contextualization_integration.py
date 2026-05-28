from __future__ import annotations

import asyncio
from typing import Any

from rag_core import RAGCore, RAGCoreConfig
from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.documents.contextualizer import (
    ChunkContextRequest,
    ChunkContextualizer,
    NoOpContextualizer,
)
from rag_core.search.providers.chunk_context_cache import InMemoryChunkContextCache
import pytest

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
)

pytestmark = [pytest.mark.plumbing]


class _StubContextualizer:
    def __init__(self, contextualizer_id: str = "test:stub") -> None:
        self.contextualizer_id = contextualizer_id
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
    store: RecordingVectorStore | None = None,
    chunk_contextualizer: ChunkContextualizer | None = None,
    chunk_context_cache: Any = None,
) -> tuple[RAGCore, FakeEmbeddingProvider, RecordingVectorStore]:
    embedding = FakeEmbeddingProvider()
    store = store or RecordingVectorStore()
    core = RAGCore(
        RAGCoreConfig(
            qdrant=QdrantConfig(location=":memory:", collection="contextual_integration"),
            embedding=EmbeddingConfig(dimensions=4),
        ),
        embedding_provider=embedding,
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
        chunk_contextualizer=chunk_contextualizer,
        chunk_context_cache=chunk_context_cache,
    )
    return core, embedding, store


def _prepare(core: RAGCore) -> Any:
    try:
        return asyncio.run(
            core.prepare_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
            )
        )
    finally:
        asyncio.run(core.close())


def test_chunk_contextualizer_threads_into_prepare_bytes_with_full_document() -> None:
    contextualizer = _StubContextualizer()
    core, _embedding, _store = _build_core(chunk_contextualizer=contextualizer)

    prepared = _prepare(core)

    assert prepared.chunks
    assert len(contextualizer.requests) == len(prepared.chunks)
    for chunk in prepared.chunks:
        assert chunk.embedding_text == f"ctx-{chunk.chunk_index}\n\n{chunk.text}"
    for index, request in enumerate(contextualizer.requests):
        assert request.chunk_index == index
        assert request.total_chunks == len(contextualizer.requests)
        assert request.document_filename == "doc.md"
        assert "Section" in request.document_markdown


def test_chunk_contextualizer_identity_participates_in_processing_fingerprint() -> None:
    store = RecordingVectorStore()
    core_a, _embedding_a, _store_a = _build_core(
        store=store,
        chunk_contextualizer=_StubContextualizer("test:ctx-a"),
    )
    try:
        first = asyncio.run(
            core_a.ingest_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help",
                document_id="doc-1",
            )
        )
    finally:
        asyncio.run(core_a.close())

    core_b, _embedding_b, _store_b = _build_core(
        store=store,
        chunk_contextualizer=_StubContextualizer("test:ctx-b"),
    )
    try:
        second = asyncio.run(
            core_b.ingest_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help",
                document_id="doc-1",
            )
        )
    finally:
        asyncio.run(core_b.close())

    assert first.ingest_state == "created"
    assert second.ingest_state == "reindexed"
    assert len(store.upsert_calls) == 2
    assert '"contextualizer_id":"test:ctx-a"' in str(
        store.upsert_calls[0][0].payload["processing_version"]
    )
    assert '"contextualizer_id":"test:ctx-b"' in str(
        store.upsert_calls[1][0].payload["processing_version"]
    )


def test_chunk_context_cache_avoids_recomputing_on_reindex() -> None:
    contextualizer = _StubContextualizer()
    cache = InMemoryChunkContextCache()
    core, _embedding, _store = _build_core(
        chunk_contextualizer=contextualizer,
        chunk_context_cache=cache,
    )
    try:
        asyncio.run(
            core.prepare_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
            )
        )
        first_call_count = len(contextualizer.requests)
        asyncio.run(
            core.prepare_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
            )
        )
    finally:
        asyncio.run(core.close())

    assert first_call_count > 0
    assert len(contextualizer.requests) == first_call_count


def test_embedding_text_equals_chunk_text_when_no_context_is_added() -> None:
    # NoOpContextualizer and "no contextualizer at all" should both leave
    # embedding_text untouched. Cover both via the same observable contract.
    for chunk_contextualizer in (None, NoOpContextualizer()):
        core, _embedding, _store = _build_core(chunk_contextualizer=chunk_contextualizer)
        prepared = _prepare(core)

        assert prepared.chunks
        for chunk in prepared.chunks:
            assert chunk.embedding_text == chunk.text
