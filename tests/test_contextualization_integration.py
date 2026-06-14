from __future__ import annotations

import asyncio
from typing import Any

from rag_core import RAGCore, RAGCoreConfig
from rag_core.config import ContextualizerConfig, EmbeddingConfig, QdrantConfig
from rag_core.documents.contextualizer import (
    ChunkContextRequest,
    ChunkContextualizer,
    NoOpContextualizer,
)
from rag_core.documents.contextualizer_adapters import CachingContextualizer
import rag_core.documents.contextualizer_adapters as contextualizer_adapters_module
from rag_core.events import EventBuffer
from rag_core.events.types import ContextualizeCompleted
from rag_core.search.context_pack import build_context_pack
from rag_core.search.policy import CorpusPolicy
from rag_core.search.stored_payload import payload_to_result
from rag_core.search.providers.chunk_context_cache import (
    ChunkContextKey,
    InMemoryChunkContextCache,
)
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


class _CappedStubContextualizer(_StubContextualizer):
    def __init__(self, chunk_cap: int) -> None:
        super().__init__("test:capped")
        self.contextualizer_chunk_cap = chunk_cap


class _RecordingChunkContextCache(InMemoryChunkContextCache):
    def __init__(self) -> None:
        super().__init__()
        self.write_count = 0

    async def put(self, key: ChunkContextKey, context: str) -> None:
        self.write_count += 1
        await super().put(key, context)


class _FakeAnthropicTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeAnthropicTextBlock(text)]


class _FakeAnthropicMessages:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> _FakeAnthropicResponse:
        self.calls.append(dict(kwargs))
        return _FakeAnthropicResponse(self._text)


class _FakeAnthropicClient:
    def __init__(self, text: str) -> None:
        self.messages = _FakeAnthropicMessages(text)


_DOCUMENT = (
    "# Title\n\nIntro paragraph one.\n\n"
    "## Section\n\nSecond paragraph with more body text. "
    "More content for chunking diversity.\n\n"
    "## Another\n\nThird paragraph keeps content flowing for the chunker."
)


def _build_core(
    *,
    store: RecordingVectorStore | None = None,
    embedding: FakeEmbeddingProvider | None = None,
    sparse: FakeSparseEmbedder | None = None,
    config: RAGCoreConfig | None = None,
    chunk_contextualizer: ChunkContextualizer | None = None,
    chunk_context_cache: Any = None,
    event_sink: Any = None,
) -> tuple[RAGCore, FakeEmbeddingProvider, RecordingVectorStore]:
    embedding = embedding or FakeEmbeddingProvider()
    store = store or RecordingVectorStore()
    core = RAGCore(
        config
        or RAGCoreConfig(
            qdrant=QdrantConfig(location=":memory:", collection="contextual_integration"),
            embedding=EmbeddingConfig(dimensions=4),
        ),
        embedding_provider=embedding,
        sparse_embedder=sparse or FakeSparseEmbedder(),
        vector_store=store,
        chunk_contextualizer=chunk_contextualizer,
        chunk_context_cache=chunk_context_cache,
        event_sink=event_sink,
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


def _config_with_contextualizer(
    *,
    collection: str,
    enabled: bool = True,
) -> RAGCoreConfig:
    return RAGCoreConfig(
        qdrant=QdrantConfig(location=":memory:", collection=collection),
        embedding=EmbeddingConfig(dimensions=4),
        contextualizer=ContextualizerConfig(
            provider="anthropic",
            model="claude-test",
            enabled=enabled,
            contextualizer_chunk_cap=1,
        ),
    )


def _install_failing_anthropic_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_create_client(*, api_key: str | None) -> object:
        raise AssertionError(f"unexpected Anthropic client for {api_key!r}")

    monkeypatch.setattr(
        contextualizer_adapters_module,
        "create_anthropic_client",
        fail_create_client,
    )


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


def test_contextualized_text_feeds_dense_and_sparse_index_only() -> None:
    contextualizer = _StubContextualizer()
    embedding = FakeEmbeddingProvider(vocabulary=("original", "context", "ctx-0"))
    sparse = FakeSparseEmbedder(include_extra_channel=False)
    core, embedding, store = _build_core(
        embedding=embedding,
        sparse=sparse,
        chunk_contextualizer=contextualizer,
    )
    try:
        asyncio.run(
            core.ingest_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help",
                document_id="doc-1",
            )
        )
    finally:
        asyncio.run(core.close())

    assert store.upsert_calls
    point = store.upsert_calls[0][0]
    assert "ctx-0" in embedding.embed_texts_calls[0][0]
    assert "ctx-0" in sparse.embed_texts_multi_calls[0][0]
    payload_text = point.payload["text"]
    assert isinstance(payload_text, str)
    assert payload_text == payload_text.replace("ctx-0\n\n", "")
    assert "ctx-0" not in payload_text

    result = payload_to_result(point_id=point.id, payload=point.payload, score=0.9)
    pack = build_context_pack([result], query="title")

    assert "ctx-0" not in result.text
    assert "ctx-0" not in pack.as_prompt_text()
    assert pack.snippets[0].text == result.text


def test_config_contextualizer_enables_contextual_ingest_and_clean_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "ctx-config-marker"
    client = _FakeAnthropicClient(marker)
    seen_api_keys: list[str | None] = []

    def create_client(*, api_key: str | None) -> _FakeAnthropicClient:
        seen_api_keys.append(api_key)
        return client

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setattr(
        contextualizer_adapters_module,
        "create_anthropic_client",
        create_client,
    )
    config = _config_with_contextualizer(
        collection="contextual_config_integration",
    )
    embedding = FakeEmbeddingProvider(vocabulary=("ctx-config-marker", "title"))
    sparse = FakeSparseEmbedder(include_extra_channel=False)
    core, embedding, store = _build_core(
        config=config,
        embedding=embedding,
        sparse=sparse,
    )
    try:
        ingested = asyncio.run(
            core.ingest_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help",
                document_id="doc-1",
            )
        )
    finally:
        asyncio.run(core.close())

    assert ingested.chunk_count > 1
    assert seen_api_keys == ["anthropic-secret"]
    assert len(client.messages.calls) == 1
    assert store.upsert_calls
    point = store.upsert_calls[0][0]
    assert marker in embedding.embed_texts_calls[0][0]
    assert marker in sparse.embed_texts_multi_calls[0][0]
    assert all(marker not in text for text in embedding.embed_texts_calls[0][1:])
    assert all(marker not in text for text in sparse.embed_texts_multi_calls[0][1:])
    processing_version = str(point.payload["processing_version"])
    assert '"contextualizer_id":"anthropic:claude-test:' in processing_version
    assert '"contextualizer_chunk_cap":1' in processing_version

    payload_text = point.payload["text"]
    assert isinstance(payload_text, str)
    assert marker not in payload_text
    result = payload_to_result(point_id=point.id, payload=point.payload, score=0.9)
    pack = build_context_pack([result], query="title")
    assert marker not in result.text
    assert marker not in pack.as_prompt_text()
    assert marker not in repr([citation.to_payload() for citation in pack.citations])


def test_disabled_contextualizer_config_keeps_contextualizer_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_failing_anthropic_client(monkeypatch)
    config = _config_with_contextualizer(
        collection="contextual_config_disabled",
        enabled=False,
    )
    core, _embedding, _store = _build_core(config=config)

    try:
        runtime = core.describe_runtime()
        prepared = asyncio.run(
            core.prepare_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
            )
        )
    finally:
        asyncio.run(core.close())

    assert '"contextualizer_id"' not in str(runtime["processing_version"])
    assert '"contextualizer_chunk_cap"' not in str(runtime["processing_version"])
    for chunk in prepared.chunks:
        assert chunk.embedding_text == chunk.text


def test_injected_contextualizer_overrides_contextualizer_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_failing_anthropic_client(monkeypatch)
    injected = _StubContextualizer("test:override")
    config = _config_with_contextualizer(
        collection="contextual_config_override",
    )
    core, _embedding, _store = _build_core(
        config=config,
        chunk_contextualizer=injected,
    )

    try:
        runtime = core.describe_runtime()
        prepared = asyncio.run(
            core.prepare_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
            )
        )
    finally:
        asyncio.run(core.close())

    assert len(injected.requests) == len(prepared.chunks)
    processing_version = str(runtime["processing_version"])
    assert '"contextualizer_id":"test:override"' in processing_version
    assert "anthropic:claude-test" not in processing_version


def test_chunk_context_cache_is_scoped_and_delete_purges_only_that_document() -> None:
    contextualizer = _StubContextualizer()
    cache = InMemoryChunkContextCache()
    core, _embedding, _store = _build_core(
        chunk_contextualizer=contextualizer,
        chunk_context_cache=cache,
    )
    try:
        first = asyncio.run(
            core.ingest_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help-a",
                document_id="doc-1",
            )
        )
        first_request_count = len(contextualizer.requests)
        second = asyncio.run(
            core.ingest_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help-b",
                document_id="doc-1",
            )
        )
        second_request_count = len(contextualizer.requests)

        asyncio.run(
            core.delete_document(
                namespace="acme",
                corpus_id="help-a",
                document_id="doc-1",
            )
        )
        asyncio.run(
            core.ingest_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help-b",
                document_id="doc-1",
                force_reindex=True,
            )
        )
        after_help_b_reindex = len(contextualizer.requests)
        asyncio.run(
            core.ingest_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help-a",
                document_id="doc-1",
            )
        )
    finally:
        asyncio.run(core.close())

    assert first.chunk_count > 0
    assert second.chunk_count == first.chunk_count
    assert first_request_count == first.chunk_count
    assert second_request_count == first.chunk_count + second.chunk_count
    assert after_help_b_reindex == second_request_count
    assert len(contextualizer.requests) == second_request_count + first.chunk_count


def test_prepare_uses_resolved_noop_chunk_context_cache_when_cache_disabled() -> None:
    contextualizer = _StubContextualizer()
    cache = _RecordingChunkContextCache()
    config = RAGCoreConfig(
        qdrant=QdrantConfig(location=":memory:", collection="contextual_disabled_cache"),
        embedding=EmbeddingConfig(dimensions=4),
        corpus_policy=CorpusPolicy(cache_disabled=True),
    )
    core, _embedding, _store = _build_core(
        config=config,
        chunk_contextualizer=contextualizer,
        chunk_context_cache=cache,
    )
    try:
        prepared = asyncio.run(
            core.prepare_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
            )
        )
    finally:
        asyncio.run(core.close())

    assert prepared.chunks
    assert len(contextualizer.requests) == len(prepared.chunks)
    assert cache.write_count == 0


def test_contextualizer_chunk_cap_indexes_raw_tail_and_reports_skipped_count() -> None:
    contextualizer = _CappedStubContextualizer(chunk_cap=1)
    events = EventBuffer()
    core, _embedding, _store = _build_core(
        chunk_contextualizer=contextualizer,
        event_sink=events,
    )
    try:
        prepared = asyncio.run(
            core.prepare_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
            )
        )
    finally:
        asyncio.run(core.close())

    assert len(prepared.chunks) > 1
    assert len(contextualizer.requests) == 1
    assert prepared.chunks[0].embedding_text.startswith("ctx-0\n\n")
    assert prepared.chunks[0].sparse_text == prepared.chunks[0].embedding_text
    for chunk in prepared.chunks[1:]:
        assert chunk.embedding_text == chunk.text
        assert chunk.sparse_text == chunk.text
    [completed] = [
        event for event in events.events if isinstance(event, ContextualizeCompleted)
    ]
    assert completed.chunk_cap == 1
    assert completed.contextualized_chunk_count == 1
    assert completed.skipped_chunk_count == len(prepared.chunks) - 1


def test_contextualizer_chunk_cap_participates_in_processing_fingerprint() -> None:
    config = RAGCoreConfig(
        qdrant=QdrantConfig(location=":memory:", collection="contextual_cap_fingerprint"),
        embedding=EmbeddingConfig(dimensions=4),
    )
    core_a, _embedding_a, _store_a = _build_core(
        config=config,
        chunk_contextualizer=_CappedStubContextualizer(chunk_cap=1),
    )
    core_b, _embedding_b, _store_b = _build_core(
        config=config,
        chunk_contextualizer=_CappedStubContextualizer(chunk_cap=2),
    )
    try:
        version_a = str(core_a.describe_runtime()["processing_version"])
        version_b = str(core_b.describe_runtime()["processing_version"])
    finally:
        asyncio.run(core_a.close())
        asyncio.run(core_b.close())

    assert version_a != version_b
    assert '"contextualizer_chunk_cap":1' in version_a
    assert '"contextualizer_chunk_cap":2' in version_b


def test_pre_wrapped_capped_contextualizer_enforces_cap_and_fingerprints_cap() -> None:
    inner = _CappedStubContextualizer(chunk_cap=1)
    wrapped = CachingContextualizer(inner, InMemoryChunkContextCache())
    events = EventBuffer()
    core, _embedding, _store = _build_core(
        chunk_contextualizer=wrapped,
        event_sink=events,
    )
    try:
        version = str(core.describe_runtime()["processing_version"])
        prepared = asyncio.run(
            core.prepare_bytes(
                file_bytes=_DOCUMENT.encode("utf-8"),
                filename="doc.md",
                mime_type="text/markdown",
            )
        )
    finally:
        asyncio.run(core.close())

    assert '"contextualizer_chunk_cap":1' in version
    assert len(prepared.chunks) > 1
    assert len(inner.requests) == 1
    assert prepared.chunks[0].embedding_text.startswith("ctx-0\n\n")
    for chunk in prepared.chunks[1:]:
        assert chunk.embedding_text == chunk.text
        assert chunk.sparse_text == chunk.text
    [completed] = [
        event for event in events.events if isinstance(event, ContextualizeCompleted)
    ]
    assert completed.chunk_cap == 1
    assert completed.contextualized_chunk_count == 1
    assert completed.skipped_chunk_count == len(prepared.chunks) - 1
