"""Rank 11. Per-tier collection isolation, embedding-identity guard,
cold-start race handling, and bounded lexical sidecar.

These cover the tiered-deployment contract: collection names physically separate
corpora, the binding step refuses to attach to a collection produced by a
different embedder, the create path tolerates a parallel creator, and the
in-memory lexical sidecar refuses to silently exhaust the heap.
"""

from __future__ import annotations
from rag_core.search.providers.qdrant_store import QdrantVectorStore

from typing import Any, cast

import pytest

from rag_core.events.sink import EventSink
from rag_core.events.types import Event
from rag_core._engine.core_runtime import (
    resolve_collection_name,
)
from rag_core.search.providers.qdrant_store import (
    EMBEDDING_DIMENSIONS_METADATA_KEY,
    EMBEDDING_MODEL_METADATA_KEY,
    EmbeddingIdentityMismatch,
    assert_embedding_identity_matches,
    extract_collection_metadata,
    pack_embedding_identity_metadata,
)
from rag_core.search.vector_models import SearchResult


# ---------------------------------------------------------------------------
# resolve_collection_name
# ---------------------------------------------------------------------------


def test_resolve_collection_name_includes_model_and_dimensions() -> None:
    name = resolve_collection_name(
        base_name="rag_core_chunks",
        model_name="text-embedding-3-large",
        dimensions=3072,
        dimension_aware=True,
    )
    assert name == "rag_core_chunks__text_embedding_3_large_3072d"


def test_resolve_collection_name_handles_dimension_aware_false() -> None:
    assert (
        resolve_collection_name(
            base_name="rag_core_chunks",
            model_name="m",
            dimensions=4,
            dimension_aware=False,
        )
        == "rag_core_chunks"
    )


# ---------------------------------------------------------------------------
# Embedding-identity sentinel
# ---------------------------------------------------------------------------


def test_pack_embedding_identity_metadata_returns_none_for_blank_model() -> None:
    assert pack_embedding_identity_metadata(embedding_model=None, dimensions=3) is None
    assert pack_embedding_identity_metadata(embedding_model="", dimensions=3) is None


def test_pack_embedding_identity_metadata_includes_model_and_dimensions() -> None:
    packed = pack_embedding_identity_metadata(
        embedding_model="text-embedding-3-large",
        dimensions=3072,
    )
    assert packed == {
        EMBEDDING_MODEL_METADATA_KEY: "text-embedding-3-large",
        EMBEDDING_DIMENSIONS_METADATA_KEY: 3072,
    }


def test_assert_embedding_identity_matches_passes_on_exact_match() -> None:
    assert_embedding_identity_matches(
        collection_name="rag_core_chunks__restricted__m_3d",
        expected_model="m",
        expected_dimensions=3,
        collection_metadata={
            EMBEDDING_MODEL_METADATA_KEY: "m",
            EMBEDDING_DIMENSIONS_METADATA_KEY: 3,
        },
        mismatch_cls=EmbeddingIdentityMismatch,
    )


def test_assert_embedding_identity_matches_refuses_model_swap() -> None:
    with pytest.raises(EmbeddingIdentityMismatch, match="embedding"):
        assert_embedding_identity_matches(
            collection_name="rag_core_chunks__restricted__m_3d",
            expected_model="other_model",
            expected_dimensions=3,
            collection_metadata={
                EMBEDDING_MODEL_METADATA_KEY: "m",
                EMBEDDING_DIMENSIONS_METADATA_KEY: 3,
            },
            mismatch_cls=EmbeddingIdentityMismatch,
        )


def test_assert_embedding_identity_matches_refuses_dim_swap() -> None:
    with pytest.raises(EmbeddingIdentityMismatch, match="dimensions"):
        assert_embedding_identity_matches(
            collection_name="rag_core_chunks__restricted__m_3d",
            expected_model="m",
            expected_dimensions=4,
            collection_metadata={
                EMBEDDING_MODEL_METADATA_KEY: "m",
                EMBEDDING_DIMENSIONS_METADATA_KEY: 3,
            },
            mismatch_cls=EmbeddingIdentityMismatch,
        )


def test_assert_embedding_identity_matches_refuses_unknown_legacy_space() -> None:
    with pytest.raises(EmbeddingIdentityMismatch, match="no embedding model identity"):
        assert_embedding_identity_matches(
            collection_name="legacy",
            expected_model="m",
            expected_dimensions=3,
            collection_metadata={},
            mismatch_cls=EmbeddingIdentityMismatch,
        )


# ---------------------------------------------------------------------------
# Cold-start race handling in create_qdrant_collection
# ---------------------------------------------------------------------------


class _FakeConfigParams:
    def __init__(self, *, size: int, sparse_names: list[str] | None) -> None:
        from rag_core.search.providers.qdrant_payloads import _DENSE_VECTOR_NAME

        self.vectors = {_DENSE_VECTOR_NAME: type("V", (), {"size": size})()}
        if sparse_names is None:
            self.sparse_vectors = None
        else:
            self.sparse_vectors = {n: object() for n in sparse_names}


class _FakeCollectionInfoWithMetadata:
    def __init__(
        self,
        *,
        size: int,
        sparse_names: list[str] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.config = type(
            "Config",
            (),
            {
                "params": _FakeConfigParams(size=size, sparse_names=sparse_names),
                "metadata": metadata or {},
            },
        )()


class _FakeQdrantClientForLifecycle:
    """Drives ``ensure_qdrant_collection_ready`` end-to-end."""

    def __init__(
        self,
        *,
        existing_names: list[str],
        sparse_names: list[str] | None = None,
        size: int = 3,
        metadata: dict[str, Any] | None = None,
        create_raises: BaseException | None = None,
    ) -> None:
        self._existing_names = list(existing_names)
        self._sparse_names = sparse_names or ["bm25"]
        self._size = size
        self._metadata = metadata or {}
        self._create_raises = create_raises
        self.create_calls: list[dict[str, Any]] = []
        self.get_collection_calls: list[str] = []
        self.create_payload_index_calls: list[dict[str, Any]] = []

    async def get_collections(self) -> object:
        from types import SimpleNamespace

        return SimpleNamespace(
            collections=[
                type("C", (), {"name": name})() for name in self._existing_names
            ]
        )

    async def get_collection(self, *, collection_name: str) -> object:
        self.get_collection_calls.append(collection_name)
        return _FakeCollectionInfoWithMetadata(
            size=self._size,
            sparse_names=self._sparse_names,
            metadata=self._metadata,
        )

    async def create_collection(self, **kwargs: Any) -> None:
        self.create_calls.append(kwargs)
        if self._create_raises is not None:
            # Simulate the race: the collection now exists for any subsequent
            # bind step, even though we raised here.
            if "docs" not in self._existing_names:
                self._existing_names.append("docs")
            raise self._create_raises

    async def create_payload_index(self, **kwargs: Any) -> None:
        self.create_payload_index_calls.append(kwargs)


def _store(client: object, *, embedding_model: str | None = None) -> QdrantVectorStore:
    store = QdrantVectorStore(
        url=None,
        api_key=None,
        location=":memory:",
        collection_name="docs",
        dense_dimensions=3,
        embedding_model=embedding_model,
    )
    store._client = cast(Any, client)
    return store


def test_ensure_ready_refuses_collection_with_mismatched_embedding_model() -> None:
    import asyncio

    client = _FakeQdrantClientForLifecycle(
        existing_names=["docs"],
        metadata={
            EMBEDDING_MODEL_METADATA_KEY: "old_embedder",
            EMBEDDING_DIMENSIONS_METADATA_KEY: 3,
        },
    )
    store = _store(client, embedding_model="new_embedder")
    with pytest.raises(EmbeddingIdentityMismatch):
        asyncio.run(store.ensure_collection())
    assert not store._ready


def test_ensure_ready_binds_when_embedding_identity_matches() -> None:
    import asyncio

    client = _FakeQdrantClientForLifecycle(
        existing_names=["docs"],
        metadata={
            EMBEDDING_MODEL_METADATA_KEY: "text-embedding-3-large",
            EMBEDDING_DIMENSIONS_METADATA_KEY: 3,
        },
    )
    store = _store(client, embedding_model="text-embedding-3-large")
    asyncio.run(store.ensure_collection())
    assert store._ready
    assert client.create_calls == []


def test_ensure_ready_refuses_legacy_collection_without_sentinel() -> None:
    import asyncio

    client = _FakeQdrantClientForLifecycle(existing_names=["docs"], metadata=None)
    store = _store(client, embedding_model="some_embedder")
    with pytest.raises(EmbeddingIdentityMismatch, match="no embedding model identity"):
        asyncio.run(store.ensure_collection())
    assert not store._ready


def test_create_collection_persists_identity_sentinel() -> None:
    import asyncio

    client = _FakeQdrantClientForLifecycle(existing_names=[])
    store = _store(client, embedding_model="text-embedding-3-large")
    asyncio.run(store.ensure_collection())
    assert len(client.create_calls) == 1
    metadata = client.create_calls[0].get("metadata")
    assert metadata == {
        EMBEDDING_MODEL_METADATA_KEY: "text-embedding-3-large",
        EMBEDDING_DIMENSIONS_METADATA_KEY: 3,
    }


def test_create_collection_omits_metadata_when_no_embedding_model() -> None:
    import asyncio

    client = _FakeQdrantClientForLifecycle(existing_names=[])
    store = _store(client, embedding_model=None)
    asyncio.run(store.ensure_collection())
    assert "metadata" not in client.create_calls[0]


def test_create_collection_falls_through_on_409_race() -> None:
    import asyncio
    from qdrant_client.http.exceptions import UnexpectedResponse

    race = UnexpectedResponse(
        status_code=409,
        reason_phrase="Conflict",
        content=b"already exists",
        headers=None,  # type: ignore[arg-type]
    )
    client = _FakeQdrantClientForLifecycle(
        existing_names=[],  # we will append on the create-call simulating the race
        create_raises=race,
        metadata={
            EMBEDDING_MODEL_METADATA_KEY: "m",
            EMBEDDING_DIMENSIONS_METADATA_KEY: 3,
        },
    )
    store = _store(client, embedding_model="m")
    asyncio.run(store.ensure_collection())
    assert store._ready
    # The lifecycle should have rebound to the now-existing collection.
    assert "docs" in client.get_collection_calls


def test_create_collection_race_still_enforces_identity() -> None:
    import asyncio
    from qdrant_client.http.exceptions import UnexpectedResponse

    race = UnexpectedResponse(
        status_code=409,
        reason_phrase="Conflict",
        content=b"already exists",
        headers=None,  # type: ignore[arg-type]
    )
    client = _FakeQdrantClientForLifecycle(
        existing_names=[],
        create_raises=race,
        metadata={
            EMBEDDING_MODEL_METADATA_KEY: "old_embedder",
            EMBEDDING_DIMENSIONS_METADATA_KEY: 3,
        },
    )
    with pytest.raises(EmbeddingIdentityMismatch):
        asyncio.run(_store(client, embedding_model="new_embedder").ensure_collection())


def test_create_collection_propagates_non_race_unexpected_response() -> None:
    import asyncio
    from qdrant_client.http.exceptions import UnexpectedResponse

    other = UnexpectedResponse(
        status_code=500,
        reason_phrase="Internal Server Error",
        content=b"boom",
        headers=None,  # type: ignore[arg-type]
    )
    client = _FakeQdrantClientForLifecycle(
        existing_names=[],
        create_raises=other,
    )
    with pytest.raises(UnexpectedResponse):
        asyncio.run(_store(client, embedding_model="m").ensure_collection())


def test_extract_collection_metadata_returns_empty_on_missing() -> None:
    info = type("Info", (), {"config": type("C", (), {})()})()
    assert extract_collection_metadata(info) == {}


# ---------------------------------------------------------------------------
# Lexical sidecar bounds
# ---------------------------------------------------------------------------


class _RecordingSink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


def _make_result(point_id: str, *, text: str = "hello") -> SearchResult:
    return SearchResult(
        id=point_id,
        text=text,
        score=0.5,
        content_type="document",
        source_type="file",
        tenant_id="ns",
        collection="corpus",
        document_id="doc-" + point_id,
        document_key="key-" + point_id,
        chunk_index=0,
        title="t",
    )
