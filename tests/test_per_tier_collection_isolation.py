"""Rank 11. Per-tier collection isolation, embedding-identity guard,
cold-start race handling, and bounded lexical sidecar.

These cover the tiered-deployment contract: collection names physically separate
corpora, the binding step refuses to attach to a collection produced by a
different embedder, the create path tolerates a parallel creator, and the
in-memory lexical sidecar refuses to silently exhaust the heap.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from rag_core.events.search_events import LexicalSidecarBoundExceeded
from rag_core.events.sink import EventSink
from rag_core.events.types import Event
from rag_core._engine.core_runtime import (
    resolve_collection_name,
)
from rag_core.search.lexical_sidecar import (
    DEFAULT_LEXICAL_SIDECAR_MAX_BYTES,
    DEFAULT_LEXICAL_SIDECAR_MAX_ENTRIES,
    LexicalSidecarRecord,
    PortableLexicalSidecar,
)
from rag_core.search.policy import CollectionPolicy, collection_slug_for
from rag_core.search.providers.qdrant_collection import (
    EMBEDDING_DIMENSIONS_METADATA_KEY,
    EMBEDDING_MODEL_METADATA_KEY,
    EmbeddingIdentityMismatch,
    QdrantAdapterConfig,
    QdrantCollectionState,
    assert_embedding_identity_matches,
    create_qdrant_collection,
    ensure_qdrant_collection_ready,
    extract_collection_metadata,
    pack_embedding_identity_metadata,
)
from rag_core.search.vector_models import SearchResult


# ---------------------------------------------------------------------------
# collection_slug + resolve_collection_name + CollectionPolicy.store_collection_slug
# ---------------------------------------------------------------------------


def test_collection_slug_lowercases_and_canonicalizes() -> None:
    assert collection_slug_for("public") == "public"
    assert collection_slug_for("Licensed") == "licensed"
    assert collection_slug_for("Restricted!!!") == "restricted"
    assert collection_slug_for("MIXED 2025/Q1") == "mixed_2025_q1"


def test_collection_slug_rejects_empty_after_sanitization() -> None:
    with pytest.raises(ValueError):
        collection_slug_for("")
    with pytest.raises(ValueError):
        collection_slug_for("///")


def test_resolve_collection_name_includes_collection_slug() -> None:
    name = resolve_collection_name(
        base_name="rag_core_chunks",
        model_name="text-embedding-3-large",
        dimensions=3072,
        dimension_aware=True,
        collection_slug="restricted",
    )
    assert name == "rag_core_chunks__restricted__text_embedding_3_large_3072d"


def test_resolve_collection_name_omits_slug_when_none() -> None:
    legacy = resolve_collection_name(
        base_name="rag_core_chunks",
        model_name="text-embedding-3-large",
        dimensions=3072,
        dimension_aware=True,
        collection_slug=None,
    )
    # Legacy single-collection layout preserved when no policy is bound.
    assert legacy == "rag_core_chunks__text_embedding_3_large_3072d"


def test_resolve_collection_name_handles_dimension_aware_false() -> None:
    assert resolve_collection_name(
        base_name="rag_core_chunks",
        model_name="m",
        dimensions=4,
        dimension_aware=False,
        collection_slug="public",
    ) == "rag_core_chunks__public"


def test_collection_policy_collection_slug_only_when_single_tier() -> None:
    bound = CollectionPolicy(
        bound_namespace="workspace_42",
        allowed_collections=frozenset({"restricted"}),
    )
    assert bound.store_collection_slug == "restricted"

    multi = CollectionPolicy(allowed_collections=frozenset({"public", "restricted"}))
    assert multi.store_collection_slug is None

    unbounded = CollectionPolicy()
    assert unbounded.store_collection_slug is None


def test_collection_policy_collection_slug_sluggifies_uppercase_tier() -> None:
    policy = CollectionPolicy(allowed_collections=frozenset({"Licensed"}))
    assert policy.store_collection_slug == "licensed"


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


def test_assert_embedding_identity_matches_tolerates_legacy_collection() -> None:
    # Legacy collection: no sentinel metadata. Assert_collection_compatible
    # still catches dimension mismatch upstream; identity check must not
    # break existing deployments.
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


def _adapter_config(
    *,
    collection_name: str = "docs",
    embedding_model: str | None = None,
    dimensions: int = 3,
) -> QdrantAdapterConfig:
    return QdrantAdapterConfig(
        collection_name=collection_name,
        dimensions=dimensions,
        quantization_enabled=False,
        is_local=True,
        max_concurrent=1,
        max_batch_size=1,
        embedding_model=embedding_model,
    )


def test_ensure_ready_refuses_collection_with_mismatched_embedding_model() -> None:
    import asyncio

    client = _FakeQdrantClientForLifecycle(
        existing_names=["docs"],
        metadata={
            EMBEDDING_MODEL_METADATA_KEY: "old_embedder",
            EMBEDDING_DIMENSIONS_METADATA_KEY: 3,
        },
    )
    config = _adapter_config(embedding_model="new_embedder")
    state = QdrantCollectionState()
    with pytest.raises(EmbeddingIdentityMismatch):
        asyncio.run(
            ensure_qdrant_collection_ready(
                client=client,  # type: ignore[arg-type]
                config=config,
                state=state,
                logger=logging.getLogger("test"),
            )
        )
    assert not state.ready


def test_ensure_ready_binds_when_embedding_identity_matches() -> None:
    import asyncio

    client = _FakeQdrantClientForLifecycle(
        existing_names=["docs"],
        metadata={
            EMBEDDING_MODEL_METADATA_KEY: "text-embedding-3-large",
            EMBEDDING_DIMENSIONS_METADATA_KEY: 3,
        },
    )
    config = _adapter_config(embedding_model="text-embedding-3-large")
    state = QdrantCollectionState()
    asyncio.run(
        ensure_qdrant_collection_ready(
            client=client,  # type: ignore[arg-type]
            config=config,
            state=state,
            logger=logging.getLogger("test"),
        )
    )
    assert state.ready
    assert client.create_calls == []


def test_ensure_ready_tolerates_legacy_collection_without_sentinel() -> None:
    import asyncio

    # No metadata field. Pre-sentinel deployment. Identity check should
    # accept it; dimension check still catches genuine shape problems.
    client = _FakeQdrantClientForLifecycle(
        existing_names=["docs"], metadata=None
    )
    state = QdrantCollectionState()
    asyncio.run(
        ensure_qdrant_collection_ready(
            client=client,  # type: ignore[arg-type]
            config=_adapter_config(embedding_model="some_embedder"),
            state=state,
            logger=logging.getLogger("test"),
        )
    )
    assert state.ready


def test_create_collection_persists_identity_sentinel() -> None:
    import asyncio

    client = _FakeQdrantClientForLifecycle(existing_names=[])
    config = _adapter_config(embedding_model="text-embedding-3-large")
    asyncio.run(
        create_qdrant_collection(
            client=client,  # type: ignore[arg-type]
            config=config,
            logger=logging.getLogger("test"),
        )
    )
    assert len(client.create_calls) == 1
    metadata = client.create_calls[0].get("metadata")
    assert metadata == {
        EMBEDDING_MODEL_METADATA_KEY: "text-embedding-3-large",
        EMBEDDING_DIMENSIONS_METADATA_KEY: 3,
    }


def test_create_collection_omits_metadata_when_no_embedding_model() -> None:
    import asyncio

    client = _FakeQdrantClientForLifecycle(existing_names=[])
    config = _adapter_config(embedding_model=None)
    asyncio.run(
        create_qdrant_collection(
            client=client,  # type: ignore[arg-type]
            config=config,
            logger=logging.getLogger("test"),
        )
    )
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
    state = QdrantCollectionState()
    asyncio.run(
        ensure_qdrant_collection_ready(
            client=client,  # type: ignore[arg-type]
            config=_adapter_config(embedding_model="m"),
            state=state,
            logger=logging.getLogger("test"),
        )
    )
    assert state.ready
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
    state = QdrantCollectionState()
    with pytest.raises(EmbeddingIdentityMismatch):
        asyncio.run(
            ensure_qdrant_collection_ready(
                client=client,  # type: ignore[arg-type]
                config=_adapter_config(embedding_model="new_embedder"),
                state=state,
                logger=logging.getLogger("test"),
            )
        )


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
    state = QdrantCollectionState()
    with pytest.raises(UnexpectedResponse):
        asyncio.run(
            ensure_qdrant_collection_ready(
                client=client,  # type: ignore[arg-type]
                config=_adapter_config(embedding_model="m"),
                state=state,
                logger=logging.getLogger("test"),
            )
        )


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
        namespace="ns",
        collection="corpus",
        document_id="doc-" + point_id,
        document_key="key-" + point_id,
        chunk_index=0,
        title="t",
    )


def test_lexical_sidecar_defaults_match_documented_bounds() -> None:
    assert DEFAULT_LEXICAL_SIDECAR_MAX_ENTRIES == 100_000
    assert DEFAULT_LEXICAL_SIDECAR_MAX_BYTES == 256 * 1024 * 1024


def test_lexical_sidecar_rejects_overflow_entries_and_emits_event() -> None:
    sink = _RecordingSink()
    sidecar = PortableLexicalSidecar(
        [], max_entries=2, max_bytes=10_000_000, event_sink=sink
    )
    sidecar.upsert_records(
        [
            LexicalSidecarRecord(namespace="ns", result=_make_result("a")),
            LexicalSidecarRecord(namespace="ns", result=_make_result("b")),
            LexicalSidecarRecord(namespace="ns", result=_make_result("c")),
            LexicalSidecarRecord(namespace="ns", result=_make_result("d")),
        ]
    )
    assert sidecar.entry_count == 2
    assert len(sink.events) == 1
    event = sink.events[0]
    assert isinstance(event, LexicalSidecarBoundExceeded)
    assert event.reason == "max_entries"
    assert event.rejected_count == 2
    assert event.current_entries == 2
    assert event.max_entries == 2


def test_lexical_sidecar_rejects_overflow_bytes_and_emits_event() -> None:
    sink = _RecordingSink()
    # Bound sized to fit exactly one record (text/title overhead ~262 bytes).
    sidecar = PortableLexicalSidecar(
        [], max_entries=100, max_bytes=300, event_sink=sink
    )
    sidecar.upsert_records(
        [
            LexicalSidecarRecord(namespace="ns", result=_make_result("a")),
            LexicalSidecarRecord(namespace="ns", result=_make_result("b")),
        ]
    )
    assert sidecar.entry_count == 1
    assert sink.events
    event = sink.events[0]
    assert isinstance(event, LexicalSidecarBoundExceeded)
    assert event.reason == "max_bytes"
    assert event.rejected_count == 1


def test_lexical_sidecar_no_event_when_no_overflow() -> None:
    sink = _RecordingSink()
    sidecar = PortableLexicalSidecar(
        [], max_entries=10, max_bytes=10_000_000, event_sink=sink
    )
    sidecar.upsert_records(
        [
            LexicalSidecarRecord(namespace="ns", result=_make_result("a")),
            LexicalSidecarRecord(namespace="ns", result=_make_result("b")),
        ]
    )
    assert sink.events == []
    assert sidecar.entry_count == 2


def test_lexical_sidecar_updates_existing_key_in_place() -> None:
    sink = _RecordingSink()
    sidecar = PortableLexicalSidecar(
        [], max_entries=1, max_bytes=10_000_000, event_sink=sink
    )
    sidecar.upsert_records(
        [LexicalSidecarRecord(namespace="ns", result=_make_result("a", text="old"))]
    )
    # Re-upserting the same (namespace, id) must not breach the entries cap.
    sidecar.upsert_records(
        [LexicalSidecarRecord(namespace="ns", result=_make_result("a", text="newtext"))]
    )
    assert sidecar.entry_count == 1
    assert sink.events == []


def test_lexical_sidecar_delete_releases_byte_quota() -> None:
    sidecar = PortableLexicalSidecar([], max_entries=100, max_bytes=10_000_000)
    sidecar.upsert_records(
        [
            LexicalSidecarRecord(namespace="ns", result=_make_result("a")),
            LexicalSidecarRecord(namespace="ns", result=_make_result("b")),
        ]
    )
    before = sidecar.byte_usage
    sidecar.delete_document(namespace="ns", document_id="doc-a")
    assert sidecar.byte_usage < before
    assert sidecar.entry_count == 1


def test_lexical_sidecar_rejects_non_positive_bounds() -> None:
    with pytest.raises(ValueError):
        PortableLexicalSidecar([], max_entries=0)
    with pytest.raises(ValueError):
        PortableLexicalSidecar([], max_bytes=0)


def test_lexical_sidecar_event_is_in_audit_event_types_or_known_set() -> None:
    # The new event type string is a stable label used by the caller's
    # event consumer; assert its literal so the contract doesn't drift.
    sink = _RecordingSink()
    sidecar = PortableLexicalSidecar(
        [], max_entries=1, max_bytes=10_000_000, event_sink=sink
    )
    sidecar.upsert_records(
        [
            LexicalSidecarRecord(namespace="ns", result=_make_result("a")),
            LexicalSidecarRecord(namespace="ns", result=_make_result("b")),
        ]
    )
    assert sink.events
    event = sink.events[0]
    assert isinstance(event, LexicalSidecarBoundExceeded)
    assert event.event_type == "lexical_sidecar.bound_exceeded"
