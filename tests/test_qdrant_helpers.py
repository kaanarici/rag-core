"""Qdrant-specific adapter behavior outside the cross-vector-store contract.

Wire-shape assertions (payload filter keys, prefetch channels, batching, dense
dimension preflight) plus the pure helpers in ``qdrant_collection`` /
``qdrant_shared`` / ``qdrant_store``. The portable retrieve/delete behavior is
covered by ``test_vector_store_contract.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest

from rag_core.search.pipeline import HybridRetrieve, PipelineContext, PipelineQuery
from rag_core.search.planning import default_query_plan, query_plan_preset
from rag_core.search.query_plan import UnsupportedQueryStage
from rag_core.search.providers.qdrant_collection import (
    CollectionConfig,
    assert_collection_compatible,
    build_quantization_config,
    collection_exists,
    create_collection,
)
from rag_core.search.providers.qdrant_client import create_qdrant_client
from rag_core.search.providers.qdrant_shared import (
    _KNOWN_SPARSE_VECTOR_NAMES,
    WriteLatencyTracker,
)
from rag_core.search.providers.qdrant_store import (
    QdrantVectorStore,
    _build_base_health,
    _build_healthy_health,
    _collection_query_plan_capabilities,
    _build_unhealthy_health,
    _extract_optimizer_ok,
)
from rag_core.search.providers.qdrant_write import upsert_qdrant_point_batches
from rag_core.search.types import (
    DeleteFilter,
    QueryPlanCapabilities,
    SearchQuery,
    SparseVector,
    VectorPoint,
)
from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder

_POINT_1 = "00000000-0000-4000-8000-000000000001"
_POINT_2 = "00000000-0000-4000-8000-000000000002"
_POINT_3 = "00000000-0000-4000-8000-000000000003"
_SECRET = "sk-test-secret"


class _SecretPayload:
    def __str__(self) -> str:
        return "secret=" + _SECRET


def _field_conditions(qdrant_filter: rest.Filter) -> list[rest.FieldCondition]:
    must = list(qdrant_filter.must or [])
    return [
        condition for condition in must if isinstance(condition, rest.FieldCondition)
    ]


def _store(client: object, *, dense_dimensions: int = 3072) -> QdrantVectorStore:
    """Build a QdrantVectorStore wired to a fake client.

    The constructor normally builds an AsyncQdrantClient; we swap it for the
    test fake so we can assert against the calls that hit the wire.
    """
    store = QdrantVectorStore(
        url=None,
        api_key=None,
        collection_name="docs",
        location=":memory:",
        dense_dimensions=dense_dimensions,
        quantization_enabled=True,
    )
    store._client = cast(Any, client)
    object.__setattr__(
        store,
        "_config",
        type(store._config)(
            collection_name=store._config.collection_name,
            dimensions=store._config.dimensions,
            quantization_enabled=store._config.quantization_enabled,
            is_local=store._config.is_local,
            max_concurrent=1,
            max_batch_size=2,
            policy=store._config.policy,
        ),
    )
    return store


class _FakeQdrantClient:
    def __init__(
        self,
        *,
        existing_names: list[str],
        sparse_names: list[str] | None = None,
        dense_dimensions: int = 3072,
        scroll_payload: dict[str, object] | None = None,
        count_response: object | None = None,
    ) -> None:
        self._existing_names = existing_names
        self._sparse_names = ["bm25"] if sparse_names is None else sparse_names
        self._dense_dimensions = dense_dimensions
        self._scroll_payload = scroll_payload
        self._count_response = (
            SimpleNamespace(count=7) if count_response is None else count_response
        )
        self.query_points_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.scroll_calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.get_collections_calls = 0
        self.close_calls = 0

    async def get_collections(self) -> object:
        self.get_collections_calls += 1
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self._existing_names]
        )

    async def get_collection(self, *, collection_name: str) -> object:
        assert collection_name == "docs"
        return _FakeCollectionInfo(
            size=self._dense_dimensions, sparse_names=self._sparse_names
        )

    async def query_points(self, **kwargs: Any) -> object:
        self.query_points_calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=_POINT_1,
                    score=0.75,
                    payload={
                        "text": "hello",
                        "content_type": "document",
                        "source_type": "file",
                    },
                )
            ]
        )

    async def delete(self, **kwargs: Any) -> None:
        self.delete_calls.append(kwargs)

    async def scroll(self, **kwargs: Any) -> tuple[list[object], None]:
        self.scroll_calls.append(kwargs)
        return (
            [
                SimpleNamespace(
                    payload=self._scroll_payload
                    or {
                        "document_id": "doc-from-payload",
                        "document_key": "/docs/guide.txt",
                        "content_sha256": "sha",
                        "processing_version": "3",
                    },
                )
            ],
            None,
        )

    async def count(self, **kwargs: Any) -> object:
        self.count_calls.append(kwargs)
        return self._count_response

    async def upsert(self, **kwargs: Any) -> None:
        self.upsert_calls.append(kwargs)

    async def close(self) -> None:
        self.close_calls += 1


def test_qdrant_local_location_uses_memory_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeAsyncQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        "rag_core.search.providers.qdrant_client.AsyncQdrantClient",
        FakeAsyncQdrantClient,
    )

    state = create_qdrant_client(
        url=None,
        api_key=None,
        location=":memory:",
        timeout=42,
    )

    assert state.is_local is True
    assert calls == [
        {"location": ":memory:", "timeout": 42, "check_compatibility": False}
    ]


def test_qdrant_local_location_uses_persistent_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeAsyncQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        "rag_core.search.providers.qdrant_client.AsyncQdrantClient",
        FakeAsyncQdrantClient,
    )

    state = create_qdrant_client(
        url=None,
        api_key=None,
        location=str(tmp_path / "qdrant"),
        timeout=42,
    )

    assert state.is_local is True
    assert calls == [
        {"path": str(tmp_path / "qdrant"), "timeout": 42, "check_compatibility": False}
    ]


def test_qdrant_remote_blank_api_key_is_not_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeAsyncQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        "rag_core.search.providers.qdrant_client.AsyncQdrantClient",
        FakeAsyncQdrantClient,
    )

    state = create_qdrant_client(
        url="https://qdrant.example.test",
        api_key="   ",
        location=None,
        timeout=42,
    )

    assert state.is_local is False
    assert calls == [{"url": "https://qdrant.example.test", "timeout": 42}]


def test_store_search_includes_scoped_filter_and_available_sparse_channels() -> None:
    client = _FakeQdrantClient(
        existing_names=["docs"], sparse_names=["bm25"], dense_dimensions=2
    )
    query = SearchQuery(
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        sparse_vectors={"splade": SparseVector(indices=[2], values=[2.0])},
        namespace=" team-space ",
        corpus_ids=["corpus-a", "corpus-b"],
        content_types=["document"],
        document_ids=["doc-1"],
    )

    results = asyncio.run(_store(client, dense_dimensions=2).search(query))

    call = client.query_points_calls[0]
    assert call["collection_name"] == "docs"
    assert [prefetch.using for prefetch in call["prefetch"]] == ["", "bm25"]
    conditions = _field_conditions(call["prefetch"][0].filter)
    assert [condition.key for condition in conditions] == [
        "namespace",
        "corpus_id",
        "content_type",
        "document_id",
    ]
    assert getattr(conditions[0].match, "value", None) == "team-space"
    assert getattr(conditions[1].match, "any", None) == ["corpus-a", "corpus-b"]
    assert getattr(conditions[2].match, "any", None) == ["document"]
    assert getattr(conditions[3].match, "any", None) == ["doc-1"]
    assert results[0].id == _POINT_1
    assert results[0].text == "hello"
    assert results[0].score == 0.75


def test_splade_only_collection_supports_sparse_capabilities_health_and_default_search() -> (
    None
):
    async def _run() -> tuple[QueryPlanCapabilities, dict[str, object], dict[str, Any]]:
        client = _FakeQdrantClient(
            existing_names=["docs"],
            sparse_names=["splade"],
            dense_dimensions=2,
        )
        store = _store(client, dense_dimensions=2)
        await store.ensure_collection()

        health = await store.check_health()
        query = SearchQuery(
            dense_vector=[0.1, 0.2],
            sparse_vector=SparseVector(indices=[], values=[]),
            sparse_vectors={"splade": SparseVector(indices=[2], values=[2.0])},
            namespace="team-space",
            corpus_ids=["corpus-a"],
            limit=5,
        )
        await store.search(query)
        return store.capabilities.query_plan, health, client.query_points_calls[0]

    capabilities, health, search_call = asyncio.run(_run())
    assert capabilities.dense is True
    assert capabilities.sparse is True
    assert capabilities.hybrid is True
    query_plan = cast(dict[str, object], health["query_plan"])
    assert query_plan["dense"] is True
    assert query_plan["sparse"] is True
    assert query_plan["hybrid"] is True
    assert [prefetch.using for prefetch in search_call["prefetch"]] == ["", "splade"]


def test_unknown_sparse_collection_reports_dense_only_capabilities() -> None:
    async def _run() -> tuple[QueryPlanCapabilities, dict[str, object]]:
        client = _FakeQdrantClient(
            existing_names=["docs"],
            sparse_names=["custom_sparse"],
            dense_dimensions=2,
        )
        store = _store(client, dense_dimensions=2)
        await store.ensure_collection()
        return store.capabilities.query_plan, await store.check_health()

    capabilities, health = asyncio.run(_run())
    assert capabilities.dense is True
    assert capabilities.sparse is False
    assert capabilities.hybrid is False
    query_plan = cast(dict[str, object], health["query_plan"])
    assert query_plan["dense"] is True
    assert query_plan["sparse"] is False
    assert query_plan["hybrid"] is False


def test_qdrant_write_rejects_non_uuid_point_ids_before_client_call() -> None:
    client = _FakeQdrantClient(existing_names=["docs"])
    latency = WriteLatencyTracker()
    point = VectorPoint(
        id="custom-doc-1-chunk-0",
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        payload={
            "text": "hello",
            "content_type": "document",
            "source_type": "file",
        },
    )

    with pytest.raises(ValueError, match="Qdrant point IDs must be UUID strings"):
        asyncio.run(
            upsert_qdrant_point_batches(
                client=cast(Any, client),
                collection_name="docs",
                dimensions=2,
                latency=latency,
                max_batch_size=2,
                write_sem=asyncio.Semaphore(1),
                points=[point],
                available_sparse_vector_names={"bm25"},
            )
        )

    assert client.upsert_calls == []


def test_store_search_uses_explicit_query_plan() -> None:
    client = _FakeQdrantClient(
        existing_names=["docs"], sparse_names=["bm25"], dense_dimensions=2
    )
    query = SearchQuery(
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
        limit=5,
        query_plan=default_query_plan(result_limit=5, fusion="dbsf", prefetch_limit=30),
    )

    asyncio.run(_store(client, dense_dimensions=2).search(query))

    call = client.query_points_calls[0]
    assert call["query"].fusion == rest.Fusion.DBSF
    assert [prefetch.limit for prefetch in call["prefetch"]] == [30, 30]


def test_qdrant_sparse_channel_preflight_runs_before_embedding() -> None:
    async def _run() -> tuple[FakeEmbeddingProvider, FakeSparseEmbedder]:
        client = _FakeQdrantClient(
            existing_names=["docs"],
            sparse_names=["bm25"],
            dense_dimensions=2,
        )
        embedding = FakeEmbeddingProvider()
        sparse = FakeSparseEmbedder()
        plan = default_query_plan(result_limit=5, sparse_channels=("bm25", "splade"))

        with pytest.raises(UnsupportedQueryStage, match="splade"):
            await HybridRetrieve().retrieve(
                PipelineQuery(
                    query="billing",
                    namespace="team-space",
                    corpus_ids=["corpus-a"],
                    query_plan=plan,
                ),
                PipelineContext(
                    embedding_provider=embedding,
                    sparse_embedder=sparse,
                    vector_store=_store(client, dense_dimensions=2),
                ),
            )
        return embedding, sparse

    embedding, sparse = asyncio.run(_run())
    assert embedding.embed_query_calls == []
    assert sparse.embed_query_multi_calls == []


def test_qdrant_implicit_plan_falls_back_to_dense_for_sparse_less_collection() -> None:
    async def _run() -> tuple[FakeEmbeddingProvider, FakeSparseEmbedder, _FakeQdrantClient]:
        client = _FakeQdrantClient(
            existing_names=["docs"],
            sparse_names=[],
            dense_dimensions=4,
        )
        embedding = FakeEmbeddingProvider()
        sparse = FakeSparseEmbedder()

        await HybridRetrieve().retrieve(
            PipelineQuery(
                query="billing",
                namespace="team-space",
                corpus_ids=["corpus-a"],
                limit=5,
            ),
            PipelineContext(
                embedding_provider=embedding,
                sparse_embedder=sparse,
                vector_store=_store(client, dense_dimensions=4),
            ),
        )
        return embedding, sparse, client

    embedding, sparse, client = asyncio.run(_run())
    assert embedding.embed_query_calls == ["billing"]
    assert sparse.embed_query_multi_calls == []
    assert len(client.query_points_calls) == 1


def test_store_search_allows_empty_dense_vector_for_sparse_only_plan() -> None:
    client = _FakeQdrantClient(existing_names=["docs"], sparse_names=["bm25"])
    query = SearchQuery(
        dense_vector=[],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
        limit=5,
        query_plan=query_plan_preset("sparse_only", limit=5),
    )

    asyncio.run(_store(client).search(query))

    assert len(client.query_points_calls) == 1


def test_store_search_implicit_plan_falls_back_to_dense_for_sparse_less_collection() -> None:
    client = _FakeQdrantClient(
        existing_names=["docs"],
        sparse_names=[],
        dense_dimensions=2,
    )
    query = SearchQuery(
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVector(indices=[], values=[]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
        limit=5,
    )

    asyncio.run(_store(client, dense_dimensions=2).search(query))

    [call] = client.query_points_calls
    assert call["prefetch"] is None
    assert call["using"] == ""


def test_store_search_rejects_empty_dense_vector_for_hybrid_plan_before_provider() -> (
    None
):
    client = _FakeQdrantClient(existing_names=["docs"], sparse_names=["bm25"])
    query = SearchQuery(
        dense_vector=[],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
        limit=5,
        query_plan=query_plan_preset("hybrid_dbsf", limit=5),
    )

    with pytest.raises(ValueError, match="dense query vector is required"):
        asyncio.run(_store(client).search(query))

    assert client.get_collections_calls == 0
    assert client.query_points_calls == []


def test_store_search_rejects_invalid_weighted_rrf_before_collection() -> None:
    client = _FakeQdrantClient(existing_names=["docs"], sparse_names=["bm25"])
    query = SearchQuery(
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
        limit=5,
        query_plan=default_query_plan(
            result_limit=5,
            fusion="weighted_rrf",
            fusion_weights=(1.0,),
        ),
    )

    with pytest.raises(UnsupportedQueryStage, match="one weight per prefetch"):
        asyncio.run(_store(client, dense_dimensions=2).search(query))

    assert client.get_collections_calls == 0
    assert client.query_points_calls == []


def test_store_delete_and_delete_point_ids_use_qdrant_selectors() -> None:
    client = _FakeQdrantClient(existing_names=["docs"])
    store = _store(client)

    asyncio.run(
        store.delete(DeleteFilter(namespace=" team-space ", corpus_id="corpus-1"))
    )
    asyncio.run(store.delete_point_ids([_POINT_1, _POINT_2]))

    filter_selector = client.delete_calls[0]["points_selector"]
    conditions = _field_conditions(filter_selector.filter)
    assert [condition.key for condition in conditions] == ["namespace", "corpus_id"]
    assert getattr(conditions[0].match, "value", None) == "team-space"
    assert getattr(conditions[1].match, "value", None) == "corpus-1"
    assert client.delete_calls[1]["points_selector"].points == [_POINT_1, _POINT_2]


def test_store_delete_rejects_missing_namespace_before_collection() -> None:
    client = _FakeQdrantClient(existing_names=["docs"])

    with pytest.raises(ValueError, match="namespace is required for delete"):
        asyncio.run(_store(client).delete(DeleteFilter(namespace="")))

    assert client.get_collections_calls == 0
    assert client.delete_calls == []


def test_store_document_record_uses_lookup_and_count_scope() -> None:
    client = _FakeQdrantClient(existing_names=["docs"])
    record = asyncio.run(
        _store(client).get_document_record(
            namespace=" team-space ",
            corpus_id=" corpus-1 ",
            document_key="/docs/guide.txt",
        )
    )

    lookup_filter = client.scroll_calls[0]["scroll_filter"]
    count_filter = client.count_calls[0]["count_filter"]
    lookup_keys = [condition.key for condition in _field_conditions(lookup_filter)]
    count_keys = [condition.key for condition in _field_conditions(count_filter)]
    assert lookup_keys == ["namespace", "corpus_id", "document_key"]
    assert count_keys == ["namespace", "corpus_id", "document_id"]
    assert record is not None
    assert record.document_id == "doc-from-payload"
    assert record.namespace == "team-space"
    assert record.corpus_id == "corpus-1"
    assert record.document_key == "/docs/guide.txt"
    assert record.content_sha256 == "sha"
    assert record.processing_version == "3"
    assert record.chunk_count == 7


def test_store_document_record_rejects_malformed_count_response() -> None:
    client = _FakeQdrantClient(
        existing_names=["docs"],
        count_response=SimpleNamespace(count="7"),
    )

    with pytest.raises(ValueError, match="document count response"):
        asyncio.run(
            _store(client).get_document_record(
                namespace="team-space",
                corpus_id="corpus-1",
                document_key="/docs/guide.txt",
            )
        )


@pytest.mark.parametrize(
    "payload,message",
    [
        (
            {"document_id": 123},
            "payload field 'document_id' must be a string",
        ),
        (
            {"document_id": "doc-1", "document_key": 123},
            "payload field 'document_key' must be a string",
        ),
        (
            {"document_id": "doc-1", "content_sha256": 123},
            "payload field 'content_sha256' must be a string",
        ),
        (
            {"document_id": "doc-1", "processing_version": 3},
            "payload field 'processing_version' must be a string",
        ),
    ],
    ids=[
        "document_id",
        "document_key",
        "content_sha256",
        "processing_version",
    ],
)
def test_store_document_record_rejects_malformed_payload_fields(
    payload: dict[str, object], message: str
) -> None:
    client = _FakeQdrantClient(existing_names=["docs"], scroll_payload=payload)

    with pytest.raises(ValueError, match=message):
        asyncio.run(
            _store(client).get_document_record(
                namespace="team-space",
                corpus_id="corpus-1",
                document_key="/docs/guide.txt",
            )
        )


def test_store_upsert_batches_points_and_uses_available_sparse_channels() -> None:
    client = _FakeQdrantClient(
        existing_names=["docs"], sparse_names=["bm25", "splade"], dense_dimensions=2
    )
    points = [
        VectorPoint(
            id=_POINT_1,
            dense_vector=[0.1, 0.2],
            sparse_vector=SparseVector(indices=[1], values=[1.0]),
            sparse_vectors={"splade": SparseVector(indices=[2], values=[2.0])},
            payload={"text": "hello"},
        ),
        VectorPoint(
            id=_POINT_2,
            dense_vector=[0.3, 0.4],
            sparse_vector=SparseVector(indices=[3], values=[3.0]),
            payload={"text": "world"},
        ),
        VectorPoint(
            id=_POINT_3,
            dense_vector=[0.5, 0.6],
            sparse_vector=SparseVector(indices=[4], values=[4.0]),
            payload={"text": "again"},
        ),
    ]

    asyncio.run(_store(client, dense_dimensions=2).upsert(points))

    assert [len(call["points"]) for call in client.upsert_calls] == [2, 1]
    first_point = client.upsert_calls[0]["points"][0]
    assert set(first_point.vector) == {"", "bm25", "splade"}


def test_store_upsert_rejects_known_sparse_vectors_missing_from_collection() -> None:
    client = _FakeQdrantClient(
        existing_names=["docs"], sparse_names=["bm25"], dense_dimensions=2
    )
    point = VectorPoint(
        id=_POINT_1,
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVector(indices=[1], values=[1.0]),
        sparse_vectors={"splade": SparseVector(indices=[2], values=[2.0])},
        payload={"text": "hello"},
    )

    with pytest.raises(ValueError, match="missing sparse vector channels.*splade"):
        asyncio.run(_store(client, dense_dimensions=2).upsert([point]))

    assert client.upsert_calls == []


def test_qdrant_store_exposes_dense_dimensions_for_pre_upsert_validation() -> None:
    store = _store(_FakeQdrantClient(existing_names=["docs"]))

    assert store.capabilities.dense_vector_dimensions == 3072


def test_qdrant_store_rejects_wrong_dense_dimensions_before_provider_upsert() -> None:
    client = _FakeQdrantClient(existing_names=["docs"])
    store = _store(client)
    point = VectorPoint(
        id=_POINT_1,
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVector(indices=[], values=[]),
        payload={"text": "hello"},
    )

    with pytest.raises(
        ValueError,
        match="qdrant dense vector dimension mismatch at point index 0",
    ):
        asyncio.run(store.upsert([point]))

    assert client.upsert_calls == []


def test_qdrant_store_rejects_unsupported_payload_object_before_provider_upsert() -> None:
    client = _FakeQdrantClient(existing_names=["docs"], dense_dimensions=2)
    store = _store(client, dense_dimensions=2)
    point = VectorPoint(
        id=_POINT_1,
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVector(indices=[], values=[]),
        payload={"text": _SecretPayload()},
    )

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(store.upsert([point]))

    message = str(exc_info.value)
    assert message == "vector payload contains unsupported value type: _SecretPayload"
    assert _SECRET not in message
    assert client.upsert_calls == []


def test_qdrant_store_rejects_wrong_query_dimensions_before_provider_search() -> None:
    client = _FakeQdrantClient(existing_names=["docs"])
    store = _store(client, dense_dimensions=3)
    query = SearchQuery(
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVector(indices=[], values=[]),
        namespace="team-space",
        corpus_ids=["corpus-a"],
    )

    with pytest.raises(ValueError, match="qdrant dense query dimension mismatch"):
        asyncio.run(store.search(query))

    assert client.get_collections_calls == 0
    assert client.query_points_calls == []


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (
            {
                "namespace": "",
                "corpus_id": "corpus-1",
                "document_id": "doc-1",
                "document_key": None,
            },
            "namespace is required for get_document_record",
        ),
        (
            {
                "namespace": "team-space",
                "corpus_id": "",
                "document_id": "doc-1",
                "document_key": None,
            },
            "corpus_id is required for get_document_record",
        ),
        (
            {
                "namespace": "team-space",
                "corpus_id": "corpus-1",
                "document_id": None,
                "document_key": None,
            },
            "document_id or document_key is required for get_document_record",
        ),
    ],
    ids=["missing_namespace", "missing_corpus_id", "missing_both_identifiers"],
)
def test_store_document_record_rejects_missing_lookup_fields(
    kwargs: dict[str, object], message: str
) -> None:
    store = _store(_FakeQdrantClient(existing_names=["docs"]))
    with pytest.raises(ValueError, match=message):
        asyncio.run(store.get_document_record(**kwargs))  # type: ignore[arg-type]


class _FakeClient:
    def __init__(self) -> None:
        self.create_collection_calls: list[dict[str, object]] = []
        self.create_payload_index_calls: list[tuple[str, rest.PayloadSchemaType]] = []

    async def create_collection(self, **kwargs: object) -> None:
        self.create_collection_calls.append(kwargs)

    async def create_payload_index(
        self,
        *,
        collection_name: str,
        field_name: str,
        field_schema: rest.PayloadSchemaType,
    ) -> None:
        assert collection_name == "docs"
        self.create_payload_index_calls.append((field_name, field_schema))


class _FakeConfigParams:
    def __init__(
        self,
        *,
        size: int,
        sparse_names: list[str] | None = None,
        dense_names: list[str] | None = None,
    ) -> None:
        self.vectors = {
            name: type("Dense", (), {"size": size})()
            for name in ([""] if dense_names is None else dense_names)
        }
        self.sparse_vectors = (
            {name: object() for name in sparse_names}
            if sparse_names is not None
            else None
        )


class _FakeCollectionInfo:
    def __init__(
        self,
        *,
        size: int,
        sparse_names: list[str] | None = None,
        dense_names: list[str] | None = None,
    ) -> None:
        self.config = type(
            "Config",
            (),
            {
                "params": _FakeConfigParams(
                    size=size,
                    sparse_names=sparse_names,
                    dense_names=dense_names,
                )
            },
        )()


class _FakeStatus:
    def __init__(self, value: str) -> None:
        self.value = value


class _FakeOptimizerStatus:
    def __init__(self, status: str) -> None:
        self.status = _FakeStatus(status)


class _FakeHealthInfo:
    def __init__(
        self, *, points_count: int, status: str, optimizer_status: object | None
    ) -> None:
        self.points_count = points_count
        self.status = _FakeStatus(status)
        self.optimizer_status = optimizer_status


def test_collection_exists_checks_name_membership() -> None:
    assert collection_exists(existing_names={"a", "b"}, collection_name="a")
    assert not collection_exists(existing_names={"a", "b"}, collection_name="c")


def test_build_quantization_config_is_toggleable() -> None:
    assert build_quantization_config(enabled=False) is None
    quantization = build_quantization_config(enabled=True)
    assert quantization is not None
    assert quantization.scalar is not None
    assert quantization.scalar.type == rest.ScalarType.INT8


def test_assert_collection_compatible_allows_dense_only_collection_metadata() -> None:
    assert (
        assert_collection_compatible(
            collection_name="docs",
            dimensions=3072,
            collection_info=_FakeCollectionInfo(size=3072, sparse_names=None),
        )
        == frozenset()
    )


def test_assert_collection_compatible_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="uses 3072 dimensions"):
        assert_collection_compatible(
            collection_name="docs",
            dimensions=1536,
            collection_info=_FakeCollectionInfo(size=3072, sparse_names=["bm25"]),
        )


def test_assert_collection_compatible_rejects_named_dense_vector_collection() -> None:
    with pytest.raises(ValueError, match="unsupported dense vector channels"):
        assert_collection_compatible(
            collection_name="docs",
            dimensions=3072,
            collection_info=_FakeCollectionInfo(
                size=3072,
                sparse_names=["bm25"],
                dense_names=["text"],
            ),
        )


def test_assert_collection_compatible_reports_available_non_primary_sparse_channels() -> None:
    assert assert_collection_compatible(
        collection_name="docs",
        dimensions=3072,
        collection_info=_FakeCollectionInfo(size=3072, sparse_names=["splade"]),
    ) == frozenset({"splade"})


def test_upsert_qdrant_point_batches_runs_batches_concurrently() -> None:
    async def _run() -> None:
        points = [
            VectorPoint(
                id=f"00000000-0000-4000-8000-{index:012d}",
                dense_vector=[0.1, 0.2],
                sparse_vector=SparseVector(indices=[index], values=[1.0]),
                payload={"text": f"point-{index}"},
            )
            for index in range(3)
        ]
        active = 0
        max_active = 0
        seen_batch_sizes: list[int] = []

        async def _fake_upsert_with_fallback(**kwargs: object) -> None:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            seen_batch_sizes.append(len(cast(list[object], kwargs["points"])))
            await asyncio.sleep(0.01)
            active -= 1

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            "rag_core.search.providers.qdrant_write.upsert_with_fallback",
            _fake_upsert_with_fallback,
        )
        try:
            await upsert_qdrant_point_batches(
                client=cast(Any, object()),
                collection_name="docs",
                dimensions=2,
                latency=WriteLatencyTracker(),
                max_batch_size=1,
                write_sem=asyncio.Semaphore(2),
                points=points,
                available_sparse_vector_names=frozenset({"bm25"}),
            )
        finally:
            monkeypatch.undo()

        assert seen_batch_sizes == [1, 1, 1]
        assert max_active == 2

    asyncio.run(_run())


def test_assert_collection_compatible_accepts_embedded_qdrant_info() -> None:
    async def check() -> frozenset[str]:
        client = AsyncQdrantClient(location=":memory:", check_compatibility=False)
        try:
            await create_collection(
                client=client,
                config=CollectionConfig(
                    collection_name="docs",
                    dimensions=8,
                    quantization_enabled=False,
                    is_local=True,
                ),
            )
            collection_info = await client.get_collection(collection_name="docs")
            return assert_collection_compatible(
                collection_name="docs",
                dimensions=8,
                collection_info=collection_info,
            )
        finally:
            await client.close()

    sparse_names = asyncio.run(check())

    assert sparse_names == _KNOWN_SPARSE_VECTOR_NAMES


def test_extract_optimizer_ok_supports_multiple_shapes() -> None:
    assert _extract_optimizer_ok(type("Status", (), {"ok": True})()) is True
    assert _extract_optimizer_ok(_FakeOptimizerStatus(status="green")) is True
    assert _extract_optimizer_ok(_FakeOptimizerStatus(status="red")) is False
    assert _extract_optimizer_ok(object()) is None


def test_build_healthy_health_includes_latency_and_optimizer_fields() -> None:
    latency = WriteLatencyTracker()
    latency.record(1.0)
    latency.record(2.0)

    base = _build_base_health(collection_name="docs", dimensions=3072)
    health = _build_healthy_health(
        base_health=base,
        collection_info=_FakeHealthInfo(
            points_count=5,
            status="green",
            optimizer_status=_FakeOptimizerStatus(status="ok"),
        ),
        latency=latency,
    )
    assert health["healthy"] is True
    assert health["collection"] == "docs"
    assert health["dimensions"] == 3072
    assert health["points_count"] == 5
    assert health["status"] == "green"
    assert health["optimizer_ok"] is True
    assert health["write_latency_p50"] is not None
    assert health["write_latency_p95"] is not None
    assert health["write_latency_samples"] == 2


def test_qdrant_health_reports_dense_only_query_plan_for_dense_only_collection() -> None:
    collection = _FakeCollectionInfo(size=3072, sparse_names=[])

    capabilities = _collection_query_plan_capabilities(collection)
    health = _build_healthy_health(
        base_health=_build_base_health(collection_name="docs", dimensions=3072),
        collection_info=collection,
        latency=WriteLatencyTracker(),
    )

    assert capabilities.dense is True
    assert capabilities.sparse is False
    query_plan = cast(dict[str, object], health["query_plan"])
    assert query_plan["dense"] is True
    assert query_plan["sparse"] is False
    assert query_plan["hybrid"] is False


def test_qdrant_health_treats_missing_sparse_metadata_as_dense_only() -> None:
    collection = _FakeCollectionInfo(size=3072, sparse_names=None)

    capabilities = _collection_query_plan_capabilities(collection)
    health = _build_healthy_health(
        base_health=_build_base_health(collection_name="docs", dimensions=3072),
        collection_info=collection,
        latency=WriteLatencyTracker(),
    )

    assert capabilities.dense is True
    assert capabilities.sparse is False
    query_plan = cast(dict[str, object], health["query_plan"])
    assert query_plan["dense"] is True
    assert query_plan["sparse"] is False
    assert query_plan["hybrid"] is False


def test_qdrant_health_treats_unknown_sparse_metadata_as_dense_only() -> None:
    collection = _FakeCollectionInfo(size=3072, sparse_names=["custom_sparse"])

    capabilities = _collection_query_plan_capabilities(collection)
    health = _build_healthy_health(
        base_health=_build_base_health(collection_name="docs", dimensions=3072),
        collection_info=collection,
        latency=WriteLatencyTracker(),
    )

    assert capabilities.dense is True
    assert capabilities.sparse is False
    query_plan = cast(dict[str, object], health["query_plan"])
    assert query_plan["dense"] is True
    assert query_plan["sparse"] is False
    assert query_plan["hybrid"] is False


def test_build_unhealthy_health_omits_exception_message() -> None:
    base = _build_base_health(collection_name="docs", dimensions=3072)
    health = _build_unhealthy_health(
        base_health=base,
        exc=RuntimeError("private adapter detail"),
    )
    assert health["healthy"] is False
    assert health["adapter"] == "qdrant"
    assert "backend" not in health
    assert health["error"] == "RuntimeError"
    assert "private adapter detail" not in str(health)


@pytest.mark.parametrize(
    "is_local,expected_payload_index_calls",
    [(True, 0), (False, 8)],
    ids=["local_skips_payload_indexes", "remote_writes_payload_indexes"],
)
def test_create_collection_payload_indexes_depend_on_deployment(
    is_local: bool, expected_payload_index_calls: int
) -> None:
    client = _FakeClient()
    asyncio.run(
        create_collection(
            client=cast(Any, client),
            config=CollectionConfig(
                collection_name="docs",
                dimensions=3072,
                quantization_enabled=not is_local,
                is_local=is_local,
            ),
        )
    )

    assert len(client.create_collection_calls) == 1
    assert len(client.create_payload_index_calls) == expected_payload_index_calls
