"""Qdrant adapter: collection lifecycle and the VectorStore entry.

Shared payload conversion and write-tuning helpers live in ``qdrant_payloads``
so query/write modules can import them without a cycle through this entry.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.document_versions import (
    KIND,
    REVISION,
    HEAD,
    CHUNK,
    Candidate,
    committed_search,
    delete_document as delete_document_version,
    head_id,
    parse_head,
    replace_document,
)
from rag_core.search.document_records import validate_document_lookup_inputs
from dataclasses import replace
from .qdrant_filters import build_document_lookup_filter
from rag_core.search.provider_protocols import (
    StoreCapabilities,
)
from rag_core.search.request_models import (
    DeleteFilter,
    SearchQuery,
    StoredDocumentRecord,
)
from rag_core.search.vector_models import (
    SearchResult,
    VectorPoint,
)

from .qdrant_payloads import (
    _DENSE_VECTOR_NAME,
    _count_result_value,
    _KNOWN_SPARSE_VECTOR_NAMES,
    _PRIMARY_SPARSE_VECTOR_NAME,
    WriteLatencyTracker,
    compute_write_params,
    get_qdrant_document_record,
)
from .qdrant_query import (
    search_qdrant_points,
    validate_qdrant_delete_filter,
    validate_qdrant_search_request,
)
from .qdrant_write import (
    delete_qdrant_filter,
    delete_qdrant_point_ids,
    upsert_qdrant_point_batches,
)
from .vector_dimensions import (
    validate_point_dense_dimensions,
)
from .vector_store_capabilities import (
    QDRANT_METADATA_FILTERS,
)

logger = logging.getLogger(__name__)


EMBEDDING_MODEL_METADATA_KEY = "rag_core.embedding_model"
EMBEDDING_DIMENSIONS_METADATA_KEY = "rag_core.embedding_dimensions"


@dataclass(frozen=True)
class CollectionConfig:
    collection_name: str
    dimensions: int
    quantization_enabled: bool
    is_local: bool
    sparse_enabled: bool = False
    policy: VectorStorePolicy = DEFAULT_POLICY


def collection_exists(*, existing_names: set[str], collection_name: str) -> bool:
    return collection_name in existing_names


def build_quantization_config(*, enabled: bool) -> rest.ScalarQuantization | None:
    if not enabled:
        return None
    return rest.ScalarQuantization(
        scalar=rest.ScalarQuantizationConfig(
            type=rest.ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        )
    )


async def create_collection(
    *,
    client: AsyncQdrantClient,
    config: CollectionConfig,
    collection_metadata: dict[str, Any] | None = None,
    create_indexes: bool = True,
) -> None:
    create_kwargs: dict[str, Any] = {
        "collection_name": config.collection_name,
        "vectors_config": _build_vectors_config(config),
        "hnsw_config": rest.HnswConfigDiff(ef_construct=100),
        "quantization_config": build_quantization_config(
            enabled=config.quantization_enabled
        ),
        "on_disk_payload": True,
    }
    if config.sparse_enabled:
        create_kwargs["sparse_vectors_config"] = {
            _PRIMARY_SPARSE_VECTOR_NAME: rest.SparseVectorParams(
                modifier=rest.Modifier.IDF
            ),
        }
    if collection_metadata:
        create_kwargs["metadata"] = collection_metadata
    await client.create_collection(**create_kwargs)

    if config.is_local or not create_indexes:
        return
    await create_payload_indexes(
        client=client,
        collection_name=config.collection_name,
        policy=config.policy,
    )


def pack_embedding_identity_metadata(
    *,
    embedding_model: str | None,
    dimensions: int,
) -> dict[str, Any] | None:
    if not embedding_model:
        return None
    return {
        EMBEDDING_MODEL_METADATA_KEY: embedding_model,
        EMBEDDING_DIMENSIONS_METADATA_KEY: dimensions,
    }


def extract_collection_metadata(collection_info: object) -> Mapping[str, Any]:
    config = getattr(collection_info, "config", None)
    metadata = getattr(config, "metadata", None)
    if isinstance(metadata, Mapping):
        return metadata
    return {}


def assert_embedding_identity_matches(
    *,
    collection_name: str,
    expected_model: str,
    expected_dimensions: int,
    collection_metadata: Mapping[str, Any],
    mismatch_cls: type[ValueError],
) -> None:
    stored_model = collection_metadata.get(EMBEDDING_MODEL_METADATA_KEY)
    stored_dim = collection_metadata.get(EMBEDDING_DIMENSIONS_METADATA_KEY)
    if stored_model is None:
        raise mismatch_cls(
            "Qdrant collection has no embedding model identity; reingest into a fresh collection"
        )
    if stored_model is not None and stored_model != expected_model:
        raise mismatch_cls(
            f"Qdrant collection {collection_name!r} was created with embedding "
            f"model {stored_model!r}, but the current process uses "
            f"{expected_model!r}. Use a different collection name or reindex."
        )
    if (
        isinstance(stored_dim, int)
        and not isinstance(stored_dim, bool)
        and stored_dim != expected_dimensions
    ):
        raise mismatch_cls(
            f"Qdrant collection {collection_name!r} was created with "
            f"{stored_dim} embedding dimensions, but the current embedder "
            f"uses {expected_dimensions}. Use a different collection name or "
            "reindex."
        )


def _build_vectors_config(config: CollectionConfig) -> dict[str, rest.VectorParams]:
    return {
        _DENSE_VECTOR_NAME: rest.VectorParams(
            size=config.dimensions,
            distance=rest.Distance.COSINE,
            on_disk=True,
        ),
    }


def assert_collection_compatible(
    *,
    collection_name: str,
    dimensions: int,
    collection_info: object,
) -> frozenset[str]:
    dense_vector_names = extract_dense_vector_names(collection_info)
    if dense_vector_names is not None and dense_vector_names != frozenset(
        {_DENSE_VECTOR_NAME}
    ):
        available = ", ".join(repr(name) for name in sorted(dense_vector_names))
        raise ValueError(
            "Existing collection %s uses unsupported dense vector channels (%s). "
            "QdrantVectorStore supports only the primary dense vector channel. "
            "Use a different collection name or reindex with a compatible collection."
            % (collection_name, available or "none")
        )

    actual_dimensions = extract_dense_vector_size(collection_info)
    if actual_dimensions is not None and actual_dimensions != dimensions:
        raise ValueError(
            "Existing collection %s uses %d dimensions, but the current embedding provider uses %d. "
            "Use a different collection name or reindex with a matching embedding configuration."
            % (collection_name, actual_dimensions, dimensions)
        )

    sparse_vector_names = extract_sparse_vector_names(collection_info)
    if sparse_vector_names is None:
        return frozenset()
    if _PRIMARY_SPARSE_VECTOR_NAME not in sparse_vector_names:
        return frozenset(sparse_vector_names)
    return frozenset(sparse_vector_names)


def extract_dense_vector_size(collection_info: object) -> int | None:
    config = getattr(collection_info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)

    size = getattr(vectors, "size", None)
    if isinstance(size, int):
        return size

    if isinstance(vectors, Mapping):
        dense = vectors.get(_DENSE_VECTOR_NAME)
        size = getattr(dense, "size", None) if dense is not None else None
        if isinstance(size, int):
            return size

    return None


def extract_dense_vector_names(collection_info: object) -> frozenset[str] | None:
    config = getattr(collection_info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    if not isinstance(vectors, Mapping):
        return None
    return frozenset(str(name) for name in vectors)


def extract_sparse_vector_names(collection_info: object) -> frozenset[str] | None:
    config = getattr(collection_info, "config", None)
    params = getattr(config, "params", None)
    sparse_vectors = getattr(params, "sparse_vectors", None)
    if sparse_vectors is None:
        return None

    if isinstance(sparse_vectors, Mapping):
        return frozenset(str(name) for name in sparse_vectors if str(name))
    return None


async def create_payload_indexes(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    policy: VectorStorePolicy,
    existing_schema: Mapping[str, rest.PayloadIndexInfo] | None = None,
) -> None:
    tenant_field = policy.tenant_payload_field
    for field_name, schema_type in collection_index_fields(policy):
        existing = (existing_schema or {}).get(field_name)
        if existing is not None:
            if existing.data_type != schema_type:
                raise ValueError(
                    f"Qdrant payload index {field_name!r} must use {schema_type.value}"
                )
            continue
        field_schema: rest.PayloadSchemaType | rest.KeywordIndexParams = schema_type
        if tenant_field is not None and field_name == tenant_field:
            field_schema = rest.KeywordIndexParams(
                type=rest.KeywordIndexType.KEYWORD,
                is_tenant=True,
            )
        result = await client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )
        if result.status != rest.UpdateStatus.COMPLETED:
            raise RuntimeError(
                f"Qdrant payload index {field_name!r} is not ready: {result.status}"
            )


def collection_index_fields(
    policy: VectorStorePolicy,
) -> tuple[tuple[str, rest.PayloadSchemaType], ...]:
    return (
        (KIND, rest.PayloadSchemaType.KEYWORD),
        (REVISION, rest.PayloadSchemaType.KEYWORD),
        (policy.tenant_id_field, rest.PayloadSchemaType.KEYWORD),
        (policy.collection_field, rest.PayloadSchemaType.KEYWORD),
        (policy.document_id_field, rest.PayloadSchemaType.KEYWORD),
        (policy.document_key_field, rest.PayloadSchemaType.KEYWORD),
        (policy.content_sha256_field, rest.PayloadSchemaType.KEYWORD),
        (policy.processing_version_field, rest.PayloadSchemaType.KEYWORD),
        (policy.content_type_field, rest.PayloadSchemaType.KEYWORD),
        (policy.source_type_field, rest.PayloadSchemaType.KEYWORD),
    )


class _QdrantHealthClient(Protocol):
    async def get_collection(self, *, collection_name: str) -> object: ...


async def check_qdrant_health(
    *,
    client: _QdrantHealthClient,
    collection_name: str,
    dimensions: int,
    latency: WriteLatencyTracker,
    logger: logging.Logger,
) -> dict[str, object]:
    health = _build_base_health(
        collection_name=collection_name,
        dimensions=dimensions,
    )
    try:
        info = await client.get_collection(collection_name=collection_name)
    except Exception as exc:
        logger.warning(
            "Qdrant health check failed: provider=%s error_type=%s "
            "collection_fingerprint=%s",
            "qdrant",
            type(exc).__name__,
            _collection_fingerprint(collection_name),
        )
        return _build_unhealthy_health(base_health=health, exc=exc)
    return _build_healthy_health(
        base_health=health,
        collection_info=info,
        latency=latency,
    )


def _collection_fingerprint(collection_name: str) -> str:
    return hashlib.sha256(collection_name.encode("utf-8")).hexdigest()[:12]


def _build_base_health(*, collection_name: str, dimensions: int) -> dict[str, object]:
    return {
        "healthy": False,
        "adapter": "qdrant",
        "collection": collection_name,
        "dimensions": dimensions,
    }


def _build_healthy_health(
    *,
    base_health: dict[str, object],
    collection_info: object,
    latency: WriteLatencyTracker,
) -> dict[str, object]:
    health = dict(base_health)
    health["healthy"] = True
    health["points_count"] = getattr(collection_info, "points_count", None)

    raw_status = getattr(collection_info, "status", None)
    if raw_status is None:
        health["status"] = "unknown"
    else:
        health["status"] = (
            raw_status.value if hasattr(raw_status, "value") else str(raw_status)
        )

    optimizer_ok = _extract_optimizer_ok(
        getattr(collection_info, "optimizer_status", None)
    )
    if optimizer_ok is not None:
        health["optimizer_ok"] = optimizer_ok
    health["search_modes"] = (
        ["dense", "hybrid", "lexical"]
        if extract_sparse_vector_names(collection_info)
        else ["dense"]
    )

    health["write_latency_p50"] = latency.p50
    health["write_latency_p95"] = latency.p95
    health["write_latency_samples"] = latency.sample_count
    return health


def _build_unhealthy_health(
    *, base_health: dict[str, object], exc: Exception
) -> dict[str, object]:
    health = dict(base_health)
    health["error"] = type(exc).__name__
    return health


def _extract_optimizer_ok(optimizer_status: object | None) -> bool | None:
    if optimizer_status is None:
        return None

    if hasattr(optimizer_status, "ok"):
        return bool(getattr(optimizer_status, "ok"))

    if hasattr(optimizer_status, "status"):
        raw_status = getattr(optimizer_status, "status")
        status_text = (
            raw_status.value if hasattr(raw_status, "value") else str(raw_status)
        )
        return status_text.lower() in {"ok", "green", "healthy"}

    return None


class EmbeddingIdentityMismatch(ValueError):
    """Refuse a collection produced by a different embedder."""


class QdrantVectorStore:
    """One connection and lifecycle for embedded or remote Qdrant."""

    def __init__(
        self,
        url: str | None,
        api_key: str | None,
        collection_name: str,
        location: str | None = None,
        dense_dimensions: int = 3072,
        quantization_enabled: bool = True,
        sparse_enabled: bool = False,
        policy: VectorStorePolicy = DEFAULT_POLICY,
        embedding_model: str | None = None,
    ) -> None:
        if bool(url) == bool(location):
            raise ValueError(
                "QdrantVectorStore requires exactly one of url or location; use QdrantConfig(location=':memory:'), pass url=..., or inject vector_store=... into Engine."
            )
        kwargs: dict[str, Any] = {"timeout": 120}
        if location is not None:
            kwargs.update(
                {
                    "location" if location == ":memory:" else "path": location,
                    "check_compatibility": False,
                }
            )
        else:
            kwargs["url"] = url
            if api_key and api_key.strip():
                kwargs["api_key"] = api_key.strip()
        self._client = AsyncQdrantClient(**kwargs)
        self._config = CollectionConfig(
            collection_name=collection_name,
            dimensions=dense_dimensions,
            quantization_enabled=quantization_enabled,
            is_local=location is not None,
            sparse_enabled=sparse_enabled,
            policy=policy,
        )
        self._policy = policy
        self._embedding_model = embedding_model
        self._max_concurrent, self._max_batch_size = compute_write_params(
            dense_dimensions
        )
        self._write_sem = asyncio.Semaphore(self._max_concurrent)
        self._write_loop: asyncio.AbstractEventLoop | None = None
        self._active_writes = 0
        self._latency = WriteLatencyTracker()
        self._ready = False
        self._ready_lock = asyncio.Lock()
        self._sparse_names = (
            _KNOWN_SPARSE_VECTOR_NAMES if sparse_enabled else frozenset()
        )
        logger.info(
            "QdrantVectorStore initialized: provider=qdrant dense_dimensions=%d max_concurrent=%d max_batch_size=%d quantization=%s local=%s collection_fingerprint=%s",
            dense_dimensions,
            self._max_concurrent,
            self._max_batch_size,
            quantization_enabled,
            location is not None,
            _collection_fingerprint(collection_name),
        )

    async def __aenter__(self) -> QdrantVectorStore:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def capabilities(self) -> StoreCapabilities:
        sparse = self._config.sparse_enabled and (
            not self._ready or _PRIMARY_SPARSE_VECTOR_NAME in self._sparse_names
        )
        return StoreCapabilities(
            per_point_delete=True,
            dense_vector_dimensions=self._config.dimensions,
            hybrid=sparse,
            lexical=sparse,
            metadata_filter=QDRANT_METADATA_FILTERS,
        )

    async def close(self) -> None:
        await self._client.close()

    def bind_embedding_space(self, model_name: str) -> None:
        if self._embedding_model is not None and self._embedding_model != model_name:
            raise EmbeddingIdentityMismatch(
                "Store is bound to a different embedding model; use a fresh index"
            )
        if self._embedding_model is None:
            self._embedding_model = model_name
            self._ready = False

    async def ensure_collection(self) -> None:
        if self._ready:
            return
        async with self._ready_lock:
            if self._ready:
                return  # type: ignore[unreachable]
            config = self._config
            existing = await self._client.get_collections()
            if config.collection_name not in {c.name for c in existing.collections}:
                try:
                    await create_collection(
                        client=self._client,
                        config=config,
                        collection_metadata=pack_embedding_identity_metadata(
                            embedding_model=self._embedding_model,
                            dimensions=config.dimensions,
                        ),
                        create_indexes=False,
                    )
                except (UnexpectedResponse, ValueError) as exc:
                    if getattr(exc, "status_code", None) != 409 and not any(
                        word in str(exc).lower()
                        for word in ("already exists", "conflict")
                    ):
                        raise
                else:
                    logger.info(
                        "Created Qdrant collection: provider=qdrant dense_dimensions=%d collection_fingerprint=%s",
                        config.dimensions,
                        _collection_fingerprint(config.collection_name),
                    )
                    if config.is_local:
                        self._ready = True
                        return
            info = await self._client.get_collection(
                collection_name=config.collection_name
            )
            available = assert_collection_compatible(
                collection_name=config.collection_name,
                dimensions=config.dimensions,
                collection_info=info,
            )
            if config.sparse_enabled and _PRIMARY_SPARSE_VECTOR_NAME not in available:
                raise ValueError(
                    f"Qdrant collection {config.collection_name!r} is missing required sparse vector channels: bm25; available sparse channels: {', '.join(sorted(available)) or 'none'}. Use a different collection name or delete and rebuild this collection before enabling sparse retrieval."
                )
            self._sparse_names = available if config.sparse_enabled else frozenset()
            if self._embedding_model is not None:
                assert_embedding_identity_matches(
                    collection_name=config.collection_name,
                    expected_model=self._embedding_model,
                    expected_dimensions=config.dimensions,
                    collection_metadata=extract_collection_metadata(info),
                    mismatch_cls=EmbeddingIdentityMismatch,
                )
            if not config.is_local:
                params = info.config.params
                replication = getattr(params, "replication_factor", None)
                write_consistency = getattr(params, "write_consistency_factor", None)
                if (
                    isinstance(replication, bool)
                    or not isinstance(replication, int)
                    or replication <= 0
                    or isinstance(write_consistency, bool)
                    or not isinstance(write_consistency, int)
                    or write_consistency != replication
                ):
                    raise ValueError(
                        "Remote Qdrant managed indexes require a positive "
                        "replication_factor and write_consistency_factor equal to "
                        "replication_factor; configure the collection before retrying"
                    )
                await create_payload_indexes(
                    client=self._client,
                    collection_name=config.collection_name,
                    policy=self._policy,
                    existing_schema=info.payload_schema,
                )
            logger.info(
                "Qdrant collection already exists: provider=qdrant dense_dimensions=%d sparse_channels=%d collection_fingerprint=%s",
                config.dimensions,
                len(self._sparse_names),
                _collection_fingerprint(config.collection_name),
            )
            self._ready = True

    async def check_health(self) -> dict[str, object]:
        return await check_qdrant_health(
            client=self._client,
            collection_name=self._config.collection_name,
            dimensions=self._config.dimensions,
            latency=self._latency,
            logger=logger,
        )

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        validate_point_dense_dimensions(
            points,
            dense_dimensions=self._config.dimensions,
            provider_name="qdrant",
        )
        await self.ensure_collection()

        loop = asyncio.get_running_loop()
        if self._config.is_local and self._write_loop is not loop:
            if self._active_writes:
                raise RuntimeError("Local Qdrant writes cannot run across concurrent event loops")
            if self._write_loop is not None:
                self._write_sem = asyncio.Semaphore(self._max_concurrent)
            self._write_loop = loop
        self._active_writes += 1
        try:
            await upsert_qdrant_point_batches(
                client=self._client,
                collection_name=self._config.collection_name,
                dimensions=self._config.dimensions,
                latency=self._latency,
                max_batch_size=self._max_batch_size,
                write_sem=self._write_sem,
                max_concurrent=self._max_concurrent,
                points=points,
                available_sparse_vector_names=(self._sparse_names),
            )
        finally:
            self._active_writes -= 1

    async def replace_document(
        self, record: StoredDocumentRecord, points: Sequence[VectorPoint]
    ) -> None:
        await replace_document(self, record, points, self._policy)

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        return await committed_search(
            query, self._search_candidates, self._read_heads, self._policy
        )

    async def _search_candidates(self, query: SearchQuery) -> list[Candidate]:
        tenant_id = validate_qdrant_search_request(
            query,
            dense_dimensions=self._config.dimensions,
        )
        if query.has_empty_allowlist():
            return []
        await self.ensure_collection()

        return await search_qdrant_points(
            client=self._client,
            collection_name=self._config.collection_name,
            query=query,
            tenant_id=tenant_id,
            policy=self._policy,
            available_sparse_vector_names=(self._sparse_names),
            consistency=None if self._config.is_local else rest.ReadConsistencyType.ALL,
        )

    async def delete(self, filter: DeleteFilter) -> None:

        if filter.document_id is not None and filter.collection is not None:
            await self.ensure_collection()
            await delete_document_version(
                self,
                StoredDocumentRecord(
                    tenant_id=filter.tenant_id or "",
                    collection=filter.collection,
                    document_id=filter.document_id,
                ),
                self._policy,
            )
            # Unversioned low-level rows have no revision; managed rows are
            # removed only by their captured version, never by a broad rollback.
            await self._delete_legacy_document(filter)
            return
        tenant_id = validate_qdrant_delete_filter(filter)
        await self.ensure_collection()

        await delete_qdrant_filter(
            client=self._client,
            collection_name=self._config.collection_name,
            filter_values=filter,
            tenant_id=tenant_id,
            policy=self._policy,
        )

    async def delete_point_ids(self, point_ids: Sequence[str]) -> None:
        if not point_ids:
            return
        await self.ensure_collection()
        await delete_qdrant_point_ids(
            client=self._client,
            collection_name=self._config.collection_name,
            point_ids=point_ids,
        )

    async def _delete_legacy_document(self, values: DeleteFilter) -> None:
        scope = build_document_lookup_filter(
            tenant_id=values.tenant_id or "",
            collection=values.collection,
            document_id=values.document_id,
            document_key=None,
            policy=self._policy,
        )
        await self._client.delete(
            collection_name=self._config.collection_name,
            points_selector=rest.FilterSelector(
                filter=rest.Filter(
                    must=[
                        scope,
                        rest.IsEmptyCondition(is_empty=rest.PayloadField(key=KIND)),
                    ]
                )
            ),
            wait=True,
        )

    async def _read_heads(self, ids: Sequence[str]) -> dict[str, dict[str, object]]:
        if not ids:
            return {}
        await self.ensure_collection()
        rows = await self._client.retrieve(
            collection_name=self._config.collection_name,
            ids=list(ids),
            with_payload=True,
            with_vectors=False,
            consistency=None if self._config.is_local else rest.ReadConsistencyType.ALL,
        )
        return {str(row.id): dict(row.payload or {}) for row in rows}

    async def _publish_head(
        self, id: str, payload: dict[str, object], expected_revision: str | None
    ) -> bool:
        result = await self._client.upsert(
            collection_name=self._config.collection_name,
            points=[rest.PointStruct(id=id, vector={}, payload=payload)],
            wait=True,
            ordering=rest.WriteOrdering.STRONG,
            update_mode=rest.UpdateMode.INSERT_ONLY
            if expected_revision is None
            else rest.UpdateMode.UPDATE_ONLY,
            update_filter=None
            if expected_revision is None
            else rest.Filter(
                must=[
                    rest.FieldCondition(
                        key=REVISION, match=rest.MatchValue(value=expected_revision)
                    )
                ]
            ),
        )
        if result.status != rest.UpdateStatus.COMPLETED:
            raise RuntimeError("Qdrant publication has not completed")
        observed = (await self._read_heads([id])).get(id)
        if observed is None:
            raise RuntimeError("Qdrant publication outcome could not be verified")
        return parse_head(observed, self._policy).revision == payload[REVISION]

    def _revision_filter(
        self, record: StoredDocumentRecord, revision: str
    ) -> rest.Filter:
        scope = build_document_lookup_filter(
            tenant_id=record.tenant_id,
            collection=record.collection,
            document_id=record.document_id,
            document_key=None,
            policy=self._policy,
        )
        return rest.Filter(
            must=[
                scope,
                rest.FieldCondition(key=KIND, match=rest.MatchValue(value=CHUNK)),
                rest.FieldCondition(
                    key=REVISION, match=rest.MatchValue(value=revision)
                ),
            ]
        )

    async def _count_revision(self, record: StoredDocumentRecord, revision: str) -> int:
        response = await self._client.count(
            collection_name=self._config.collection_name,
            count_filter=self._revision_filter(record, revision),
            exact=True,
        )
        return _count_result_value(response.count)

    async def _delete_revision(
        self, record: StoredDocumentRecord, revision: str
    ) -> None:
        await self._client.delete(
            collection_name=self._config.collection_name,
            points_selector=rest.FilterSelector(
                filter=self._revision_filter(record, revision)
            ),
            wait=True,
        )

    async def get_document_record(
        self,
        *,
        tenant_id: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None:
        tenant_id, collection = validate_document_lookup_inputs(
            tenant_id=tenant_id,
            collection=collection,
            document_id=document_id,
            document_key=document_key,
        )
        await self.ensure_collection()
        if document_id is not None:
            id = head_id(tenant_id, collection, document_id)
            heads = await self._read_heads([id])
            payload = heads.get(id)
        else:
            scope = build_document_lookup_filter(
                tenant_id=tenant_id,
                collection=collection,
                document_id=None,
                document_key=document_key,
                policy=self._policy,
            )
            rows, _ = await self._client.scroll(
                collection_name=self._config.collection_name,
                scroll_filter=rest.Filter(
                    must=[
                        scope,
                        rest.FieldCondition(
                            key=KIND, match=rest.MatchValue(value=HEAD)
                        ),
                    ]
                ),
                limit=2,
                with_payload=True,
                with_vectors=False,
                consistency=None if self._config.is_local else rest.ReadConsistencyType.ALL,
            )
            if len(rows) > 1:
                raise ValueError("document key identifies multiple published documents")
            payload = dict(rows[0].payload or {}) if rows else None
        if payload is not None:
            record = parse_head(payload, self._policy)
            if (
                record.tenant_id != tenant_id
                or record.collection != collection
                or (document_id is not None and record.document_id != document_id)
                or (document_key is not None and record.document_key != document_key)
            ):
                raise ValueError("Qdrant document head returned a different scope")
            if record.deleted:
                return None
            assert record.revision is not None
            return replace(
                record, chunk_count=await self._count_revision(record, record.revision)
            )
        return await get_qdrant_document_record(
            client=self._client,
            collection_name=self._config.collection_name,
            tenant_id=tenant_id,
            collection=collection,
            document_id=document_id,
            document_key=document_key,
            policy=self._policy,
            consistency=None if self._config.is_local else rest.ReadConsistencyType.ALL,
        )
