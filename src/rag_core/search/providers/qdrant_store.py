"""Qdrant adapter: collection lifecycle and the VectorStore entry.

Shared payload conversion and write-tuning helpers live in ``qdrant_payloads``
so query/write modules can import them without a cycle through this entry.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.provider_protocols import QueryPlanCapabilities, StoreCapabilities
from rag_core.search.query_plan import QueryPlan
from rag_core.search.query_plan_presets import (
    QUERY_PLAN_PRESET_DENSE_ONLY,
    query_plan_preset,
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
    _KNOWN_SPARSE_VECTOR_NAMES,
    _PRIMARY_SPARSE_VECTOR_NAME,
    _SECONDARY_SPARSE_VECTOR_NAME,
    WriteLatencyTracker,
    compute_write_params,
    get_qdrant_chunks_by_index,
    get_qdrant_document_record,
)
from .qdrant_query import (
    search_qdrant_points,
    validate_qdrant_delete_filter,
    validate_qdrant_query_plan_preflight,
    validate_qdrant_query_plan_sparse_channels,
    validate_qdrant_search_request,
)
from .qdrant_write import (
    delete_qdrant_filter,
    delete_qdrant_point_ids,
    upsert_qdrant_point_batches,
)
from .registry import VECTOR_STORES
from .vector_dimensions import (
    validate_point_dense_dimensions,
)
from .vector_store_capabilities import (
    QDRANT_VECTOR_STORE_CAPABILITY_SPEC,
    QDRANT_VECTOR_STORE_PROVIDER_SPEC,
    describe_query_plan_capabilities,
    qdrant_query_plan_capabilities_for_sparse_names,
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
            _SECONDARY_SPARSE_VECTOR_NAME: rest.SparseVectorParams(
                modifier=rest.Modifier.IDF
            ),
        }
    if collection_metadata:
        create_kwargs["metadata"] = collection_metadata
    await client.create_collection(**create_kwargs)

    if config.is_local:
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
    if stored_model is None and stored_dim is None:
        # Legacy collection: no identity sentinel was written. Dimension
        # mismatch is already caught by assert_collection_compatible upstream;
        # silently allow binding so existing deployments keep working.
        return
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
) -> None:
    tenant_field = policy.tenant_payload_field
    for field_name, schema_type in collection_index_fields(policy):
        if tenant_field is not None and field_name == tenant_field:
            await client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=rest.KeywordIndexParams(
                    type=rest.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            )
            continue
        await client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )


def collection_index_fields(
    policy: VectorStorePolicy,
) -> tuple[tuple[str, rest.PayloadSchemaType], ...]:
    return (
        (policy.namespace_field, rest.PayloadSchemaType.KEYWORD),
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
            QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
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
        "adapter": QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
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
    health["query_plan"] = describe_query_plan_capabilities(
        _collection_query_plan_capabilities(collection_info)
    )

    health["write_latency_p50"] = latency.p50
    health["write_latency_p95"] = latency.p95
    health["write_latency_samples"] = latency.sample_count
    return health


def _collection_query_plan_capabilities(collection_info: object) -> QueryPlanCapabilities:
    sparse_names = extract_sparse_vector_names(collection_info)
    return qdrant_query_plan_capabilities_for_sparse_names(sparse_names)


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


@dataclass(frozen=True)
class QdrantAdapterConfig:
    collection_name: str
    dimensions: int
    quantization_enabled: bool
    is_local: bool
    max_concurrent: int
    max_batch_size: int
    sparse_enabled: bool = False
    policy: VectorStorePolicy = DEFAULT_POLICY
    embedding_model: str | None = None


@dataclass
class QdrantCollectionState:
    available_sparse_vector_names: frozenset[str] = _KNOWN_SPARSE_VECTOR_NAMES
    ready: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def ensure_qdrant_collection_ready(
    *,
    client: AsyncQdrantClient,
    config: QdrantAdapterConfig,
    state: QdrantCollectionState,
    logger: logging.Logger,
) -> None:
    if state.ready:
        return

    async with state.lock:
        if state.ready:
            return  # type: ignore[unreachable]

        collections = await client.get_collections()
        existing = {c.name for c in collections.collections}
        if collection_exists(
            existing_names=existing,
            collection_name=config.collection_name,
        ):
            await _bind_existing_collection(
                client=client,
                config=config,
                state=state,
                logger=logger,
            )
            return

        await create_qdrant_collection(
            client=client,
            config=config,
            logger=logger,
            on_race_state=state,
        )
        if not state.ready:
            state.ready = True


async def _bind_existing_collection(
    *,
    client: AsyncQdrantClient,
    config: QdrantAdapterConfig,
    state: QdrantCollectionState,
    logger: logging.Logger,
) -> None:
    info = await client.get_collection(collection_name=config.collection_name)
    available_sparse_vector_names = assert_collection_compatible(
        collection_name=config.collection_name,
        dimensions=config.dimensions,
        collection_info=info,
    )
    if config.sparse_enabled:
        missing_sparse_names = (
            _KNOWN_SPARSE_VECTOR_NAMES - available_sparse_vector_names
        )
        if missing_sparse_names:
            missing = ", ".join(sorted(missing_sparse_names))
            available = ", ".join(sorted(available_sparse_vector_names)) or "none"
            raise ValueError(
                f"Qdrant collection {config.collection_name!r} is missing required "
                f"sparse vector channels: {missing}; available sparse channels: "
                f"{available}. Use a different collection name or delete and rebuild "
                "this collection before enabling sparse retrieval."
            )
        state.available_sparse_vector_names = available_sparse_vector_names
    else:
        state.available_sparse_vector_names = frozenset()
    stored_metadata = extract_collection_metadata(info)
    if config.embedding_model is not None:
        assert_embedding_identity_matches(
            collection_name=config.collection_name,
            expected_model=config.embedding_model,
            expected_dimensions=config.dimensions,
            collection_metadata=stored_metadata,
            mismatch_cls=EmbeddingIdentityMismatch,
        )
    logger.info(
        "Qdrant collection already exists: provider=%s "
        "dense_dimensions=%d sparse_channels=%d collection_fingerprint=%s",
        QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        config.dimensions,
        len(state.available_sparse_vector_names),
        _collection_fingerprint(config.collection_name),
    )
    state.ready = True


async def create_qdrant_collection(
    *,
    client: AsyncQdrantClient,
    config: QdrantAdapterConfig,
    logger: logging.Logger,
    on_race_state: QdrantCollectionState | None = None,
) -> None:
    collection_metadata = pack_embedding_identity_metadata(
        embedding_model=config.embedding_model,
        dimensions=config.dimensions,
    )
    try:
        await create_collection(
            client=client,
            config=CollectionConfig(
                collection_name=config.collection_name,
                dimensions=config.dimensions,
                quantization_enabled=config.quantization_enabled,
                is_local=config.is_local,
                sparse_enabled=config.sparse_enabled,
                policy=config.policy,
            ),
            collection_metadata=collection_metadata,
        )
    except UnexpectedResponse as exc:
        if not _is_already_exists_response(exc):
            raise
        await _handle_create_race(
            client=client,
            config=config,
            state=on_race_state,
            logger=logger,
        )
        return
    except ValueError as exc:
        # Some Qdrant client versions wrap "already exists" as ValueError.
        if not _is_already_exists_message(str(exc)):
            raise
        await _handle_create_race(
            client=client,
            config=config,
            state=on_race_state,
            logger=logger,
        )
        return
    logger.info(
        "Created Qdrant collection: provider=%s dense_dimensions=%d "
        "quantization=%s hnsw_ef=%d collection_fingerprint=%s",
        QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        config.dimensions,
        "INT8" if config.quantization_enabled else "none",
        100,
        _collection_fingerprint(config.collection_name),
    )


async def _handle_create_race(
    *,
    client: AsyncQdrantClient,
    config: QdrantAdapterConfig,
    state: QdrantCollectionState | None,
    logger: logging.Logger,
) -> None:
    logger.info(
        "Qdrant collection create raced (already exists), falling through: "
        "provider=%s collection_fingerprint=%s",
        QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        _collection_fingerprint(config.collection_name),
    )
    target = state if state is not None else QdrantCollectionState()
    await _bind_existing_collection(
        client=client,
        config=config,
        state=target,
        logger=logger,
    )


def _is_already_exists_response(exc: UnexpectedResponse) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 409:
        return True
    return _is_already_exists_message(str(exc))


def _is_already_exists_message(message: str) -> bool:
    lowered = message.lower()
    return "already exists" in lowered or "conflict" in lowered


async def load_qdrant_sparse_channels(
    *,
    client: AsyncQdrantClient,
    config: QdrantAdapterConfig,
) -> frozenset[str]:
    info = await client.get_collection(collection_name=config.collection_name)
    return assert_collection_compatible(
        collection_name=config.collection_name,
        dimensions=config.dimensions,
        collection_info=info,
    )


@dataclass(frozen=True)
class QdrantClientState:
    client: AsyncQdrantClient
    is_local: bool


def create_qdrant_client(
    *,
    url: str | None,
    api_key: str | None,
    location: str | None,
    timeout: int = 120,
) -> QdrantClientState:
    api_key = _normalize_api_key(api_key)
    if bool(url) == bool(location):
        raise ValueError(
            "QdrantVectorStore requires exactly one of url or location; "
            "use QdrantConfig(location=':memory:'), pass url=..., "
            "or inject vector_store=... into Engine."
        )

    if location is not None:
        if location != ":memory:":
            return QdrantClientState(
                client=AsyncQdrantClient(
                    path=location,
                    timeout=timeout,
                    check_compatibility=False,
                ),
                is_local=True,
            )
        return QdrantClientState(
            client=AsyncQdrantClient(
                location=location,
                timeout=timeout,
                check_compatibility=False,
            ),
            is_local=True,
        )

    if api_key is not None:
        return QdrantClientState(
            client=AsyncQdrantClient(url=url, api_key=api_key, timeout=timeout),
            is_local=False,
        )

    return QdrantClientState(
        client=AsyncQdrantClient(url=url, timeout=timeout),
        is_local=False,
    )


def _normalize_api_key(api_key: str | None) -> str | None:
    if api_key is None:
        return None
    stripped = api_key.strip()
    return stripped or None


@dataclass(frozen=True)
class QdrantAdapterRuntime:
    client: AsyncQdrantClient
    is_local: bool
    config: QdrantAdapterConfig
    write_sem: asyncio.Semaphore
    latency: WriteLatencyTracker


def create_qdrant_adapter_runtime(
    *,
    url: str | None,
    api_key: str | None,
    location: str | None,
    collection_name: str,
    dense_dimensions: int,
    quantization_enabled: bool,
    sparse_enabled: bool,
    policy: VectorStorePolicy,
    logger: logging.Logger,
    embedding_model: str | None = None,
) -> QdrantAdapterRuntime:
    client_state = create_qdrant_client(
        url=url,
        api_key=api_key,
        location=location,
    )
    max_concurrent, max_batch_size = compute_write_params(dense_dimensions)
    config = QdrantAdapterConfig(
        collection_name=collection_name,
        dimensions=dense_dimensions,
        quantization_enabled=quantization_enabled,
        is_local=client_state.is_local,
        max_concurrent=max_concurrent,
        max_batch_size=max_batch_size,
        sparse_enabled=sparse_enabled,
        policy=policy,
        embedding_model=embedding_model,
    )
    _log_qdrant_adapter_initialized(logger, config=config)
    return QdrantAdapterRuntime(
        client=client_state.client,
        is_local=client_state.is_local,
        config=config,
        write_sem=asyncio.Semaphore(max_concurrent),
        latency=WriteLatencyTracker(),
    )


def _log_qdrant_adapter_initialized(
    logger: logging.Logger,
    *,
    config: QdrantAdapterConfig,
) -> None:
    logger.info(
        "QdrantVectorStore initialized: provider=%s dense_dimensions=%d "
        "max_concurrent=%d max_batch_size=%d quantization=%s local=%s "
        "collection_fingerprint=%s",
        QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        config.dimensions,
        config.max_concurrent,
        config.max_batch_size,
        config.quantization_enabled,
        config.is_local,
        _collection_fingerprint(config.collection_name),
    )


class QdrantVectorStore:
    """Qdrant adapter implementing the full ``VectorStore`` capability surface."""

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
        runtime = create_qdrant_adapter_runtime(
            url=url,
            api_key=api_key,
            location=location,
            collection_name=collection_name,
            dense_dimensions=dense_dimensions,
            quantization_enabled=quantization_enabled,
            sparse_enabled=sparse_enabled,
            policy=policy,
            logger=logger,
            embedding_model=embedding_model,
        )
        self._client = runtime.client
        self._config = runtime.config
        self._policy = policy
        self._collection_state = QdrantCollectionState(
            available_sparse_vector_names=(
                _KNOWN_SPARSE_VECTOR_NAMES if sparse_enabled else frozenset()
            )
        )
        self._write_sem = runtime.write_sem
        self._latency = runtime.latency

    async def __aenter__(self) -> "QdrantVectorStore":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def capabilities(self) -> StoreCapabilities:
        if not self._config.sparse_enabled:
            query_plan_capabilities = qdrant_query_plan_capabilities_for_sparse_names(
                frozenset()
            )
        elif self._collection_state.ready:
            query_plan_capabilities = qdrant_query_plan_capabilities_for_sparse_names(
                self._collection_state.available_sparse_vector_names,
            )
        else:
            query_plan_capabilities = QDRANT_VECTOR_STORE_CAPABILITY_SPEC.query_plan
        return QDRANT_VECTOR_STORE_CAPABILITY_SPEC.to_store_capabilities(
            dense_vector_dimensions=self._config.dimensions,
            query_plan=query_plan_capabilities,
        )

    async def close(self) -> None:
        await self._client.close()

    async def ensure_collection(self) -> None:
        await ensure_qdrant_collection_ready(
            client=self._client,
            config=self._config,
            state=self._collection_state,
            logger=logger,
        )

    async def check_health(self) -> dict[str, object]:
        return await check_qdrant_health(
            client=self._client,
            collection_name=self._config.collection_name,
            dimensions=self._config.dimensions,
            latency=self._latency,
            logger=logger,
        )

    def validate_query_plan(self, plan: QueryPlan) -> None:
        validate_qdrant_query_plan_preflight(plan)

    def default_query_plan(self, *, result_limit: int) -> QueryPlan:
        return query_plan_preset(
            QUERY_PLAN_PRESET_DENSE_ONLY,
            limit=result_limit,
        )

    async def prepare_query_plan(self, plan: QueryPlan) -> None:
        validate_qdrant_query_plan_preflight(plan)
        await self.ensure_collection()
        validate_qdrant_query_plan_sparse_channels(
            plan,
            available_sparse_names=self._collection_state.available_sparse_vector_names,
        )

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        validate_point_dense_dimensions(
            points,
            dense_dimensions=self._config.dimensions,
            provider_name=QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        )
        await self.ensure_collection()

        await upsert_qdrant_point_batches(
            client=self._client,
            collection_name=self._config.collection_name,
            dimensions=self._config.dimensions,
            latency=self._latency,
            max_batch_size=self._config.max_batch_size,
            write_sem=self._write_sem,
            points=points,
            available_sparse_vector_names=(
                self._collection_state.available_sparse_vector_names
            ),
        )

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        namespace = validate_qdrant_search_request(
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
            namespace=namespace,
            policy=self._policy,
            available_sparse_vector_names=(
                self._collection_state.available_sparse_vector_names
            ),
        )

    async def delete(self, filter: DeleteFilter) -> None:
        namespace = validate_qdrant_delete_filter(filter)
        await self.ensure_collection()

        await delete_qdrant_filter(
            client=self._client,
            collection_name=self._config.collection_name,
            filter_values=filter,
            namespace=namespace,
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

    async def get_document_record(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None:
        await self.ensure_collection()
        return await get_qdrant_document_record(
            client=self._client,
            collection_name=self._config.collection_name,
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            document_key=document_key,
            policy=self._policy,
        )

    async def get_chunks_by_index(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
        chunk_indices: Sequence[int],
    ) -> list[SearchResult]:
        await self.ensure_collection()
        return await get_qdrant_chunks_by_index(
            client=self._client,
            collection_name=self._config.collection_name,
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            chunk_indices=chunk_indices,
            policy=self._policy,
        )


VECTOR_STORES.register(
    QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
    lambda **kw: QdrantVectorStore(**kw),
)
