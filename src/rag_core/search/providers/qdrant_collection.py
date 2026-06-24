"""Collection setup, payload indexes, readiness lifecycle, and health for Qdrant."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy
from rag_core.search.provider_protocols import QueryPlanCapabilities

from .qdrant_payloads import (
    _DENSE_VECTOR_NAME,
    _KNOWN_SPARSE_VECTOR_NAMES,
    _PRIMARY_SPARSE_VECTOR_NAME,
    _SECONDARY_SPARSE_VECTOR_NAME,
    WriteLatencyTracker,
)
from .vector_store_capabilities import (
    QDRANT_VECTOR_STORE_PROVIDER_SPEC,
    describe_query_plan_capabilities,
    qdrant_query_plan_capabilities_for_sparse_names,
)


# Collection-level metadata keys used to record embedding identity. These are
# durable in Qdrant's CollectionConfig.metadata mapping so a later process can
# refuse to bind to a collection that was produced by a different embedder.
EMBEDDING_MODEL_METADATA_KEY = "rag_core.embedding_model"
EMBEDDING_DIMENSIONS_METADATA_KEY = "rag_core.embedding_dimensions"


@dataclass(frozen=True)
class CollectionConfig:
    collection_name: str
    dimensions: int
    quantization_enabled: bool
    is_local: bool
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
        "sparse_vectors_config": {
            _PRIMARY_SPARSE_VECTOR_NAME: rest.SparseVectorParams(
                modifier=rest.Modifier.IDF
            ),
            _SECONDARY_SPARSE_VECTOR_NAME: rest.SparseVectorParams(
                modifier=rest.Modifier.IDF
            ),
        },
        "hnsw_config": rest.HnswConfigDiff(ef_construct=100),
        "quantization_config": build_quantization_config(
            enabled=config.quantization_enabled
        ),
        "on_disk_payload": True,
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
    """Refuse to bind to a Qdrant collection produced by a different embedder.

    Binding to a collection whose stored ``embedding_model`` / dimensions do
    not match the configured embedder would silently mix incompatible vector
    spaces. Catch that at the seam before any write or search reaches the store.
    """


@dataclass(frozen=True)
class QdrantAdapterConfig:
    collection_name: str
    dimensions: int
    quantization_enabled: bool
    is_local: bool
    max_concurrent: int
    max_batch_size: int
    policy: VectorStorePolicy = DEFAULT_POLICY
    # ``None`` means "do not assert identity". Preserves backward compatibility
    # for legacy collections that pre-date the sentinel.
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
        # If create_qdrant_collection raced with a parallel creator and fell
        # through to the compatibility path, ``state.ready`` is already set.
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
    state.available_sparse_vector_names = assert_collection_compatible(
        collection_name=config.collection_name,
        dimensions=config.dimensions,
        collection_info=info,
    )
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
    # Rebind to the now-existing collection; if no outer state was provided
    # the loser of the race still wants to fail closed on an identity
    # mismatch, so we run the compatibility check against a scratch state.
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
