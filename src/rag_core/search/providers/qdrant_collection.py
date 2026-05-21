"""Collection setup and shape introspection for the Qdrant vector store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest

from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy

from .qdrant_collection_indexes import create_payload_indexes
from .qdrant_shared import (
    _DENSE_VECTOR_NAME,
    _PRIMARY_SPARSE_VECTOR_NAME,
    _SECONDARY_SPARSE_VECTOR_NAME,
)


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
) -> None:
    await client.create_collection(
        collection_name=config.collection_name,
        vectors_config=_build_vectors_config(config),
        sparse_vectors_config={
            _PRIMARY_SPARSE_VECTOR_NAME: rest.SparseVectorParams(
                modifier=rest.Modifier.IDF
            ),
            _SECONDARY_SPARSE_VECTOR_NAME: rest.SparseVectorParams(
                modifier=rest.Modifier.IDF
            ),
        },
        hnsw_config=rest.HnswConfigDiff(ef_construct=100),
        quantization_config=build_quantization_config(enabled=config.quantization_enabled),
        on_disk_payload=True,
    )

    if config.is_local:
        return
    await create_payload_indexes(
        client=client,
        collection_name=config.collection_name,
        policy=config.policy,
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
