"""Collection setup and shape introspection for the Qdrant vector store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest

from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy

from .qdrant_collection_indexes import create_payload_indexes
from .qdrant_shared import (
    _DENSE_VECTOR_NAME,
    _PRIMARY_SPARSE_VECTOR_NAME,
    _SECONDARY_SPARSE_VECTOR_NAME,
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
