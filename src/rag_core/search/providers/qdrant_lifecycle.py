"""Collection readiness lifecycle for the Qdrant vector store."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient

from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy

from .qdrant_collection import (
    CollectionConfig,
    assert_collection_compatible,
    collection_exists,
    create_collection,
)
from .qdrant_health import _collection_fingerprint
from .qdrant_shared import _KNOWN_SPARSE_VECTOR_NAMES
from .vector_store_capabilities import QDRANT_VECTOR_STORE_PROVIDER_SPEC


@dataclass(frozen=True)
class QdrantAdapterConfig:
    collection_name: str
    dimensions: int
    quantization_enabled: bool
    is_local: bool
    max_concurrent: int
    max_batch_size: int
    policy: VectorStorePolicy = DEFAULT_POLICY


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
            state.available_sparse_vector_names = await load_qdrant_sparse_channels(
                client=client,
                config=config,
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
            return

        await create_qdrant_collection(
            client=client,
            config=config,
            logger=logger,
        )
        state.ready = True


async def create_qdrant_collection(
    *,
    client: AsyncQdrantClient,
    config: QdrantAdapterConfig,
    logger: logging.Logger,
) -> None:
    await create_collection(
        client=client,
        config=CollectionConfig(
            collection_name=config.collection_name,
            dimensions=config.dimensions,
            quantization_enabled=config.quantization_enabled,
            is_local=config.is_local,
            policy=config.policy,
        ),
    )
    logger.info(
        "Created Qdrant collection: provider=%s dense_dimensions=%d "
        "quantization=%s hnsw_ef=%d collection_fingerprint=%s",
        QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        config.dimensions,
        "INT8" if config.quantization_enabled else "none",
        100,
        _collection_fingerprint(config.collection_name),
    )


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
