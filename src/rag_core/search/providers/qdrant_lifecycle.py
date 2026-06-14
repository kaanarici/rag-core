"""Collection readiness lifecycle for the Qdrant vector store."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy

from .qdrant_collection import (
    CollectionConfig,
    assert_collection_compatible,
    assert_embedding_identity_matches,
    collection_exists,
    create_collection,
    extract_collection_metadata,
    pack_embedding_identity_metadata,
)
from .qdrant_health import _collection_fingerprint
from .qdrant_shared import _KNOWN_SPARSE_VECTOR_NAMES
from .vector_store_capabilities import QDRANT_VECTOR_STORE_PROVIDER_SPEC


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
