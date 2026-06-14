"""Qdrant adapter construction helpers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient

from rag_core.search.policy import VectorStorePolicy

from .qdrant_client import create_qdrant_client
from .qdrant_health import _collection_fingerprint
from .qdrant_lifecycle import QdrantAdapterConfig
from .qdrant_shared import WriteLatencyTracker, compute_write_params
from .vector_store_capabilities import QDRANT_VECTOR_STORE_PROVIDER_SPEC


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


__all__ = ["QdrantAdapterRuntime", "create_qdrant_adapter_runtime"]
