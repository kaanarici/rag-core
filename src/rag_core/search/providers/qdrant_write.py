"""Write-path helpers for the Qdrant vector store."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest

from rag_core.search.vector_models import VectorPoint

from .qdrant_shared import WriteLatencyTracker
from .qdrant_write_batches import build_qdrant_point_batches
from .qdrant_write_retry import upsert_with_fallback


async def upsert_qdrant_point_batches(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    dimensions: int,
    latency: WriteLatencyTracker,
    max_batch_size: int,
    write_sem: asyncio.Semaphore,
    points: Sequence[VectorPoint],
    available_sparse_vector_names: frozenset[str] | set[str],
) -> None:
    async def _upsert_single_batch(batch: list[rest.PointStruct]) -> None:
        async with write_sem:
            await upsert_with_fallback(
                client=client,
                collection_name=collection_name,
                dimensions=dimensions,
                latency=latency,
                max_batch_size=max_batch_size,
                points=batch,
                split_depth=0,
            )

    batches = build_qdrant_point_batches(
        points=points,
        batch_size=max_batch_size,
        available_sparse_vector_names=available_sparse_vector_names,
    )
    if not batches:
        return

    async with asyncio.TaskGroup() as task_group:
        for batch in batches:
            task_group.create_task(_upsert_single_batch(batch))
