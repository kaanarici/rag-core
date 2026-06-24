"""Write-path helpers for the Qdrant vector store.

Batching, hardened upsert with adaptive split/retry, sanitized write logging,
and point/filter deletes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from rag_core.search.policy import VectorStorePolicy
from rag_core.search.request_models import DeleteFilter
from rag_core.search.vector_models import VectorPoint

from .qdrant_filters import build_delete_filter
from .qdrant_payloads import (
    _MAX_SPLIT_DEPTH,
    _SLOW_WRITE_THRESHOLD_SECONDS,
    _SPLIT_PAUSE_SECONDS,
    WriteLatencyTracker,
    _build_point,
    _qdrant_point_id,
)
from .vector_store_capabilities import QDRANT_VECTOR_STORE_PROVIDER_SPEC

logger = logging.getLogger(__name__)


def build_qdrant_point_batches(
    *,
    points: Sequence[VectorPoint],
    batch_size: int,
    available_sparse_vector_names: frozenset[str] | set[str],
) -> list[list[rest.PointStruct]]:
    qdrant_points = [
        _build_point(
            point,
            available_sparse_vector_names=available_sparse_vector_names,
        )
        for point in points
    ]
    return split_into_batches(qdrant_points, batch_size)


def split_into_batches(
    points: list[rest.PointStruct],
    batch_size: int,
) -> list[list[rest.PointStruct]]:
    if batch_size <= 0:
        return [points] if points else []
    return [points[i : i + batch_size] for i in range(0, len(points), batch_size)]


def log_qdrant_upsert_error(
    logger: logging.Logger,
    *,
    exc: Exception,
    dimensions: int,
    points: list[rest.PointStruct],
    split_depth: int,
) -> None:
    http_status = exc.status_code if isinstance(exc, UnexpectedResponse) else None
    logger.error(
        "Qdrant upsert failed: provider=%s error_type=%s http_status=%s "
        "batch_size=%d dense_dimensions=%d split_depth=%d max_split_depth=%d",
        QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        type(exc).__name__,
        http_status,
        len(points),
        dimensions,
        split_depth,
        _MAX_SPLIT_DEPTH,
    )


def log_successful_qdrant_upsert(
    logger: logging.Logger,
    *,
    point_count: int,
    duration: float,
    dimensions: int,
    split_depth: int,
    latency: WriteLatencyTracker,
    slow_write_threshold_seconds: float,
) -> None:
    if duration > slow_write_threshold_seconds:
        logger.warning(
            "Slow Qdrant write: provider=%s point_count=%d duration=%.2fs "
            "dense_dimensions=%d split_depth=%d latency_p50=%.3fs latency_p95=%.3fs",
            QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
            point_count,
            duration,
            dimensions,
            split_depth,
            latency.p50 or 0.0,
            latency.p95 or 0.0,
        )
        return
    logger.debug(
        "Qdrant write completed: provider=%s point_count=%d duration=%.2fs "
        "dense_dimensions=%d split_depth=%d latency_p50=%.3fs latency_p95=%.3fs",
        QDRANT_VECTOR_STORE_PROVIDER_SPEC.name,
        point_count,
        duration,
        dimensions,
        split_depth,
        latency.p50 or 0.0,
        latency.p95 or 0.0,
    )


def log_unsplittable_qdrant_batch(
    logger: logging.Logger,
    *,
    point_count: int,
    split_depth: int,
    min_batch: int,
) -> None:
    logger.error(
        "Cannot split further: %d points, depth=%d/%d, min_batch=%d. "
        "Raising original error.",
        point_count,
        split_depth,
        _MAX_SPLIT_DEPTH,
        min_batch,
    )


def log_split_qdrant_batch(
    logger: logging.Logger,
    *,
    point_count: int,
    left_count: int,
    right_count: int,
    split_depth: int,
) -> None:
    logger.warning(
        "Splitting failed batch: %d -> %d + %d (depth=%d/%d)",
        point_count,
        left_count,
        right_count,
        split_depth + 1,
        _MAX_SPLIT_DEPTH,
    )


def log_upsert_error(
    exc: Exception,
    collection_name: str,
    dimensions: int,
    points: list[rest.PointStruct],
    split_depth: int,
) -> None:
    del collection_name
    log_qdrant_upsert_error(
        logger,
        exc=exc,
        dimensions=dimensions,
        points=points,
        split_depth=split_depth,
    )


async def upsert_with_fallback(
    client: AsyncQdrantClient,
    collection_name: str,
    dimensions: int,
    latency: WriteLatencyTracker,
    max_batch_size: int,
    points: list[rest.PointStruct],
    split_depth: int,
) -> None:
    n = len(points)
    min_batch = max(1, max_batch_size // 16)

    try:
        start = time.monotonic()
        await client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        duration = time.monotonic() - start
        latency.record(duration)
        log_successful_qdrant_upsert(
            logger,
            point_count=n,
            duration=duration,
            dimensions=dimensions,
            split_depth=split_depth,
            latency=latency,
            slow_write_threshold_seconds=_SLOW_WRITE_THRESHOLD_SECONDS,
        )

    except (UnexpectedResponse, ResponseHandlingException) as exc:
        log_upsert_error(exc, collection_name, dimensions, points, split_depth)
        await _maybe_split_and_retry(
            client=client,
            collection_name=collection_name,
            dimensions=dimensions,
            latency=latency,
            max_batch_size=max_batch_size,
            points=points,
            split_depth=split_depth,
            min_batch=min_batch,
            original_error=exc,
        )

    except Exception as exc:
        if "timeout" not in type(exc).__name__.lower():
            raise
        log_upsert_error(exc, collection_name, dimensions, points, split_depth)
        await _maybe_split_and_retry(
            client=client,
            collection_name=collection_name,
            dimensions=dimensions,
            latency=latency,
            max_batch_size=max_batch_size,
            points=points,
            split_depth=split_depth,
            min_batch=min_batch,
            original_error=exc,
        )


async def _maybe_split_and_retry(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    dimensions: int,
    latency: WriteLatencyTracker,
    max_batch_size: int,
    points: list[rest.PointStruct],
    split_depth: int,
    min_batch: int,
    original_error: Exception,
) -> None:
    n = len(points)
    if n <= min_batch or split_depth >= _MAX_SPLIT_DEPTH:
        log_unsplittable_qdrant_batch(
            logger,
            point_count=n,
            split_depth=split_depth,
            min_batch=min_batch,
        )
        raise original_error

    mid = n // 2
    left, right = points[:mid], points[mid:]
    log_split_qdrant_batch(
        logger,
        point_count=n,
        left_count=len(left),
        right_count=len(right),
        split_depth=split_depth,
    )

    await asyncio.sleep(_SPLIT_PAUSE_SECONDS)
    await upsert_with_fallback(
        client=client,
        collection_name=collection_name,
        dimensions=dimensions,
        latency=latency,
        max_batch_size=max_batch_size,
        points=left,
        split_depth=split_depth + 1,
    )
    await asyncio.sleep(_SPLIT_PAUSE_SECONDS)
    await upsert_with_fallback(
        client=client,
        collection_name=collection_name,
        dimensions=dimensions,
        latency=latency,
        max_batch_size=max_batch_size,
        points=right,
        split_depth=split_depth + 1,
    )


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


async def delete_qdrant_filter(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    filter_values: DeleteFilter,
    namespace: str,
    policy: VectorStorePolicy,
) -> None:
    await client.delete(
        collection_name=collection_name,
        points_selector=rest.FilterSelector(
            filter=build_delete_filter(
                filter_values=filter_values,
                namespace=namespace,
                policy=policy,
            ),
        ),
    )


async def delete_qdrant_point_ids(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    point_ids: Sequence[str],
) -> None:
    qdrant_point_ids = [_qdrant_point_id(point_id) for point_id in point_ids]
    await client.delete(
        collection_name=collection_name,
        points_selector=rest.PointIdsList(points=cast(Any, qdrant_point_ids)),
    )
