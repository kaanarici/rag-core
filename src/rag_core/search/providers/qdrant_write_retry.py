"""Qdrant upsert retry and logging helpers."""

from __future__ import annotations

import asyncio
import logging
import time

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as rest
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from .qdrant_shared import (
    _MAX_SPLIT_DEPTH,
    _SLOW_WRITE_THRESHOLD_SECONDS,
    _SPLIT_PAUSE_SECONDS,
    WriteLatencyTracker,
)
from .qdrant_write_logging import (
    log_qdrant_upsert_error,
    log_split_qdrant_batch,
    log_successful_qdrant_upsert,
    log_unsplittable_qdrant_batch,
)

logger = logging.getLogger(__name__)


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
