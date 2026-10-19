"""Bounded, idempotent chunk writes and scoped deletes."""

from __future__ import annotations
import asyncio
import logging
import math
import random
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, cast
import httpx
from qdrant_client import AsyncQdrantClient, models as rest
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from rag_core.search.policy import VectorStorePolicy
from rag_core.search.request_models import DeleteFilter
from rag_core.search.vector_models import VectorPoint
from .qdrant_filters import build_delete_filter
from rag_core.search.stored_payload import validate_json_payload
from .qdrant_payloads import (
    _MAX_SPLIT_DEPTH,
    _SLOW_WRITE_THRESHOLD_SECONDS,
    WriteLatencyTracker,
    _build_point,
    _qdrant_point_id,
    validate_point_channels,
)

logger = logging.getLogger(__name__)
_MAX_TRANSIENT_RETRIES = 2
_MAX_RETRY_DELAY_SECONDS = 8.0


def _retry_delay(exc: Exception, retry: int) -> float | None:
    hint: str | int | float | None = None
    if isinstance(exc, ResourceExhaustedResponse):
        hint = exc.retry_after_s
    elif isinstance(exc, UnexpectedResponse):
        if exc.status_code not in {408, 429, 500, 502, 503, 504}:
            return None
        hint = exc.headers.get("Retry-After")
    else:
        source = exc.source if isinstance(exc, ResponseHandlingException) else exc
        if not isinstance(source, (httpx.TransportError, TimeoutError, ConnectionError)):
            return None
    minimum = 0.0
    if hint is not None:
        try:
            minimum = float(hint)
        except (ValueError, TypeError):
            try:
                minimum = (
                    parsedate_to_datetime(str(hint)) - datetime.now(timezone.utc)
                ).total_seconds()
            except (ValueError, TypeError, OverflowError):
                minimum = 0.0
        if not math.isfinite(minimum):
            minimum = 0.0
    if minimum > _MAX_RETRY_DELAY_SECONDS:
        return None
    return max(minimum, random.uniform(0, min(_MAX_RETRY_DELAY_SECONDS, 0.5 * 2**retry)))


async def upsert_with_fallback(
    client: AsyncQdrantClient,
    collection_name: str,
    dimensions: int,
    latency: WriteLatencyTracker,
    points: list[rest.PointStruct],
    split_depth: int,
) -> None:
    # One retry budget spans the split tree; overload never multiplies requests.
    retries = 0

    async def write(batch: list[rest.PointStruct], depth: int) -> None:
        nonlocal retries
        started = time.monotonic()
        while True:
            try:
                await client.upsert(collection_name=collection_name, points=batch, wait=True)
                break
            except Exception as exc:
                logger.warning(
                    "Qdrant upsert failed: provider=qdrant error_type=%s http_status=%s batch_size=%d dense_dimensions=%d split_depth=%d",
                    type(exc).__name__, getattr(exc, "status_code", None),
                    len(batch), dimensions, depth,
                )
                if isinstance(exc, UnexpectedResponse) and exc.status_code == 413:
                    if len(batch) <= 1 or depth >= _MAX_SPLIT_DEPTH:
                        raise
                    mid = len(batch) // 2
                    await write(batch[:mid], depth + 1)
                    await write(batch[mid:], depth + 1)
                    return
                delay = _retry_delay(exc, retries)
                if delay is None or retries >= _MAX_TRANSIENT_RETRIES:
                    raise
                retries += 1
                await asyncio.sleep(delay)
        duration = time.monotonic() - started
        latency.record(duration)
        logger.log(
            logging.WARNING
            if duration > _SLOW_WRITE_THRESHOLD_SECONDS
            else logging.DEBUG,
            "%s: provider=qdrant point_count=%d duration=%.2fs dense_dimensions=%d split_depth=%d latency_p50=%.3fs latency_p95=%.3fs",
            "Slow Qdrant write"
            if duration > _SLOW_WRITE_THRESHOLD_SECONDS
            else "Qdrant write completed",
            len(batch),
            duration,
            dimensions,
            depth,
            latency.p50 or 0.0,
            latency.p95 or 0.0,
        )

    await write(points, split_depth)


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
    max_concurrent: int = 16,
) -> None:
    if max_batch_size <= 0 or max_concurrent <= 0:
        raise ValueError("write batch size and concurrency must be positive")
    for point in points:
        _qdrant_point_id(point.id)
        validate_point_channels(point, available_sparse_vector_names)
        validate_json_payload(point.payload)
    starts = iter(range(0, len(points), max_batch_size))

    async def worker() -> None:
        for start in starts:
            async with write_sem:
                batch = [
                    _build_point(
                        point,
                        available_sparse_vector_names=available_sparse_vector_names,
                    )
                    for point in points[start : start + max_batch_size]
                ]
                await upsert_with_fallback(
                    client=client,
                    collection_name=collection_name,
                    dimensions=dimensions,
                    latency=latency,
                    points=batch,
                    split_depth=0,
                )
                del batch

    async with asyncio.TaskGroup() as group:
        for _ in range(
            min(max_concurrent, (len(points) + max_batch_size - 1) // max_batch_size)
        ):
            group.create_task(worker())


async def delete_qdrant_filter(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    filter_values: DeleteFilter,
    tenant_id: str,
    policy: VectorStorePolicy,
) -> None:
    await client.delete(
        collection_name=collection_name,
        points_selector=rest.FilterSelector(
            filter=build_delete_filter(
                filter_values=filter_values, tenant_id=tenant_id, policy=policy
            )
        ),
    )


async def delete_qdrant_point_ids(
    *, client: AsyncQdrantClient, collection_name: str, point_ids: Sequence[str]
) -> None:
    await client.delete(
        collection_name=collection_name,
        points_selector=rest.PointIdsList(
            points=cast(Any, [_qdrant_point_id(point_id) for point_id in point_ids])
        ),
    )
