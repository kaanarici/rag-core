"""Sanitized Qdrant write logging helpers."""

from __future__ import annotations

import logging

from qdrant_client import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from .qdrant_shared import _MAX_SPLIT_DEPTH, WriteLatencyTracker
from .vector_store_capabilities import QDRANT_VECTOR_STORE_PROVIDER_SPEC


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
