"""Prefetch task lifecycle for search sidecar postprocessing."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import cast

from rag_core.events.emit import now_ms
from rag_core.search.pipeline.stages.sidecar_application import build_sidecar_query
from rag_core.search.pipeline.types import PipelineQuery
from rag_core.search.types import SearchResult, SearchSidecar

logger = logging.getLogger("rag_core.search.pipeline.stages.sidecar_postprocess")

_SIDECAR_FUTURE_KEY = "__sidecar_future"


@dataclass(frozen=True)
class _SidecarPrefetch:
    task: asyncio.Task[list[SearchResult]]
    started_ms: float


@dataclass(frozen=True)
class SidecarResolution:
    results: list[SearchResult]
    duration_ms: float
    error_type: str = ""


async def start_prefetched_sidecar(
    query: PipelineQuery,
    sidecar: SearchSidecar,
) -> None:
    await cancel_prefetched_sidecar(query)
    sidecar_query = build_sidecar_query(query)
    # asyncio.create_task starts the coroutine so its I/O can overlap with Retrieve.
    query.extra[_SIDECAR_FUTURE_KEY] = _SidecarPrefetch(
        task=asyncio.create_task(sidecar.search(sidecar_query)),
        started_ms=now_ms(),
    )


async def resolve_sidecar_results(
    query: PipelineQuery,
    sidecar: SearchSidecar | None,
) -> SidecarResolution:
    prefetched = query.extra.pop(_SIDECAR_FUTURE_KEY, None)
    if prefetched is not None:
        prefetch = cast(_SidecarPrefetch, prefetched)
        try:
            sidecar_results = await prefetch.task
        except asyncio.CancelledError:
            if _is_current_task_cancelling():
                raise
            error_type = "CancelledError"
            _log_sidecar_failure(error_type)
            return SidecarResolution(
                results=[],
                duration_ms=now_ms() - prefetch.started_ms,
                error_type=error_type,
            )
        except Exception as exc:
            error_type = type(exc).__name__
            _log_sidecar_failure(error_type)
            return SidecarResolution(
                results=[],
                duration_ms=now_ms() - prefetch.started_ms,
                error_type=error_type,
            )
        return SidecarResolution(
            results=sidecar_results,
            duration_ms=now_ms() - prefetch.started_ms,
        )

    if sidecar is None:
        return SidecarResolution(results=[], duration_ms=0.0)
    sidecar_query = build_sidecar_query(query)
    started_ms = now_ms()
    try:
        sidecar_results = await sidecar.search(sidecar_query)
    except asyncio.CancelledError:
        if _is_current_task_cancelling():
            raise
        error_type = "CancelledError"
        _log_sidecar_failure(error_type)
        return SidecarResolution(
            results=[],
            duration_ms=now_ms() - started_ms,
            error_type=error_type,
        )
    except Exception as exc:
        error_type = type(exc).__name__
        _log_sidecar_failure(error_type)
        return SidecarResolution(
            results=[],
            duration_ms=now_ms() - started_ms,
            error_type=error_type,
        )
    return SidecarResolution(
        results=sidecar_results,
        duration_ms=now_ms() - started_ms,
    )


async def cancel_prefetched_sidecar(query: PipelineQuery) -> None:
    prefetched = query.extra.pop(_SIDECAR_FUTURE_KEY, None)
    if prefetched is None:
        return
    prefetch = cast(_SidecarPrefetch, prefetched)
    if not prefetch.task.done():
        prefetch.task.cancel()
    try:
        await prefetch.task
    except asyncio.CancelledError:
        return
    except Exception as exc:
        _log_sidecar_failure(type(exc).__name__)
        return


def _log_sidecar_failure(error_type: str) -> None:
    logger.warning(
        "Search sidecar failed; returning vector-store results only: %s",
        error_type,
    )


def _is_current_task_cancelling() -> bool:
    task = asyncio.current_task()
    if task is None:
        return False
    return task.cancelling() > 0
