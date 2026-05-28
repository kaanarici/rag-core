"""Prefetch task lifecycle for search sidecar postprocessing."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from rag_core.events.emit import now_ms
from rag_core.search.pipeline.stages.sidecar_application import build_sidecar_query
from rag_core.search.pipeline.types import PipelineQuery, PipelineSidecarPrefetch
from rag_core.search.provider_protocols import SearchSidecar
from rag_core.search.vector_models import SearchResult

logger = logging.getLogger("rag_core.search.pipeline.stages.sidecar_postprocess")

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
    query.state.sidecar_prefetch = PipelineSidecarPrefetch(
        task=asyncio.create_task(sidecar.search(sidecar_query)),
        started_ms=now_ms(),
    )


async def resolve_sidecar_results(
    query: PipelineQuery,
    sidecar: SearchSidecar | None,
) -> SidecarResolution:
    prefetched = query.state.sidecar_prefetch
    query.state.sidecar_prefetch = None
    if prefetched is not None:
        try:
            sidecar_results = await prefetched.task
        except asyncio.CancelledError:
            if _is_current_task_cancelling():
                raise
            error_type = "CancelledError"
            _log_sidecar_failure(error_type)
            return SidecarResolution(
                results=[],
                duration_ms=now_ms() - prefetched.started_ms,
                error_type=error_type,
            )
        except Exception as exc:
            error_type = type(exc).__name__
            _log_sidecar_failure(error_type)
            return SidecarResolution(
                results=[],
                duration_ms=now_ms() - prefetched.started_ms,
                error_type=error_type,
            )
        return SidecarResolution(
            results=sidecar_results,
            duration_ms=now_ms() - prefetched.started_ms,
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
    prefetched = query.state.sidecar_prefetch
    query.state.sidecar_prefetch = None
    if prefetched is None:
        return
    if not prefetched.task.done():
        prefetched.task.cancel()
    try:
        await prefetched.task
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
