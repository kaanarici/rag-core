"""Sidecar merge stages that keep vector search and sidecar search overlapped."""

from __future__ import annotations

from rag_core.search.pipeline.merge_strategies import (
    PreferSidecarMerge,
    SidecarMergeStrategy,
)
from rag_core.search.pipeline.types import PipelineContext, PipelineQuery
from rag_core.search.pipeline.stages.sidecar_application import (
    emit_sidecar_failure,
    emit_sidecar_success,
    prepare_sidecar_application,
    sidecar_provider_name,
)
from rag_core.search.pipeline.stages.sidecar_prefetch import (
    cancel_prefetched_sidecar,
    resolve_sidecar_results,
    start_prefetched_sidecar,
)
from rag_core.search.vector_models import SearchResult

class SidecarPrefetchTransform:
    """Fire the sidecar query early so its I/O overlaps with vector search.

    The in-flight task is stored in ``query.state`` for ``SidecarPostprocess``.
    """

    async def transform(
        self, query: PipelineQuery, ctx: PipelineContext
    ) -> PipelineQuery:
        if ctx.sidecar is None or not query.use_lexical_search:
            await cancel_prefetched_sidecar(query)
            return query
        await start_prefetched_sidecar(query, ctx.sidecar)
        return query


class SidecarPostprocess:
    """Await the prefetched sidecar results (or run the query inline) and merge."""

    def __init__(self, strategy: SidecarMergeStrategy | None = None) -> None:
        self._strategy: SidecarMergeStrategy = strategy or PreferSidecarMerge()

    async def postprocess(
        self,
        results: list[SearchResult],
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> list[SearchResult]:
        if ctx.sidecar is None or not query.use_lexical_search:
            await cancel_prefetched_sidecar(query)
            return results
        ctx.execution.attempted_sidecar = True
        sink = ctx.event_sink
        provider = sidecar_provider_name(ctx.sidecar)
        resolution = await resolve_sidecar_results(query, ctx.sidecar)
        if resolution.error_type:
            emit_sidecar_failure(
                sink,
                provider=provider,
                input_count=len(results),
                result_count=len(results),
                duration_ms=resolution.duration_ms,
                fallback_reason=resolution.error_type,
            )
            return results
        application = prepare_sidecar_application(query, resolution.results, results)
        if not application.accepted_results:
            emit_sidecar_success(
                sink,
                provider=provider,
                input_count=len(results),
                application=application,
                result_count=len(results),
                duration_ms=resolution.duration_ms,
            )
            return results
        merged = (await self._strategy.merge(results, application.accepted_results))[
            : query.limit
        ]
        ctx.execution.applied_sidecar = True
        emit_sidecar_success(
            sink,
            provider=provider,
            input_count=len(results),
            application=application,
            result_count=len(merged),
            duration_ms=resolution.duration_ms,
        )
        return merged
