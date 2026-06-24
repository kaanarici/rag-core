"""Linear retrieval pipeline runner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from rag_core.events.emit import emit_event, now_ms
from rag_core.events.trace_payload_fields import (
    FUSE_SEARCH_STAGE,
    POSTPROCESS_SEARCH_STAGE,
    QUERY_TRANSFORM_SEARCH_STAGE,
    RERANK_SEARCH_STAGE,
    RETRIEVE_SEARCH_STAGE,
    SearchStageName,
)
from rag_core.events.types import SearchStageCompleted, StageError
from rag_core.search.pipeline.types import (
    FuseStage,
    PipelineContext,
    PipelineQuery,
    PipelineStageState,
    Postprocess,
    QueryTransform,
    Rerank,
    Retrieve,
)
from rag_core.search.pipeline.stages.sidecar_prefetch import cancel_prefetched_sidecar
from rag_core.search.query_plan_presets import rerank_candidate_pool_limit
from rag_core.search.vector_models import SearchResult

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink

_MAX_RETRIEVE_FANOUT_CONCURRENCY = 4
_MAX_QUERY_FANOUT_VARIANTS = 8


@dataclass(frozen=True)
class RetrievalPipeline:
    """Five-stage linear pipeline: QueryTransform[] -> Retrieve -> Fuse -> Rerank -> Postprocess[].

    Frozen on purpose to keep stage composition immutable once constructed.
    """

    retrieve: Retrieve
    fuse: FuseStage
    rerank: Rerank
    query_transforms: tuple[QueryTransform, ...] = ()
    postprocesses: tuple[Postprocess, ...] = ()

    async def run(
        self, query: PipelineQuery, ctx: PipelineContext
    ) -> list[SearchResult]:
        sink = ctx.event_sink
        try:
            for transform in self.query_transforms:
                started_ms = now_ms()
                try:
                    query = await transform.transform(query, ctx)
                except Exception as exc:
                    _emit_stage_error(
                        sink,
                        stage=QUERY_TRANSFORM_SEARCH_STAGE,
                        exc=exc,
                    )
                    raise
                emit_event(
                    sink,
                    SearchStageCompleted(
                        stage=QUERY_TRANSFORM_SEARCH_STAGE,
                        stage_name=_stage_name(transform),
                        duration_ms=now_ms() - started_ms,
                    ),
                )
            started_ms = now_ms()
            if query.query_variants:
                candidate_lists = await _retrieve_fanout(self, query, ctx, sink)
                candidate_count = sum(len(candidates) for candidates in candidate_lists)
            else:
                try:
                    candidates = await self.retrieve.retrieve(query, ctx)
                except Exception as exc:
                    _emit_stage_error(sink, stage=RETRIEVE_SEARCH_STAGE, exc=exc)
                    raise
                emit_event(
                    sink,
                    SearchStageCompleted(
                        stage=RETRIEVE_SEARCH_STAGE,
                        stage_name=_stage_name(self.retrieve),
                        result_count=len(candidates),
                        duration_ms=now_ms() - started_ms,
                    ),
                )
                candidate_lists = [candidates]
                candidate_count = len(candidates)
            started_ms = now_ms()
            try:
                results = await self.fuse.fuse(candidate_lists, query, ctx)
            except Exception as exc:
                _emit_stage_error(sink, stage=FUSE_SEARCH_STAGE, exc=exc)
                raise
            emit_event(
                sink,
                SearchStageCompleted(
                    stage=FUSE_SEARCH_STAGE,
                    stage_name=_stage_name(self.fuse),
                    candidate_count=candidate_count,
                    result_count=len(results),
                    duration_ms=now_ms() - started_ms,
                ),
            )
            if (
                query.query_variants
                and query.rerank
                and getattr(self.rerank, "real_rerank", True)
            ):
                # Bound the fused multi-query pool to the rerank candidate pool
                # (not the final limit) so the reranker still sees documents
                # ranked below it. The rerank stage and the final slice below
                # both reduce back to query.limit.
                results = results[
                    : rerank_candidate_pool_limit(
                        final_limit=query.limit,
                        requested=(
                            query.rerank_budget.candidate_count
                            if query.rerank_budget is not None
                            else None
                        ),
                    )
                ]
            if query.rerank and getattr(self.rerank, "real_rerank", True):
                started_ms = now_ms()
                candidate_count = len(results)
                try:
                    results = await self.rerank.rerank(results, query, ctx)
                except Exception as exc:
                    _emit_stage_error(sink, stage=RERANK_SEARCH_STAGE, exc=exc)
                    raise
                emit_event(
                    sink,
                    SearchStageCompleted(
                        stage=RERANK_SEARCH_STAGE,
                        stage_name=_stage_name(self.rerank),
                        candidate_count=candidate_count,
                        result_count=len(results),
                        duration_ms=now_ms() - started_ms,
                    ),
                )
            for postprocess in self.postprocesses:
                started_ms = now_ms()
                candidate_count = len(results)
                try:
                    results = await postprocess.postprocess(results, query, ctx)
                except Exception as exc:
                    _emit_stage_error(sink, stage=POSTPROCESS_SEARCH_STAGE, exc=exc)
                    raise
                emit_event(
                    sink,
                    SearchStageCompleted(
                        stage=POSTPROCESS_SEARCH_STAGE,
                        stage_name=_stage_name(postprocess),
                        candidate_count=candidate_count,
                        result_count=len(results),
                        duration_ms=now_ms() - started_ms,
                    ),
                )
            return results[: query.limit]
        finally:
            await cancel_prefetched_sidecar(query)


def _stage_name(stage: object) -> str:
    return type(stage).__name__


async def _retrieve_fanout(
    pipeline: RetrievalPipeline,
    query: PipelineQuery,
    ctx: PipelineContext,
    sink: "EventSink | None",
) -> list[list[SearchResult]]:
    semaphore = asyncio.Semaphore(_MAX_RETRIEVE_FANOUT_CONCURRENCY)
    fanout_queries = [
        query,
        *[
            _variant_query(query, variant)
            for variant in query.query_variants[:_MAX_QUERY_FANOUT_VARIANTS]
        ],
    ]

    async def retrieve_one(fanout_query: PipelineQuery) -> list[SearchResult]:
        async with semaphore:
            started_ms = now_ms()
            try:
                candidates = await pipeline.retrieve.retrieve(fanout_query, ctx)
            except Exception as exc:
                _emit_stage_error(sink, stage=RETRIEVE_SEARCH_STAGE, exc=exc)
                raise
            emit_event(
                sink,
                SearchStageCompleted(
                    stage=RETRIEVE_SEARCH_STAGE,
                    stage_name=_stage_name(pipeline.retrieve),
                    result_count=len(candidates),
                    duration_ms=now_ms() - started_ms,
                ),
            )
            return candidates

    tasks = [
        asyncio.create_task(retrieve_one(fanout_query))
        for fanout_query in fanout_queries
    ]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        raise


def _variant_query(query: PipelineQuery, variant: str) -> PipelineQuery:
    return replace(
        query,
        query=variant,
        dense_query_text=None,
        query_vector=None,
        query_sparse_vectors=None,
        query_plan_trace_callback=None,
        query_plan_trace_emitted=True,
        state=PipelineStageState(collection_ensured=query.state.collection_ensured),
    )


def _emit_stage_error(
    sink: "EventSink | None",
    *,
    stage: SearchStageName,
    exc: Exception,
) -> None:
    emit_event(
        sink,
        StageError(
            stage=stage,
            error_type=type(exc).__name__,
        ),
    )
