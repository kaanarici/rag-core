"""Linear retrieval pipeline runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import SearchStageCompleted, StageError
from rag_core.search.pipeline.types import (
    FuseStage,
    PipelineContext,
    PipelineQuery,
    Postprocess,
    QueryTransform,
    Rerank,
    Retrieve,
)
from rag_core.search.pipeline.stages.sidecar_prefetch import cancel_prefetched_sidecar
from rag_core.search.types import SearchResult

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


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
        sink = cast("EventSink | None", ctx.event_sink)
        try:
            for transform in self.query_transforms:
                started_ms = now_ms()
                try:
                    query = await transform.transform(query, ctx)
                except Exception as exc:
                    _emit_stage_error(sink, stage="query_transform", exc=exc)
                    raise
                emit_event(
                    sink,
                    SearchStageCompleted(
                        stage="query_transform",
                        stage_name=_stage_name(transform),
                        duration_ms=now_ms() - started_ms,
                    ),
                )
            started_ms = now_ms()
            try:
                candidates = await self.retrieve.retrieve(query, ctx)
            except Exception as exc:
                _emit_stage_error(sink, stage="retrieve", exc=exc)
                raise
            emit_event(
                sink,
                SearchStageCompleted(
                    stage="retrieve",
                    stage_name=_stage_name(self.retrieve),
                    result_count=len(candidates),
                    duration_ms=now_ms() - started_ms,
                ),
            )
            started_ms = now_ms()
            try:
                results = await self.fuse.fuse([candidates], query, ctx)
            except Exception as exc:
                _emit_stage_error(sink, stage="fuse", exc=exc)
                raise
            emit_event(
                sink,
                SearchStageCompleted(
                    stage="fuse",
                    stage_name=_stage_name(self.fuse),
                    candidate_count=len(candidates),
                    result_count=len(results),
                    duration_ms=now_ms() - started_ms,
                ),
            )
            if query.rerank:
                started_ms = now_ms()
                candidate_count = len(results)
                try:
                    results = await self.rerank.rerank(results, query, ctx)
                except Exception as exc:
                    _emit_stage_error(sink, stage="rerank", exc=exc)
                    raise
                emit_event(
                    sink,
                    SearchStageCompleted(
                        stage="rerank",
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
                    _emit_stage_error(sink, stage="postprocess", exc=exc)
                    raise
                emit_event(
                    sink,
                    SearchStageCompleted(
                        stage="postprocess",
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


def _emit_stage_error(
    sink: "EventSink | None",
    *,
    stage: str,
    exc: Exception,
) -> None:
    emit_event(
        sink,
        StageError(
            stage=stage,
            error_type=type(exc).__name__,
        ),
    )
