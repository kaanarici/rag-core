from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

import pytest

from rag_core.events.sinks import EventBuffer
from rag_core.events.types import (
    SearchCompleted,
    SearchPlanned,
    SearchStageCompleted,
    StageError,
)
from rag_core.search.pipeline import IdentityFuse, PassThroughRerank, RetrievalPipeline
from rag_core.search.pipeline.types import PipelineContext, PipelineQuery
from rag_core.search.pipeline_runner import SearchPipelineRunner, SearchRequest
from rag_core.search.vector_models import SearchResult

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
)


class _PipelineFactory(Protocol):
    def __call__(self, results: list[SearchResult]) -> RetrievalPipeline: ...


@dataclass(frozen=True)
class _StageCase:
    stage: str
    query_rerank: bool
    factory: _PipelineFactory


class _StaticRetrieve:
    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = list(results or [])

    async def retrieve(
        self, query: PipelineQuery, ctx: PipelineContext
    ) -> list[SearchResult]:
        return list(self._results)


class _FailingTransform:
    async def transform(
        self, query: PipelineQuery, ctx: PipelineContext
    ) -> PipelineQuery:
        raise RuntimeError(f"private transform failure for {query.query}")


class _FailingRetrieve:
    async def retrieve(
        self, query: PipelineQuery, ctx: PipelineContext
    ) -> list[SearchResult]:
        raise RuntimeError(f"private retrieve failure for {query.query}")


class _FailingFuse:
    async def fuse(
        self,
        results: list[list[SearchResult]],
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> list[SearchResult]:
        raise RuntimeError(f"private fuse failure for {query.query}")


class _FailingRerank:
    async def rerank(
        self,
        results: list[SearchResult],
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> list[SearchResult]:
        raise RuntimeError(f"private rerank failure for {query.query}")


class _FailingPostprocess:
    async def postprocess(
        self,
        results: list[SearchResult],
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> list[SearchResult]:
        raise RuntimeError(f"private postprocess failure for {query.query}")


@pytest.mark.parametrize(
    "case",
    [
        _StageCase(
            stage="query_transform",
            query_rerank=False,
            factory=lambda results: RetrievalPipeline(
                retrieve=_StaticRetrieve(results),
                fuse=IdentityFuse(),
                rerank=PassThroughRerank(),
                query_transforms=(_FailingTransform(),),
            ),
        ),
        _StageCase(
            stage="retrieve",
            query_rerank=False,
            factory=lambda results: RetrievalPipeline(
                retrieve=_FailingRetrieve(),
                fuse=IdentityFuse(),
                rerank=PassThroughRerank(),
            ),
        ),
        _StageCase(
            stage="fuse",
            query_rerank=False,
            factory=lambda results: RetrievalPipeline(
                retrieve=_StaticRetrieve(results),
                fuse=_FailingFuse(),
                rerank=PassThroughRerank(),
            ),
        ),
        _StageCase(
            stage="rerank",
            query_rerank=True,
            factory=lambda results: RetrievalPipeline(
                retrieve=_StaticRetrieve(results),
                fuse=IdentityFuse(),
                rerank=_FailingRerank(),
            ),
        ),
        _StageCase(
            stage="postprocess",
            query_rerank=False,
            factory=lambda results: RetrievalPipeline(
                retrieve=_StaticRetrieve(results),
                fuse=IdentityFuse(),
                rerank=PassThroughRerank(),
                postprocesses=(_FailingPostprocess(),),
            ),
        ),
    ],
)
def test_pipeline_stage_failures_emit_sanitized_stage_error(case: _StageCase) -> None:
    async def run() -> None:
        events = EventBuffer()
        pipeline = case.factory([make_search_result(id="hit")])
        query = PipelineQuery(
            query="private query text",
            namespace="ns",
            collections=["corpus"],
            rerank=case.query_rerank,
        )

        with pytest.raises(RuntimeError, match="private"):
            await pipeline.run(query, _context(events))

        errors = [event for event in events.events if isinstance(event, StageError)]
        assert len(errors) == 1
        assert errors[0].stage == case.stage
        assert errors[0].error_type == "RuntimeError"
        assert errors[0].message == ""
        assert "private query text" not in str(errors[0])

        completed = [
            event
            for event in events.events
            if isinstance(event, SearchStageCompleted)
        ]
        assert case.stage not in {event.stage for event in completed}

    asyncio.run(run())


def test_search_pipeline_runner_stage_errors_stay_sanitized_on_public_path() -> None:
    async def run() -> None:
        events = EventBuffer()
        pipeline = RetrievalPipeline(
            retrieve=_FailingRetrieve(),
            fuse=IdentityFuse(),
            rerank=PassThroughRerank(),
        )
        pipeline_runner = SearchPipelineRunner(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(),
            event_sink=events,
            pipeline=pipeline,
        )

        with pytest.raises(RuntimeError, match="private"):
            await pipeline_runner.search(
                SearchRequest(
                    query="private query text",
                    namespace="ns",
                    collections=["corpus"],
                )
            )

        errors = [event for event in events.events if isinstance(event, StageError)]
        assert [(event.stage, event.error_type, event.message) for event in errors] == [
            ("retrieve", "RuntimeError", ""),
            ("search", "RuntimeError", ""),
        ]
        planned = [event for event in events.events if isinstance(event, SearchPlanned)]
        assert len(planned) == 1
        completed = [event for event in events.events if isinstance(event, SearchCompleted)]
        assert len(completed) == 1
        assert completed[0].succeeded is False
        assert completed[0].requested_rerank is False
        assert completed[0].requested_sidecar is True
        assert completed[0].attempted_rerank is False
        assert completed[0].attempted_sidecar is False
        serialized = "\n".join(str(event) for event in errors)
        assert "private query text" not in serialized
        assert "private retrieve failure" not in serialized

    asyncio.run(run())


def _context(events: EventBuffer) -> PipelineContext:
    return PipelineContext(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(),
        event_sink=events,
    )
