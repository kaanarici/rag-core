"""Tests for the linear retrieval pipeline shape and runner."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import cast

import pytest

from rag_core.events.sinks import EventBuffer
from rag_core.events.types import RerankApplied
from rag_core.retrieval_defaults import DEFAULT_SEARCH_LIMIT
from rag_core.search.pipeline import (
    HybridRetrieve,
    IdentityFuse,
    IdentityPostprocess,
    IdentityQueryTransform,
    PassThroughRerank,
    PipelineContext,
    PipelineQuery,
    Postprocess,
    ProviderRerankStage,
    QueryTransform,
    RetrievalPipeline,
)
from rag_core.search.provider_protocols import RerankerProvider
from rag_core.search.query_plan import DenseChannel
from rag_core.search.request_models import (
    RerankBudget,
    RerankResult,
)
from rag_core.search.vector_models import (
    SearchResult,
    SparseVector,
)

from tests.support import (
    FakeEmbeddingProvider,
    FakeReranker,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
)


def _build_query(**overrides: object) -> PipelineQuery:
    base: dict[str, object] = {
        "query": "fox query",
        "namespace": "space-1",
        "collections": ["corpus-1"],
        "limit": 20,
    }
    base.update(overrides)
    return PipelineQuery(**base)  # type: ignore[arg-type]


def _build_context(
    *,
    store: RecordingVectorStore | None = None,
    reranker: FakeReranker | None = None,
    event_sink: EventBuffer | None = None,
) -> PipelineContext:
    return PipelineContext(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store or RecordingVectorStore(),
        reranker=reranker,
        event_sink=event_sink,
    )


class _StaticRetrieve:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def retrieve(
        self, query: PipelineQuery, ctx: PipelineContext
    ) -> list[SearchResult]:
        return list(self._results)


def _identity_pipeline(
    results: list[SearchResult],
    *,
    rerank: object | None = None,
    query_transforms: tuple[QueryTransform, ...] = (),
    postprocesses: tuple[Postprocess, ...] = (),
) -> RetrievalPipeline:
    return RetrievalPipeline(
        retrieve=_StaticRetrieve(results),
        fuse=IdentityFuse(),
        rerank=rerank or PassThroughRerank(),  # type: ignore[arg-type]
        query_transforms=query_transforms,
        postprocesses=postprocesses,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidate_count": 0},
        {"timeout_seconds": 0.0},
        {"max_output": 0},
    ],
)
def test_rerank_budget_rejects_non_positive_limits(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RerankBudget(**kwargs)  # type: ignore[arg-type]


def test_identity_pipeline_passes_results_through_and_handles_empty() -> None:
    async def _run() -> None:
        hits = [make_search_result(id="a"), make_search_result(id="b")]
        through = await _identity_pipeline(hits).run(_build_query(), _build_context())
        assert [r.id for r in through] == ["a", "b"]
        empty = await _identity_pipeline([]).run(_build_query(), _build_context())
        assert empty == []

    asyncio.run(_run())

def test_identity_stages_are_no_ops() -> None:
    """IdentityPostprocess / IdentityQueryTransform / PassThroughRerank leave inputs alone."""

    async def _run() -> None:
        hits = [make_search_result(id="a")]
        query = _build_query()
        ctx = _build_context()

        pp_out = await IdentityPostprocess().postprocess(hits, query, ctx)
        assert pp_out == hits

        rerank_out = await PassThroughRerank().rerank(hits, query, ctx)
        assert rerank_out == hits

        transform_out = await IdentityQueryTransform().transform(query, ctx)
        assert transform_out is query

    asyncio.run(_run())


def test_query_transforms_observed_in_retrieve_and_run_in_order() -> None:
    async def _run() -> None:
        seen: list[str] = []
        order: list[str] = []

        class CaptureRetrieve:
            async def retrieve(
                self, query: PipelineQuery, ctx: PipelineContext
            ) -> list[SearchResult]:
                seen.append(query.query)
                return []

        def make_tag(name: str, suffix: str) -> QueryTransform:
            class Tag:
                async def transform(
                    self, query: PipelineQuery, ctx: PipelineContext
                ) -> PipelineQuery:
                    order.append(name)
                    return replace(query, query=f"{query.query}{suffix}")

            return Tag()

        pipeline = RetrievalPipeline(
            retrieve=CaptureRetrieve(),
            fuse=IdentityFuse(),
            rerank=PassThroughRerank(),
            query_transforms=(make_tag("first", "::a"), make_tag("second", "::b")),
        )
        await pipeline.run(_build_query(query="raw"), _build_context())
        assert order == ["first", "second"]
        assert seen == ["raw::a::b"]

    asyncio.run(_run())


def test_postprocess_can_decorate_and_read_query_state() -> None:
    async def _run() -> None:
        captured_limit: list[int] = []

        class TagAndRead:
            async def postprocess(
                self,
                results: list[SearchResult],
                query: PipelineQuery,
                ctx: PipelineContext,
            ) -> list[SearchResult]:
                captured_limit.append(query.limit)
                return [
                    replace(r, metadata={**r.metadata, "tagged": True}) for r in results
                ]

        pipeline = _identity_pipeline(
            [make_search_result(id="a")], postprocesses=(TagAndRead(),)
        )
        results = await pipeline.run(_build_query(limit=7), _build_context())
        assert results[0].metadata["tagged"] is True
        assert captured_limit == [7]

    asyncio.run(_run())


def test_postprocesses_run_in_order() -> None:
    async def _run() -> None:
        order: list[str] = []

        def make_postprocess(name: str) -> Postprocess:
            class Tag:
                async def postprocess(
                    self,
                    results: list[SearchResult],
                    query: PipelineQuery,
                    ctx: PipelineContext,
                ) -> list[SearchResult]:
                    order.append(name)
                    return results

            return Tag()

        pipeline = _identity_pipeline(
            [make_search_result()],
            postprocesses=(make_postprocess("a"), make_postprocess("b")),
        )
        await pipeline.run(_build_query(), _build_context())
        assert order == ["a", "b"]

    asyncio.run(_run())


def test_pipeline_truncates_to_query_limit() -> None:
    async def _run() -> None:
        hits = [make_search_result(id=str(i)) for i in range(5)]
        pipeline = _identity_pipeline(hits)
        results = await pipeline.run(_build_query(limit=3), _build_context())
        assert [r.id for r in results] == ["0", "1", "2"]

    asyncio.run(_run())


def test_identity_fuse_handles_empty_and_returns_first_list() -> None:
    async def _run() -> None:
        fuse = IdentityFuse()
        assert await fuse.fuse([], _build_query(), _build_context()) == []
        first = [make_search_result(id="a")]
        second = [make_search_result(id="b")]
        assert await fuse.fuse([first, second], _build_query(), _build_context()) == first

    asyncio.run(_run())


def test_hybrid_retrieve_uses_precomputed_vectors_when_present() -> None:
    async def _run() -> None:
        embedding = FakeEmbeddingProvider()
        sparse = FakeSparseEmbedder()
        store = RecordingVectorStore(search_results=[make_search_result()])
        sparse_vectors = {"bm25": SparseVector(indices=[1], values=[1.0])}
        pipeline = RetrievalPipeline(
            retrieve=HybridRetrieve(),
            fuse=IdentityFuse(),
            rerank=PassThroughRerank(),
        )
        ctx = PipelineContext(
            embedding_provider=embedding,
            sparse_embedder=sparse,
            vector_store=store,
        )
        await pipeline.run(
            _build_query(
                query_vector=[1.0, 2.0, 3.0, 4.0], query_sparse_vectors=sparse_vectors
            ),
            ctx,
        )
        assert embedding.embed_query_calls == []
        assert sparse.embed_query_multi_calls == []
        assert store.search_calls[0].dense_vector == [1.0, 2.0, 3.0, 4.0]
        assert store.search_calls[0].sparse_vector == sparse_vectors["bm25"]

    asyncio.run(_run())


def test_hybrid_retrieve_embeds_when_vectors_missing() -> None:
    async def _run() -> None:
        embedding = FakeEmbeddingProvider()
        sparse = FakeSparseEmbedder()
        pipeline = RetrievalPipeline(
            retrieve=HybridRetrieve(),
            fuse=IdentityFuse(),
            rerank=PassThroughRerank(),
        )
        ctx = PipelineContext(
            embedding_provider=embedding,
            sparse_embedder=sparse,
            vector_store=RecordingVectorStore(),
        )
        await pipeline.run(_build_query(query="raw"), ctx)
        assert embedding.embed_query_calls == ["raw"]
        assert sparse.embed_query_multi_calls == ["raw"]

    asyncio.run(_run())


def test_hybrid_retrieve_implicit_plan_falls_back_to_dense_when_sparse_is_missing() -> None:
    async def _run() -> RecordingVectorStore:
        store = RecordingVectorStore()
        pipeline = RetrievalPipeline(
            retrieve=HybridRetrieve(),
            fuse=IdentityFuse(),
            rerank=PassThroughRerank(),
        )
        ctx = PipelineContext(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(empty_query_multi=True),
            vector_store=store,
        )
        await pipeline.run(_build_query(query_vector=[1.0, 2.0, 3.0, 4.0]), ctx)
        return store

    store = asyncio.run(_run())
    [call] = store.search_calls
    assert call.query_plan is not None
    assert len(call.query_plan.prefetches) == 1
    assert isinstance(call.query_plan.prefetches[0].channel, DenseChannel)


def test_hybrid_retrieve_widens_candidate_pool_for_provider_rerank() -> None:
    """When a real reranker is present, retrieve must fetch a candidate pool
    wider than the final limit so the reranker can promote documents ranked
    below it. Without a reranker, only the final limit is fetched. The caller's
    final limit is never mutated. This is the upstream half of the recall fix:
    a reranker can only reorder what retrieval returned.
    """

    async def _retrieve(
        *, reranker: "RerankerProvider | None"
    ) -> tuple[RecordingVectorStore, int]:
        store = RecordingVectorStore(search_results=[make_search_result()])
        query = _build_query(query="raw", limit=10, rerank=True)
        ctx = PipelineContext(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            reranker=reranker,
        )
        await HybridRetrieve().retrieve(query, ctx)
        return store, query.limit

    wide_store, wide_final = asyncio.run(_retrieve(reranker=FakeReranker()))
    narrow_store, narrow_final = asyncio.run(_retrieve(reranker=None))

    # limit=10 -> rerank pool of 40 (4x, floor 20, cap 200).
    assert wide_store.search_calls[0].limit == 40
    assert wide_store.search_calls[0].query_plan is not None
    assert wide_store.search_calls[0].query_plan.final_limit == 40
    assert wide_final == 10  # caller's final limit preserved, only the pool widened

    # No real reranker present: fetch only the final limit, no waste.
    assert narrow_store.search_calls[0].limit == 10
    assert narrow_store.search_calls[0].query_plan is not None
    assert narrow_store.search_calls[0].query_plan.final_limit == 10
    assert narrow_final == 10


def test_rerank_stage_reorders_results_when_requested() -> None:
    async def _run() -> None:
        hits = [
            make_search_result(id="a", text="alpha", score=0.4),
            make_search_result(id="b", text="beta", score=0.7),
        ]
        reranker = FakeReranker(
            results=[
                RerankResult(index=1, score=0.95, text=hits[1].text),
                RerankResult(index=0, score=0.9, text=hits[0].text),
            ]
        )
        pipeline = _identity_pipeline(hits, rerank=ProviderRerankStage())
        result = await pipeline.run(
            _build_query(rerank=True), _build_context(reranker=reranker)
        )
        assert [hit.id for hit in result] == ["b", "a"]
        assert [hit.score for hit in result] == [0.7, 0.4]
        rerank_meta = cast(dict[str, object], result[0].metadata["rerank"])
        assert rerank_meta["rerank_rank"] == 1
        assert rerank_meta["provider_score"] == 0.95
        assert rerank_meta["search_score"] == 0.7

    asyncio.run(_run())


def test_rerank_stage_applies_budget_and_emits_event() -> None:
    async def _run() -> None:
        hits = [
            make_search_result(id="a", text="a", score=0.7),
            make_search_result(id="b", text="b", score=0.2),
            make_search_result(id="c", text="c", score=0.4),
            make_search_result(id="d", text="d", score=0.1),
        ]
        reranker = FakeReranker(
            results=[
                RerankResult(index=1, score=0.95, text=hits[1].text),
                RerankResult(index=0, score=0.9, text=hits[0].text),
            ]
        )
        events = EventBuffer()
        pipeline = _identity_pipeline(hits, rerank=ProviderRerankStage())

        result = await pipeline.run(
            _build_query(
                rerank=True,
                rerank_budget=RerankBudget(candidate_count=3, max_output=2),
                limit=4,
            ),
            _build_context(reranker=reranker, event_sink=events),
        )

        assert reranker.calls == [("fox query", ["a", "b", "c"], 2)]
        assert [hit.id for hit in result] == ["b", "a", "c", "d"]
        assert [hit.score for hit in result] == [0.2, 0.7, 0.4, 0.1]
        rerank_meta = cast(dict[str, object], result[0].metadata["rerank"])
        assert rerank_meta["provider_score"] == 0.95
        assert rerank_meta["search_score"] == 0.2
        applied = [e for e in events.events if isinstance(e, RerankApplied)]
        assert len(applied) == 1
        assert applied[0].input_count == 4
        assert applied[0].candidate_count == 3
        assert applied[0].result_count == 4
        assert applied[0].top_k == 2
        assert applied[0].truncation_reason == "candidate_count,max_output"

    asyncio.run(_run())


def test_rerank_stage_clamps_over_returned_provider_rows_to_top_k() -> None:
    async def _run() -> None:
        hits = [
            make_search_result(id="a", text="a", score=0.7),
            make_search_result(id="b", text="b", score=0.2),
            make_search_result(id="c", text="c", score=0.4),
            make_search_result(id="d", text="d", score=0.1),
        ]
        reranker = FakeReranker(
            results=[
                RerankResult(index=1, score=0.95, text=hits[1].text),
                RerankResult(index=0, score=0.9, text=hits[0].text),
                RerankResult(index=2, score=0.85, text=hits[2].text),
                RerankResult(index=3, score=0.8, text=hits[3].text),
            ]
        )
        events = EventBuffer()
        pipeline = _identity_pipeline(hits, rerank=ProviderRerankStage())

        result = await pipeline.run(
            _build_query(
                rerank=True,
                rerank_budget=RerankBudget(candidate_count=4, max_output=2),
                limit=4,
            ),
            _build_context(reranker=reranker, event_sink=events),
        )

        assert reranker.calls == [("fox query", ["a", "b", "c", "d"], 2)]
        assert [hit.id for hit in result] == ["b", "a", "c", "d"]
        assert [hit.score for hit in result] == [0.2, 0.7, 0.4, 0.1]
        applied = [e for e in events.events if isinstance(e, RerankApplied)]
        assert len(applied) == 1
        assert applied[0].provider_result_count == 4
        assert applied[0].accepted_count == 2
        assert applied[0].dropped_count == 2
        assert applied[0].succeeded is False

    asyncio.run(_run())


def test_rerank_stage_skipped_when_disabled_or_no_provider() -> None:
    """Rerank is opt-in: ``query.rerank=False`` or a missing provider keeps order."""

    async def _run() -> None:
        hits = [make_search_result(id="a"), make_search_result(id="b")]
        reranker = FakeReranker(
            results=[RerankResult(index=1, score=0.9, text=hits[1].text)]
        )
        pipeline = _identity_pipeline(hits, rerank=ProviderRerankStage())

        skipped = await pipeline.run(_build_query(), _build_context(reranker=reranker))
        assert [hit.id for hit in skipped] == ["a", "b"]
        assert reranker.calls == []

        no_provider = await pipeline.run(_build_query(rerank=True), _build_context())
        assert [hit.id for hit in no_provider] == ["a", "b"]

    asyncio.run(_run())


def test_rerank_failure_falls_back_unless_explicitly_disabled() -> None:
    async def _run() -> None:
        hits = [make_search_result(id="a"), make_search_result(id="b")]
        reranker = FakeReranker(error=RuntimeError("boom"))
        pipeline = _identity_pipeline(hits, rerank=ProviderRerankStage())

        fallback = await pipeline.run(
            _build_query(rerank=True), _build_context(reranker=reranker)
        )
        assert [hit.id for hit in fallback] == ["a", "b"]

        with pytest.raises(RuntimeError, match="boom"):
            await pipeline.run(
                _build_query(
                    rerank=True,
                    rerank_budget=RerankBudget(fallback_on_error=False),
                ),
                _build_context(reranker=reranker),
            )

    asyncio.run(_run())


def test_rerank_cancelled_error_falls_back_unless_explicitly_disabled() -> None:
    class _CancelledReranker:
        async def rerank(
            self,
            query: str,
            documents: list[str],
            top_k: int = DEFAULT_SEARCH_LIMIT,
        ) -> list[RerankResult]:
            raise asyncio.CancelledError()

    async def _run() -> None:
        hits = [make_search_result(id="a"), make_search_result(id="b")]
        reranker = _CancelledReranker()
        pipeline = _identity_pipeline(hits, rerank=ProviderRerankStage())

        fallback = await pipeline.run(
            _build_query(rerank=True),
            _build_context(reranker=cast(FakeReranker, reranker)),
        )
        assert [hit.id for hit in fallback] == ["a", "b"]

        with pytest.raises(asyncio.CancelledError):
            await pipeline.run(
                _build_query(
                    rerank=True,
                    rerank_budget=RerankBudget(fallback_on_error=False),
                ),
                _build_context(reranker=cast(FakeReranker, reranker)),
            )

    asyncio.run(_run())


def test_rerank_timeout_falls_back_without_leaking_query_text() -> None:
    async def _run() -> None:
        hits = [make_search_result(id="a"), make_search_result(id="b")]
        reranker = FakeReranker(delay_seconds=0.05)
        events = EventBuffer()
        pipeline = _identity_pipeline(hits, rerank=ProviderRerankStage())

        result = await pipeline.run(
            _build_query(
                query="private query text",
                rerank=True,
                rerank_budget=RerankBudget(timeout_seconds=0.001),
            ),
            _build_context(reranker=reranker, event_sink=events),
        )

        assert [hit.id for hit in result] == ["a", "b"]
        applied = [e for e in events.events if isinstance(e, RerankApplied)]
        assert len(applied) == 1
        assert applied[0].succeeded is False
        assert applied[0].fallback_reason == "timeout"
        assert "private query text" not in str(applied[0])

    asyncio.run(_run())


def test_rerank_stage_ignores_invalid_indices_and_empty_input() -> None:
    """Reranker returning out-of-range indices is a degenerate response; we drop them."""

    async def _run() -> None:
        hits = [make_search_result(id="a"), make_search_result(id="b")]
        reranker = FakeReranker(
            results=[
                RerankResult(index=99, score=0.9, text="bogus"),
                RerankResult(index=0, score=0.8, text=hits[0].text),
            ]
        )
        pipeline = _identity_pipeline(hits, rerank=ProviderRerankStage())
        result = await pipeline.run(
            _build_query(rerank=True), _build_context(reranker=reranker)
        )
        assert [hit.id for hit in result] == ["a", "b"]
        assert result[0].score == 0.9
        rerank_meta = cast(dict[str, object], result[0].metadata["rerank"])
        assert rerank_meta["rerank_rank"] == 1
        assert rerank_meta["provider_score"] == 0.8

        empty_reranker = FakeReranker()
        empty_pipeline = _identity_pipeline([], rerank=ProviderRerankStage())
        empty_result = await empty_pipeline.run(
            _build_query(rerank=True), _build_context(reranker=empty_reranker)
        )
        assert empty_result == []
        assert empty_reranker.calls == []

    asyncio.run(_run())


def test_pipeline_runs_with_no_query_transforms_or_postprocesses() -> None:
    async def _run() -> None:
        pipeline = _identity_pipeline([make_search_result()])
        assert pipeline.query_transforms == ()
        assert pipeline.postprocesses == ()
        result = await pipeline.run(_build_query(), _build_context())
        assert len(result) == 1

    asyncio.run(_run())
