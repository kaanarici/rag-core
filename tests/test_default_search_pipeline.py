"""SearchOrchestrator drives the default pipeline and accepts custom pipelines."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from rag_core.events import EventBuffer, SearchPlanned, SearchStarted
from rag_core.search.pipeline import (
    HybridRetrieve,
    IdentityFuse,
    PassThroughRerank,
    PipelineContext,
    PipelineQuery,
    PreferSidecarMerge,
    RetrievalPipeline,
    SidecarPostprocess,
    SidecarPrefetchTransform,
)
from rag_core.search.searcher import (
    SearchOrchestrator,
    SearchRequest,
)
from rag_core.search.planning import query_plan_preset, search_profile
from rag_core.search.query_plan import DenseChannel, Prefetch, QueryPlan
from rag_core.search.types import (
    QueryPlanCapabilities,
    RerankBudget,
    RerankResult,
    SearchQuery,
    SearchResult,
    SearchSidecarQuery,
    StoreCapabilities,
)

from tests.support import (
    FakeEmbeddingProvider,
    FakeReranker,
    FakeSearchSidecar,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
)


def _orchestrator(
    *,
    store: RecordingVectorStore | None = None,
    sidecar: FakeSearchSidecar | None = None,
    reranker: FakeReranker | None = None,
    pipeline: RetrievalPipeline | None = None,
) -> SearchOrchestrator:
    return SearchOrchestrator(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store or RecordingVectorStore(),
        sidecar=sidecar,
        reranker=reranker,
        pipeline=pipeline,
    )


class _InjectDenseOnlyPlanTransform:
    async def transform(
        self,
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> PipelineQuery:
        del ctx
        query.query_plan = query_plan_preset("dense_only", limit=1)
        return query


def test_default_pipeline_with_sidecar_uses_prefetch_and_postprocess() -> None:
    """Wiring contract: with a sidecar, prefetch + postprocess get installed."""
    orchestrator = _orchestrator(sidecar=FakeSearchSidecar())
    pipeline = orchestrator._pipeline
    assert any(isinstance(t, SidecarPrefetchTransform) for t in pipeline.query_transforms)
    assert any(isinstance(p, SidecarPostprocess) for p in pipeline.postprocesses)

    bare = _orchestrator()
    assert bare._pipeline.query_transforms == ()
    assert bare._pipeline.postprocesses == ()


def test_search_with_default_pipeline_merges_sidecar_and_vector_results() -> None:
    async def _run() -> None:
        semantic = make_search_result(id="doc-semantic", text="fox context")
        exact = make_search_result(
            id="doc-exact", text="fox query text", title="Fox Query", score=1.0
        )
        sidecar = FakeSearchSidecar(results=[exact])
        orchestrator = _orchestrator(
            store=RecordingVectorStore(search_results=[semantic]),
            sidecar=sidecar,
        )
        results = await orchestrator.search(
            SearchRequest(
                query="fox query", corpus_ids=["corpus-1"], namespace="space-1"
            )
        )
        assert [r.id for r in results] == ["doc-exact", "doc-semantic"]
        assert sidecar.calls[0].namespace == "space-1"

    asyncio.run(_run())


def test_custom_pipeline_overrides_default() -> None:
    async def _run() -> None:
        captured: list[str] = []

        class Tag:
            async def postprocess(
                self,
                results: list[SearchResult],
                query: PipelineQuery,
                ctx: PipelineContext,
            ) -> list[SearchResult]:
                captured.append("custom")
                return [replace(r, score=99.0) for r in results]

        custom = RetrievalPipeline(
            retrieve=HybridRetrieve(),
            fuse=IdentityFuse(),
            rerank=PassThroughRerank(),
            postprocesses=(Tag(),),
        )
        orchestrator = _orchestrator(
            store=RecordingVectorStore(search_results=[make_search_result(id="a")]),
            pipeline=custom,
        )
        results = await orchestrator.search(
            SearchRequest(query="query", corpus_ids=["corpus-1"], namespace="space-1")
        )
        assert captured == ["custom"]
        assert results[0].score == 99.0

    asyncio.run(_run())


def test_sidecar_can_be_skipped_per_request_and_failures_are_isolated() -> None:
    async def _run() -> None:
        sidecar = FakeSearchSidecar(
            results=[make_search_result(id="doc-exact", score=1.0)]
        )
        orchestrator = _orchestrator(
            store=RecordingVectorStore(
                search_results=[make_search_result(id="doc-store")]
            ),
            sidecar=sidecar,
        )

        skipped = await orchestrator.search(
            SearchRequest(
                query="fox query",
                corpus_ids=["corpus-1"],
                namespace="space-1",
                use_sidecar=False,
            )
        )
        assert [r.id for r in skipped] == ["doc-store"]
        assert sidecar.calls == []

        broken = FakeSearchSidecar(error=RuntimeError("sidecar failure"))
        with_failure = _orchestrator(
            store=RecordingVectorStore(
                search_results=[make_search_result(id="vec-only", score=0.7)]
            ),
            sidecar=broken,
        )
        results = await with_failure.search(
            SearchRequest(query="q", corpus_ids=["c"], namespace="n")
        )
        assert [r.id for r in results] == ["vec-only"]

    asyncio.run(_run())


def test_explicit_query_plan_disables_sidecar_merge() -> None:
    async def _run() -> None:
        sidecar = FakeSearchSidecar(results=[make_search_result(id="doc-sidecar")])
        store = RecordingVectorStore(search_results=[make_search_result(id="doc-store")])
        orchestrator = _orchestrator(store=store, sidecar=sidecar)
        explicit_plan = QueryPlan(
            prefetches=(Prefetch(channel=DenseChannel(), limit=5),),
            final_limit=1,
        )

        results = await orchestrator.search(
            SearchRequest(
                query="q",
                corpus_ids=["c"],
                namespace="n",
                limit=1,
                query_plan=explicit_plan,
            )
        )

        assert [r.id for r in results] == ["doc-store"]
        assert sidecar.calls == []

    asyncio.run(_run())


def test_explicit_query_plan_preset_disables_sidecar_merge() -> None:
    async def _run() -> None:
        sidecar = FakeSearchSidecar(results=[make_search_result(id="doc-sidecar")])
        store = RecordingVectorStore(search_results=[make_search_result(id="doc-store")])
        orchestrator = _orchestrator(store=store, sidecar=sidecar)

        results = await orchestrator.search(
            SearchRequest(
                query="q",
                corpus_ids=["c"],
                namespace="n",
                limit=2,
                query_plan=query_plan_preset("hybrid_rrf", limit=2),
            )
        )

        assert [r.id for r in results] == ["doc-store"]
        assert sidecar.calls == []

    asyncio.run(_run())


def test_named_search_profile_plan_disables_sidecar_merge() -> None:
    async def _run() -> None:
        sidecar = FakeSearchSidecar(
            results=[make_search_result(id="doc-sidecar", score=1.0)]
        )
        store = RecordingVectorStore(
            search_results=[make_search_result(id="doc-store", score=0.4)]
        )
        orchestrator = _orchestrator(store=store, sidecar=sidecar)

        results = await orchestrator.search(
            SearchRequest(
                query="q",
                corpus_ids=["corpus-1"],
                namespace="space-1",
                limit=2,
                query_plan=search_profile("fast", limit=2),
            )
        )

        assert [r.id for r in results] == ["doc-store"]
        assert sidecar.calls == []

    asyncio.run(_run())


def test_query_plan_final_limit_mismatch_canonicalizes_request_limit() -> None:
    async def _run() -> None:
        events = EventBuffer()
        sidecar = FakeSearchSidecar(
            results=[make_search_result(id="doc-sidecar", namespace="n", corpus_id="c")]
        )
        store = RecordingVectorStore(
            search_results=[
                make_search_result(id="doc-store", namespace="n", corpus_id="c")
            ]
        )
        orchestrator = SearchOrchestrator(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            sidecar=sidecar,
            event_sink=events,
        )
        mismatched_plan = search_profile("fast", limit=1)

        results = await orchestrator.search(
            SearchRequest(
                query="q",
                corpus_ids=["c"],
                namespace="n",
                limit=2,
                query_plan=mismatched_plan,
            )
        )

        assert [r.id for r in results] == ["doc-store"]
        assert sidecar.calls == []
        assert store.search_calls[0].limit == 1
        assert store.search_calls[0].query_plan is mismatched_plan
        [started] = [event for event in events.events if isinstance(event, SearchStarted)]
        [planned] = [event for event in events.events if isinstance(event, SearchPlanned)]
        assert started.limit == 1
        assert planned.limit == 1

    asyncio.run(_run())


def test_custom_pipeline_with_prefer_sidecar_merge_wins_on_duplicate_id() -> None:
    async def _run() -> None:
        sidecar = FakeSearchSidecar(
            results=[
                make_search_result(
                    id="dup",
                    score=0.0,
                    text="sidecar text",
                    namespace="n",
                    corpus_id="c",
                )
            ]
        )
        store = RecordingVectorStore(
            search_results=[make_search_result(id="dup", score=0.7, text="vector text")]
        )
        pipeline = RetrievalPipeline(
            retrieve=HybridRetrieve(),
            fuse=IdentityFuse(),
            rerank=PassThroughRerank(),
            query_transforms=(SidecarPrefetchTransform(),),
            postprocesses=(SidecarPostprocess(strategy=PreferSidecarMerge()),),
        )
        orchestrator = _orchestrator(store=store, sidecar=sidecar, pipeline=pipeline)
        results = await orchestrator.search(
            SearchRequest(query="q", corpus_ids=["c"], namespace="n")
        )
        assert [r.id for r in results] == ["dup"]
        assert results[0].text == "sidecar text"

    asyncio.run(_run())


def test_custom_pipeline_keeps_sidecar_with_explicit_query_plan() -> None:
    async def _run() -> None:
        sidecar = FakeSearchSidecar(
            results=[
                make_search_result(
                    id="dup",
                    score=0.0,
                    text="sidecar text",
                    namespace="n",
                    corpus_id="c",
                )
            ]
        )
        store = RecordingVectorStore(
            search_results=[make_search_result(id="dup", score=0.7, text="vector text")]
        )
        pipeline = RetrievalPipeline(
            retrieve=HybridRetrieve(),
            fuse=IdentityFuse(),
            rerank=PassThroughRerank(),
            query_transforms=(SidecarPrefetchTransform(),),
            postprocesses=(SidecarPostprocess(strategy=PreferSidecarMerge()),),
        )
        orchestrator = _orchestrator(store=store, sidecar=sidecar, pipeline=pipeline)
        results = await orchestrator.search(
            SearchRequest(
                query="q",
                corpus_ids=["c"],
                namespace="n",
                query_plan=query_plan_preset("dense_only", limit=1),
            )
        )
        assert [r.id for r in results] == ["dup"]
        assert results[0].text == "sidecar text"
        assert len(sidecar.calls) == 1

    asyncio.run(_run())


def test_custom_pipeline_search_planned_reflects_transform_injected_plan() -> None:
    async def _run() -> None:
        events = EventBuffer()
        store = RecordingVectorStore(search_results=[make_search_result(id="hit")])
        pipeline = RetrievalPipeline(
            retrieve=HybridRetrieve(),
            fuse=IdentityFuse(),
            rerank=PassThroughRerank(),
            query_transforms=(_InjectDenseOnlyPlanTransform(),),
        )
        orchestrator = SearchOrchestrator(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            event_sink=events,
            pipeline=pipeline,
        )

        await orchestrator.search(SearchRequest(query="q", corpus_ids=["c"], namespace="n"))

        [planned] = [event for event in events.events if isinstance(event, SearchPlanned)]
        assert planned.channels == ("dense:dense:primary",)
        assert planned.final_limit == 1

    asyncio.run(_run())


def test_non_hybrid_custom_pipeline_search_planned_runs_after_transforms() -> None:
    async def _run() -> None:
        events = EventBuffer()

        class StaticRetrieve:
            async def retrieve(
                self,
                query: PipelineQuery,
                ctx: PipelineContext,
            ) -> list[SearchResult]:
                del query, ctx
                return [make_search_result(id="a"), make_search_result(id="b")]

        pipeline = RetrievalPipeline(
            retrieve=StaticRetrieve(),
            fuse=IdentityFuse(),
            rerank=PassThroughRerank(),
            query_transforms=(_InjectDenseOnlyPlanTransform(),),
        )
        orchestrator = SearchOrchestrator(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(),
            event_sink=events,
            pipeline=pipeline,
        )

        results = await orchestrator.search(
            SearchRequest(query="q", corpus_ids=["c"], namespace="n")
        )

        [planned] = [event for event in events.events if isinstance(event, SearchPlanned)]
        assert planned.channels == ("dense:dense:primary",)
        assert planned.limit == 1
        assert planned.final_limit == 1
        assert [result.id for result in results] == ["a"]

    asyncio.run(_run())


def test_default_pipeline_runs_rerank_when_requested_and_skips_when_not() -> None:
    async def _run() -> None:
        hits = [
            make_search_result(id="a", text="alpha"),
            make_search_result(id="b", text="beta"),
        ]
        reranker = FakeReranker(
            results=[
                RerankResult(index=1, score=0.95, text=hits[1].text),
                RerankResult(index=0, score=0.9, text=hits[0].text),
            ]
        )
        orchestrator = _orchestrator(
            store=RecordingVectorStore(search_results=hits), reranker=reranker
        )

        reranked = await orchestrator.search(
            SearchRequest(
                query="q", corpus_ids=["c"], namespace="n", rerank=True
            )
        )
        assert [r.id for r in reranked] == ["b", "a"]
        assert [r.score for r in reranked] == [0.9, 0.9]

        skipped = await orchestrator.search(
            SearchRequest(
                query="q", corpus_ids=["c"], namespace="n", rerank=False
            )
        )
        assert [r.id for r in skipped] == ["a", "b"]
        assert len(reranker.calls) == 1

    asyncio.run(_run())


def test_default_pipeline_applies_rerank_before_sidecar_merge() -> None:
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
        sidecar = FakeSearchSidecar(
            results=[
                make_search_result(
                    id="side",
                    text="exact",
                    score=1.0,
                    namespace="n",
                    corpus_id="c",
                )
            ]
        )
        orchestrator = _orchestrator(
            store=RecordingVectorStore(search_results=hits),
            reranker=reranker,
            sidecar=sidecar,
        )

        results = await orchestrator.search(
            SearchRequest(query="q", corpus_ids=["c"], namespace="n", rerank=True)
        )

        assert [r.id for r in results] == ["side", "b", "a"]
        assert [r.score for r in results] == [1.0, 0.7, 0.4]
        assert len(sidecar.calls) == 1
        rerank_meta = results[1].metadata["rerank"]
        assert isinstance(rerank_meta, dict)
        assert rerank_meta["search_score"] == 0.7
        assert rerank_meta["provider_score"] == 0.95

    asyncio.run(_run())


def test_default_pipeline_preserves_budgeted_rerank_order_through_sidecar_merge() -> None:
    async def _run() -> None:
        hits = [
            make_search_result(id="a", text="alpha", score=0.95, namespace="n", corpus_id="c"),
            make_search_result(id="b", text="beta", score=0.7, namespace="n", corpus_id="c"),
            make_search_result(id="c", text="gamma", score=0.6, namespace="n", corpus_id="c"),
        ]
        reranker = FakeReranker(
            results=[
                RerankResult(index=1, score=0.99, text=hits[1].text),
                RerankResult(index=0, score=0.98, text=hits[0].text),
            ]
        )
        sidecar = FakeSearchSidecar(
            results=[make_search_result(id="side", score=1.0, namespace="n", corpus_id="c")]
        )
        orchestrator = _orchestrator(
            store=RecordingVectorStore(search_results=hits),
            reranker=reranker,
            sidecar=sidecar,
        )

        results = await orchestrator.search(
            SearchRequest(
                query="q",
                corpus_ids=["c"],
                namespace="n",
                limit=4,
                rerank=True,
                rerank_budget=RerankBudget(candidate_count=2, max_output=2),
            )
        )

        assert [r.id for r in results] == ["side", "b", "a", "c"]

    asyncio.run(_run())


def test_sidecar_hit_can_displace_weaker_vector_hit_when_limit_is_full() -> None:
    async def _run() -> None:
        sidecar = FakeSearchSidecar(
            results=[make_search_result(id="exact", score=1.0, namespace="n", corpus_id="c")]
        )
        store = RecordingVectorStore(
            search_results=[
                make_search_result(id="vec-a", score=0.9, namespace="n", corpus_id="c"),
                make_search_result(id="vec-b", score=0.8, namespace="n", corpus_id="c"),
            ]
        )
        orchestrator = _orchestrator(store=store, sidecar=sidecar)

        results = await orchestrator.search(
            SearchRequest(query="q", corpus_ids=["c"], namespace="n", limit=2)
        )

        assert [r.id for r in results] == ["exact", "vec-a"]

    asyncio.run(_run())


def test_sidecar_prefetch_runs_concurrently_with_vector_search() -> None:
    async def _run() -> None:
        started: dict[str, asyncio.Event] = {
            "sidecar": asyncio.Event(),
            "store": asyncio.Event(),
        }

        class SlowSidecar(FakeSearchSidecar):
            async def search(self, query: SearchSidecarQuery) -> list[SearchResult]:
                started["sidecar"].set()
                await started["store"].wait()
                return await super().search(query)

        class SlowStore(RecordingVectorStore):
            async def search(self, query: SearchQuery) -> list[SearchResult]:
                started["store"].set()
                await started["sidecar"].wait()
                return await super().search(query)

        sidecar = SlowSidecar(
            results=[make_search_result(id="side", score=1.0, namespace="n", corpus_id="c")]
        )
        store = SlowStore(search_results=[make_search_result(id="vec", score=0.5)])
        orchestrator = _orchestrator(store=store, sidecar=sidecar)
        results = await asyncio.wait_for(
            orchestrator.search(
                SearchRequest(query="q", corpus_ids=["c"], namespace="n")
            ),
            timeout=2.0,
        )
        assert {r.id for r in results} == {"side", "vec"}

    asyncio.run(_run())


def test_prefetched_sidecar_is_cancelled_when_retrieve_fails() -> None:
    async def _run() -> None:
        cancelled = asyncio.Event()

        class WaitingSidecar(FakeSearchSidecar):
            async def search(self, query: SearchSidecarQuery) -> list[SearchResult]:
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                return []

        class FailingStore(RecordingVectorStore):
            async def search(self, query: SearchQuery) -> list[SearchResult]:
                raise RuntimeError("retrieve failed")

        orchestrator = _orchestrator(store=FailingStore(), sidecar=WaitingSidecar())
        try:
            await orchestrator.search(SearchRequest(query="q", corpus_ids=["c"], namespace="n"))
        except RuntimeError:
            pass

        assert cancelled.is_set()

    asyncio.run(_run())


def test_search_planned_reflects_prepared_default_query_plan() -> None:
    async def _run() -> None:
        class SparseLaterDisabledStore(RecordingVectorStore):
            capabilities = StoreCapabilities(
                per_point_delete=True,
                document_record_lookup=True,
                query_plan=QueryPlanCapabilities(
                    dense=True,
                    sparse=True,
                    hybrid_rrf=True,
                )
            )

            async def ensure_collection(self) -> None:
                await super().ensure_collection()
                self.capabilities = StoreCapabilities(
                    per_point_delete=True,
                    document_record_lookup=True,
                    query_plan=QueryPlanCapabilities(dense=True)
                )

            async def prepare_query_plan(self, plan: QueryPlan) -> None:
                return None

        events = EventBuffer()
        store = SparseLaterDisabledStore()
        orchestrator = SearchOrchestrator(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            event_sink=events,
        )

        await orchestrator.search(SearchRequest(query="q", corpus_ids=["c"], namespace="n"))

        [planned] = [event for event in events.events if isinstance(event, SearchPlanned)]
        assert store.ensure_collection_calls == 1
        assert planned.channels == ("dense:dense:primary",)
        assert planned.fusion == "none"
        assert store.search_calls[0].query_plan is not None
        assert len(store.search_calls[0].query_plan.prefetches) == 1

    asyncio.run(_run())
