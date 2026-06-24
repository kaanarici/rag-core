"""Tests for experimental query expansion pipeline stages."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import uuid

import pytest

from rag_core.demo import build_demo_core
from rag_core.events import EventBuffer, SearchStageCompleted
from rag_core.search.pipeline import (
    AnthropicQueryVariantGenerator,
    HybridRetrieve,
    HydeTransform,
    IdentityFuse,
    MultiQueryTransform,
    PassThroughRerank,
    PipelineContext,
    PipelineQuery,
    RetrievalPipeline,
    RrfFuse,
)
from rag_core.search.pipeline_runner import SearchPipelineRunner, SearchRequest
from rag_core.search.vector_models import SearchResult
from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
)

pytestmark = [pytest.mark.plumbing]


def _build_query(**overrides: object) -> PipelineQuery:
    base: dict[str, object] = {
        "query": "original query",
        "namespace": "space-1",
        "collections": ["corpus-1"],
        "limit": 20,
    }
    base.update(overrides)
    return PipelineQuery(**base)  # type: ignore[arg-type]


def _build_context(
    *,
    embedding: FakeEmbeddingProvider | None = None,
    sparse: FakeSparseEmbedder | None = None,
    store: RecordingVectorStore | None = None,
    event_sink: EventBuffer | None = None,
) -> PipelineContext:
    return PipelineContext(
        embedding_provider=embedding or FakeEmbeddingProvider(),
        sparse_embedder=sparse or FakeSparseEmbedder(),
        vector_store=store or RecordingVectorStore(),
        event_sink=event_sink,
    )


class _StaticGenerator:
    def __init__(
        self,
        *,
        variants: Sequence[str] = (),
        passage: str = "",
        error: Exception | None = None,
    ) -> None:
        self._variants = tuple(variants)
        self._passage = passage
        self._error = error
        self.variant_calls: list[tuple[str, int]] = []
        self.hyde_calls: list[str] = []

    async def generate_variants(self, query: str, *, count: int) -> Sequence[str]:
        self.variant_calls.append((query, count))
        if self._error is not None:
            raise self._error
        return self._variants[:count]

    async def generate_hypothetical_passage(self, query: str) -> str:
        self.hyde_calls.append(query)
        if self._error is not None:
            raise self._error
        return self._passage


class _RecordingRetrieve:
    def __init__(self, *, delay_seconds: float = 0.0) -> None:
        self.calls: list[tuple[str, str | None, list[float] | None, object]] = []
        self.in_flight = 0
        self.high_water = 0
        self._delay_seconds = delay_seconds

    async def retrieve(
        self, query: PipelineQuery, ctx: PipelineContext
    ) -> list[SearchResult]:
        del ctx
        self.calls.append(
            (
                query.query,
                query.dense_query_text,
                query.query_vector,
                query.query_sparse_vectors,
            )
        )
        self.in_flight += 1
        self.high_water = max(self.high_water, self.in_flight)
        try:
            if self._delay_seconds:
                await asyncio.sleep(self._delay_seconds)
            return [
                make_search_result(
                    id=f"{query.query}-hit",
                    text=query.query,
                    document_id=query.query,
                )
            ]
        finally:
            self.in_flight -= 1


class _RecordingRerank:
    real_rerank = True

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def rerank(
        self,
        results: list[SearchResult],
        query: PipelineQuery,
        ctx: PipelineContext,
    ) -> list[SearchResult]:
        del query, ctx
        self.calls.append([result.id for result in results])
        return results


class _FakeAnthropicBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeAnthropicBlock(text)]


class _FakeAnthropicMessages:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> _FakeAnthropicResponse:
        self.calls.append(dict(kwargs))
        return _FakeAnthropicResponse(self._responses.pop(0))


class _FakeAnthropicClient:
    def __init__(self, responses: Sequence[str]) -> None:
        self.messages = _FakeAnthropicMessages(responses)


def test_anthropic_query_variant_generator_parses_tolerant_json() -> None:
    async def _run() -> None:
        client = _FakeAnthropicClient(
            (
                'Here is JSON: {"queries": ["pay invoice", "card billing"]}',
                '```json\n{"passage": "Invoices are paid by card or ACH."}\n```',
            )
        )
        generator = AnthropicQueryVariantGenerator(client=client)

        variants = await generator.generate_variants("How do I pay?", count=2)
        passage = await generator.generate_hypothetical_passage("How do I pay?")

        assert variants == ["pay invoice", "card billing"]
        assert passage == "Invoices are paid by card or ACH."
        assert client.messages.calls[0]["temperature"] == 0
        assert client.messages.calls[1]["temperature"] == 0

    asyncio.run(_run())


def test_multi_query_fanout_preserves_order_and_clears_variant_vectors() -> None:
    async def _run() -> None:
        retrieve = _RecordingRetrieve(delay_seconds=0.01)
        pipeline = RetrievalPipeline(
            retrieve=retrieve,
            fuse=RrfFuse(),
            rerank=PassThroughRerank(),
        )
        results = await pipeline.run(
            _build_query(
                query="original",
                query_variants=("variant-b", "variant-c"),
                dense_query_text="hyde passage",
                query_vector=[1.0, 2.0],
            ),
            _build_context(),
        )

        assert [call[0] for call in retrieve.calls] == [
            "original",
            "variant-b",
            "variant-c",
        ]
        assert retrieve.calls[0][1] == "hyde passage"
        assert retrieve.calls[0][2] == [1.0, 2.0]
        assert retrieve.calls[1][1:] == (None, None, None)
        assert retrieve.calls[2][1:] == (None, None, None)
        assert [result.id for result in results] == [
            "original-hit",
            "variant-b-hit",
            "variant-c-hit",
        ]

    asyncio.run(_run())


def test_multi_query_retrieve_fanout_is_bounded_to_four_in_flight() -> None:
    async def _run() -> None:
        retrieve = _RecordingRetrieve(delay_seconds=0.01)
        pipeline = RetrievalPipeline(
            retrieve=retrieve,
            fuse=RrfFuse(),
            rerank=PassThroughRerank(),
        )
        await pipeline.run(
            _build_query(
                query_variants=tuple(f"variant-{index}" for index in range(8))
            ),
            _build_context(),
        )

        assert len(retrieve.calls) == 9
        assert retrieve.high_water <= 4

    asyncio.run(_run())


def test_multi_query_generator_failure_falls_back_to_original_with_event() -> None:
    async def _run() -> None:
        events = EventBuffer()
        retrieve = _RecordingRetrieve()
        generator = _StaticGenerator(error=RuntimeError("provider unavailable"))
        pipeline = RetrievalPipeline(
            retrieve=retrieve,
            fuse=IdentityFuse(),
            rerank=PassThroughRerank(),
            query_transforms=(MultiQueryTransform(generator),),
        )
        results = await pipeline.run(
            _build_query(query="original"),
            _build_context(event_sink=events),
        )

        assert [call[0] for call in retrieve.calls] == ["original"]
        assert [result.id for result in results] == ["original-hit"]
        [fallback] = [
            event
            for event in events.events
            if isinstance(event, SearchStageCompleted)
            and event.stage_name == "MultiQueryTransform.fallback"
        ]
        assert fallback.stage == "query_transform"
        assert fallback.dropped_count == 1

    asyncio.run(_run())


def test_multi_query_generator_failure_preserves_upstream_variants() -> None:
    async def _run() -> None:
        retrieve = _RecordingRetrieve()
        pipeline = RetrievalPipeline(
            retrieve=retrieve,
            fuse=IdentityFuse(),
            rerank=PassThroughRerank(),
            query_transforms=(
                MultiQueryTransform(_StaticGenerator(variants=("prior variant",))),
                MultiQueryTransform(
                    _StaticGenerator(error=RuntimeError("provider unavailable"))
                ),
            ),
        )

        await pipeline.run(
            _build_query(query="original"),
            _build_context(),
        )

        assert [call[0] for call in retrieve.calls] == ["original", "prior variant"]

    asyncio.run(_run())


def test_hyde_generator_failure_preserves_upstream_dense_override() -> None:
    async def _run() -> None:
        query = _build_query(
            dense_query_text="prior dense passage",
            query_vector=[0.5, 0.25],
        )
        transformed = await HydeTransform(
            _StaticGenerator(error=RuntimeError("provider unavailable"))
        ).transform(query, _build_context())

        assert transformed.dense_query_text == "prior dense passage"
        assert transformed.query_vector == [0.5, 0.25]

    asyncio.run(_run())


def test_rrf_fuse_deduplicates_and_scores_by_reciprocal_rank() -> None:
    async def _run() -> None:
        results = await RrfFuse(k=60).fuse(
            [
                [make_search_result(id="a"), make_search_result(id="b")],
                [make_search_result(id="b"), make_search_result(id="c")],
                [make_search_result(id="a"), make_search_result(id="c")],
            ],
            _build_query(),
            _build_context(),
        )

        assert [result.id for result in results] == ["a", "b", "c"]
        assert results[0].score == pytest.approx((1 / 61) + (1 / 61))
        assert results[1].score == pytest.approx((1 / 62) + (1 / 61))
        assert results[2].score == pytest.approx((1 / 62) + (1 / 62))

    asyncio.run(_run())


def test_rrf_fuse_counts_duplicate_id_once_per_ranked_list() -> None:
    async def _run() -> None:
        results = await RrfFuse(k=60).fuse(
            [
                [
                    make_search_result(id="a", score=0.9),
                    make_search_result(id="a", score=0.8),
                ],
                [make_search_result(id="b", score=0.7)],
            ],
            _build_query(),
            _build_context(),
        )

        assert [result.id for result in results] == ["a", "b"]
        assert results[0].score == pytest.approx(1 / 61)
        assert results[1].score == pytest.approx(1 / 61)

    asyncio.run(_run())


def test_hyde_transform_overrides_precomputed_dense_vector_and_keeps_sparse_original() -> None:
    async def _run() -> None:
        embedding = FakeEmbeddingProvider(
            vocabulary=("original", "query", "hypothetical", "answer")
        )
        sparse = FakeSparseEmbedder()
        store = RecordingVectorStore(search_results=[make_search_result()])
        pipeline = RetrievalPipeline(
            retrieve=HybridRetrieve(),
            fuse=IdentityFuse(),
            rerank=PassThroughRerank(),
            query_transforms=(
                HydeTransform(
                    _StaticGenerator(passage="hypothetical answer passage")
                ),
            ),
        )

        await pipeline.run(
            _build_query(query="original query", query_vector=[99.0, 99.0]),
            _build_context(embedding=embedding, sparse=sparse, store=store),
        )

        assert embedding.embed_query_calls == ["hypothetical answer passage"]
        assert sparse.embed_query_multi_calls == ["original query"]
        assert store.search_calls[0].lexical_query == "original query"

    asyncio.run(_run())


def test_multi_query_fanout_feeds_rerank_pool_not_final_limit() -> None:
    # The fused multi-query candidates are bounded by the rerank candidate pool
    # (>= final limit), not truncated to the final limit before reranking, so
    # the reranker can still promote a document ranked below the final limit.
    async def _run() -> None:
        rerank = _RecordingRerank()
        pipeline = RetrievalPipeline(
            retrieve=_RecordingRetrieve(),
            fuse=RrfFuse(),
            rerank=rerank,
        )

        await pipeline.run(
            _build_query(
                query="original",
                query_variants=("variant-a", "variant-b"),
                limit=2,
                rerank=True,
            ),
            _build_context(),
        )

        # Final limit is 2, but all three fused hits reach the reranker because
        # the pool (>= 2) bounds the rerank input, not the final limit.
        assert rerank.calls == [["original-hit", "variant-a-hit", "variant-b-hit"]]

    asyncio.run(_run())


@pytest.mark.integration
def test_query_expansion_rrf_runs_against_demo_qdrant() -> None:
    async def _run() -> None:
        async with build_demo_core(
            store_collection=f"query_expansion_{uuid.uuid4().hex}"
        ) as core:
            await core.add_bytes(
                file_bytes=b"Invoices can be paid by ACH transfer or credit card.",
                filename="billing.md",
                mime_type="text/markdown",
                namespace="integration",
                collection="docs",
                document_id="billing",
                document_key="billing.md",
            )
            await core.add_bytes(
                file_bytes=b"International shipping requires customs forms.",
                filename="shipping.md",
                mime_type="text/markdown",
                namespace="integration",
                collection="docs",
                document_id="shipping",
                document_key="shipping.md",
            )
            pipeline = RetrievalPipeline(
                retrieve=HybridRetrieve(),
                fuse=RrfFuse(),
                rerank=PassThroughRerank(),
                query_transforms=(
                    MultiQueryTransform(
                        _StaticGenerator(
                            variants=(
                                "pay invoices with ACH credit card",
                                "invoice payment methods",
                            )
                        )
                    ),
                ),
            )
            runner = SearchPipelineRunner(
                embedding_provider=core._embedding,
                sparse_embedder=core._sparse,
                vector_store=core._store,
                pipeline=pipeline,
            )

            hits = await runner.search(
                SearchRequest(
                    query="settle customer bills",
                    namespace="integration",
                    collections=["docs"],
                    limit=5,
                )
            )

            result_ids = [hit.id for hit in hits]
            assert len(result_ids) == len(set(result_ids))
            assert "billing" in [hit.document_id for hit in hits[:3]]

    asyncio.run(_run())
