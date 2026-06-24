from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from rag_core.events import EventBuffer, SearchPlanned
from rag_core.search.planning import (
    default_query_plan,
    default_query_plan_for_capabilities,
    query_plan_preset,
)
from rag_core.search.providers.vector_store_capabilities import (
    QDRANT_VECTOR_STORE_CAPABILITY_SPEC,
    TURBOPUFFER_VECTOR_STORE_CAPABILITY_SPEC,
)
from rag_core.search.pipeline import HybridRetrieve, PipelineContext, PipelineQuery
from rag_core.search.query_plan import (
    PRIMARY_DENSE_QUERY_VECTOR,
    DenseChannel,
    Prefetch,
    SparseChannel,
    UnsupportedQueryStage,
)
from rag_core.search.pipeline_runner import SearchPipelineRunner, SearchRequest
from rag_core.search.sparse_channels import (
    PRIMARY_SPARSE_CHANNEL,
    SECONDARY_SPARSE_CHANNEL,
)
from rag_core.search.provider_protocols import (
    QueryPlanCapabilities,
    StoreCapabilities,
)
from rag_core.search.request_models import SearchQuery

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
)

_DENSE_PRIMARY_CHANNEL = f"dense:dense:{PRIMARY_DENSE_QUERY_VECTOR}"


class _DenseOnlyStore(RecordingVectorStore):
    capabilities = StoreCapabilities(
        per_point_delete=True,
        document_record_lookup=True,
        query_plan=QueryPlanCapabilities(dense=True),
    )


class _NoQueryPlanStore(RecordingVectorStore):
    capabilities = StoreCapabilities(
        per_point_delete=True,
        document_record_lookup=True,
    )


class _SparseChannelStrictStore(RecordingVectorStore):
    def default_query_plan(self, *, result_limit: int):
        return default_query_plan(
            result_limit=result_limit,
            sparse_channels=("splade",),
        )

    async def search(self, query: SearchQuery):
        if query.query_plan is not None:
            available = query.all_sparse_vectors()
            for prefetch in query.query_plan.prefetches:
                _assert_sparse_query_vectors_present(prefetch, available)
        return await super().search(query)


class _MultiSparseChannelStrictStore(_SparseChannelStrictStore):
    def default_query_plan(self, *, result_limit: int):
        return default_query_plan(
            result_limit=result_limit,
            sparse_channels=(PRIMARY_SPARSE_CHANNEL, SECONDARY_SPARSE_CHANNEL),
        )


def _assert_sparse_query_vectors_present(
    prefetch: Prefetch, available: Mapping[str, object]
) -> None:
    channel = prefetch.channel
    if (
        isinstance(channel, SparseChannel)
        and channel.using_query_vector not in available
    ):
        raise UnsupportedQueryStage(
            f"SparseChannel({channel.using_query_vector!r}) has no matching sparse query vector"
        )
    for nested in prefetch.nested:
        _assert_sparse_query_vectors_present(nested, available)


def test_default_query_plan_prefers_hybrid_when_declared() -> None:
    plan = default_query_plan_for_capabilities(
        capabilities=QDRANT_VECTOR_STORE_CAPABILITY_SPEC.query_plan,
        result_limit=5,
    )

    assert plan is not None
    assert len(plan.prefetches) == 2
    assert isinstance(plan.prefetches[0].channel, DenseChannel)
    assert isinstance(plan.prefetches[1].channel, SparseChannel)
    assert plan.fuse is not None
    assert plan.fuse.kind == "rrf"


def test_default_query_plan_uses_hybrid_when_turbopuffer_declares_bm25_hybrid() -> None:
    plan = default_query_plan_for_capabilities(
        capabilities=TURBOPUFFER_VECTOR_STORE_CAPABILITY_SPEC.query_plan,
        result_limit=5,
    )

    assert plan is not None
    assert len(plan.prefetches) == 2
    assert isinstance(plan.prefetches[0].channel, DenseChannel)
    assert isinstance(plan.prefetches[1].channel, SparseChannel)
    assert plan.fuse is not None
    assert plan.fuse.kind == "rrf"


def test_default_query_plan_uses_sparse_only_when_only_sparse_is_declared() -> None:
    plan = default_query_plan_for_capabilities(
        capabilities=QueryPlanCapabilities(sparse=True),
        result_limit=5,
    )

    assert plan is not None
    assert len(plan.prefetches) == 1
    assert isinstance(plan.prefetches[0].channel, SparseChannel)
    assert plan.fuse is None


def test_default_query_plan_uses_weighted_rrf_when_it_is_the_only_hybrid() -> None:
    plan = default_query_plan_for_capabilities(
        capabilities=QueryPlanCapabilities(
            dense=True,
            sparse=True,
            hybrid_weighted_rrf=True,
        ),
        result_limit=5,
    )

    assert plan is not None
    assert plan.fuse is not None
    assert plan.fuse.kind == "weighted_rrf"
    assert plan.fuse.weights == (1.0, 1.0)


def test_default_query_plan_prefers_documented_balanced_profile_before_weighted_rrf() -> (
    None
):
    plan = default_query_plan_for_capabilities(
        capabilities=QueryPlanCapabilities(
            dense=True,
            sparse=True,
            hybrid_rrf=True,
            hybrid_weighted_rrf=True,
        ),
        result_limit=5,
    )

    assert plan is not None
    assert plan.fuse is not None
    assert plan.fuse.kind == "rrf"


def test_default_query_plan_is_absent_for_baseline_store_capability() -> None:
    assert (
        default_query_plan_for_capabilities(
            capabilities=QueryPlanCapabilities(),
            result_limit=5,
        )
        is None
    )


def test_search_default_uses_capability_aware_dense_plan() -> None:
    async def scenario() -> _DenseOnlyStore:
        store = _DenseOnlyStore()
        pipeline_runner = _pipeline_runner(store)
        await pipeline_runner.search(
            SearchRequest(query="billing", collections=["docs"], namespace="acme")
        )
        return store

    store = asyncio.run(scenario())
    [call] = store.search_calls
    assert call.query_plan is not None
    assert len(call.query_plan.prefetches) == 1
    assert isinstance(call.query_plan.prefetches[0].channel, DenseChannel)


def test_search_default_leaves_baseline_store_query_plan_unset() -> None:
    async def scenario() -> tuple[_NoQueryPlanStore, SearchPlanned]:
        store = _NoQueryPlanStore()
        events = EventBuffer()
        pipeline_runner = _pipeline_runner(store, event_sink=events)
        await pipeline_runner.search(
            SearchRequest(query="billing", collections=["docs"], namespace="acme")
        )
        [planned] = [
            event for event in events.events if isinstance(event, SearchPlanned)
        ]
        return store, planned

    store, planned = asyncio.run(scenario())
    [call] = store.search_calls
    assert call.query_plan is None
    assert planned.channels == ()
    assert planned.fusion == "store_default"


def test_explicit_query_plan_rejects_baseline_store_before_embedding() -> None:
    explicit_plan = query_plan_preset("hybrid_dbsf", limit=5)
    embedding = FakeEmbeddingProvider()
    sparse = FakeSparseEmbedder()
    retrieve = HybridRetrieve()

    async def scenario() -> None:
        store = _NoQueryPlanStore()
        with pytest.raises(UnsupportedQueryStage, match="does not declare"):
            await retrieve.retrieve(
                PipelineQuery(
                    query="billing",
                    namespace="acme",
                    collections=["docs"],
                    query_plan=explicit_plan,
                ),
                PipelineContext(
                    embedding_provider=embedding,
                    sparse_embedder=sparse,
                    vector_store=store,
                ),
            )

    asyncio.run(scenario())
    assert embedding.embed_query_calls == []
    assert sparse.embed_query_multi_calls == []


def test_sparse_only_plan_skips_dense_embedding() -> None:
    embedding = FakeEmbeddingProvider()
    sparse = FakeSparseEmbedder()
    retrieve = HybridRetrieve()

    async def scenario() -> RecordingVectorStore:
        store = RecordingVectorStore()
        await retrieve.retrieve(
            PipelineQuery(
                query="billing",
                namespace="acme",
                collections=["docs"],
                query_plan=query_plan_preset("sparse_only", limit=5),
            ),
            PipelineContext(
                embedding_provider=embedding,
                sparse_embedder=sparse,
                vector_store=store,
            ),
        )
        return store

    store = asyncio.run(scenario())
    assert embedding.embed_query_calls == []
    assert sparse.embed_query_multi_calls == ["billing"]
    [call] = store.search_calls
    assert call.dense_vector == []


def test_implicit_default_plan_downgrades_when_sparse_query_channel_is_missing() -> (
    None
):
    embedding = FakeEmbeddingProvider()
    sparse = FakeSparseEmbedder(include_extra_channel=False)
    retrieve = HybridRetrieve()

    async def scenario() -> _SparseChannelStrictStore:
        store = _SparseChannelStrictStore()
        await retrieve.retrieve(
            PipelineQuery(
                query="billing",
                namespace="acme",
                collections=["docs"],
            ),
            PipelineContext(
                embedding_provider=embedding,
                sparse_embedder=sparse,
                vector_store=store,
            ),
        )
        return store

    store = asyncio.run(scenario())
    [call] = store.search_calls
    assert call.query_plan is not None
    assert len(call.query_plan.prefetches) == 1
    assert isinstance(call.query_plan.prefetches[0].channel, DenseChannel)


def test_implicit_default_plan_preserves_available_sparse_channels() -> None:
    embedding = FakeEmbeddingProvider()
    sparse = FakeSparseEmbedder(include_extra_channel=False)
    retrieve = HybridRetrieve()

    async def scenario() -> _MultiSparseChannelStrictStore:
        store = _MultiSparseChannelStrictStore()
        await retrieve.retrieve(
            PipelineQuery(
                query="billing",
                namespace="acme",
                collections=["docs"],
            ),
            PipelineContext(
                embedding_provider=embedding,
                sparse_embedder=sparse,
                vector_store=store,
            ),
        )
        return store

    store = asyncio.run(scenario())
    [call] = store.search_calls
    assert call.query_plan is not None
    assert len(call.query_plan.prefetches) == 2
    dense, sparse_prefetch = call.query_plan.prefetches
    assert isinstance(dense.channel, DenseChannel)
    assert isinstance(sparse_prefetch.channel, SparseChannel)
    assert sparse_prefetch.channel.using_query_vector == PRIMARY_SPARSE_CHANNEL


def test_search_planned_reflects_resolved_sparse_channels() -> None:
    async def scenario() -> tuple[_MultiSparseChannelStrictStore, SearchPlanned]:
        events = EventBuffer()
        store = _MultiSparseChannelStrictStore()
        pipeline_runner = SearchPipelineRunner(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(include_extra_channel=False),
            vector_store=store,
            event_sink=events,
        )
        await pipeline_runner.search(
            SearchRequest(query="billing", collections=["docs"], namespace="acme")
        )
        [planned] = [
            event for event in events.events if isinstance(event, SearchPlanned)
        ]
        return store, planned

    store, planned = asyncio.run(scenario())
    [call] = store.search_calls
    assert call.query_plan is not None
    assert planned.channels == (
        _DENSE_PRIMARY_CHANNEL,
        f"sparse:{PRIMARY_SPARSE_CHANNEL}:{PRIMARY_SPARSE_CHANNEL}",
    )


def test_declared_store_rejects_unsupported_explicit_plan_before_embedding() -> None:
    embedding = FakeEmbeddingProvider()
    sparse = FakeSparseEmbedder()
    retrieve = HybridRetrieve()

    async def scenario() -> None:
        with pytest.raises(UnsupportedQueryStage, match="DBSF"):
            await retrieve.retrieve(
                PipelineQuery(
                    query="billing",
                    namespace="acme",
                    collections=["docs"],
                    query_plan=query_plan_preset("hybrid_dbsf", limit=5),
                ),
                PipelineContext(
                    embedding_provider=embedding,
                    sparse_embedder=sparse,
                    vector_store=_DenseOnlyStore(),
                ),
            )

    asyncio.run(scenario())
    assert embedding.embed_query_calls == []
    assert sparse.embed_query_multi_calls == []


def _pipeline_runner(
    store: RecordingVectorStore,
    *,
    event_sink: EventBuffer | None = None,
) -> SearchPipelineRunner:
    return SearchPipelineRunner(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
        event_sink=event_sink,
    )
