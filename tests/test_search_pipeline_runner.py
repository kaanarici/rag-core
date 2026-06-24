"""SearchPipelineRunner end-to-end contract: vectors, sidecar, rerank, filters."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from rag_core.search.lexical_sidecar import LexicalSidecarRecord, PortableLexicalSidecar
from rag_core.search.planning import query_plan_preset
from rag_core.search.pipeline_runner import (
    SearchExecutionOptions,
    SearchPipelineRunner,
    SearchRequest,
)
from rag_core.search.filters import Term
from rag_core.search.request_models import (
    RerankResult,
    SearchSidecarQuery,
)
from rag_core.search.vector_models import SparseVector

from tests.support import (
    FakeEmbeddingProvider,
    FakeReranker,
    FakeSearchSidecar,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
)


def _pipeline_runner(
    *,
    store: RecordingVectorStore | None = None,
    sparse: FakeSparseEmbedder | None = None,
    embedding: FakeEmbeddingProvider | None = None,
    sidecar: object | None = None,
    reranker: FakeReranker | None = None,
) -> SearchPipelineRunner:
    return SearchPipelineRunner(
        embedding_provider=embedding or FakeEmbeddingProvider(),
        sparse_embedder=sparse or FakeSparseEmbedder(),
        vector_store=store or RecordingVectorStore(),
        sidecar=sidecar,  # type: ignore[arg-type]
        reranker=reranker,
    )


def test_search_uses_precomputed_query_vectors_without_re_embedding() -> None:
    async def _run() -> None:
        embedding = FakeEmbeddingProvider()
        sparse = FakeSparseEmbedder()
        store = RecordingVectorStore(search_results=[make_search_result()])
        pipeline_runner = _pipeline_runner(store=store, sparse=sparse, embedding=embedding)
        dense = [9.0, 8.0, 7.0, 6.0]
        sparse_vectors = {"bm25": SparseVector(indices=[1, 2], values=[1.0, 2.0])}

        results = await pipeline_runner.search(
            SearchRequest(
                query="unused",
                collections=["corpus-1"],
                namespace="space-1",
                execution=SearchExecutionOptions(
                    query_vector=dense,
                    query_sparse_vectors=sparse_vectors,
                ),
            )
        )

        assert len(results) == 1
        assert embedding.embed_query_calls == []
        assert sparse.embed_query_multi_calls == []
        assert store.search_calls[0].dense_vector == dense
        assert store.search_calls[0].sparse_vector == sparse_vectors["bm25"]

    asyncio.run(_run())


def test_search_request_keeps_execution_options_outside_retrieval_intent() -> None:
    from rag_core.search.planning import default_query_plan

    plan = default_query_plan(result_limit=5, fusion="dbsf")
    req = SearchRequest(
        query="query",
        collections=["corpus-1"],
        namespace="space-1",
        limit=5,
        execution=SearchExecutionOptions(query_plan=plan),
    )

    assert req.execution.query_plan is plan
    assert req.execution.use_lexical_search is True


def test_search_execution_options_validates_lexical_flag_type() -> None:
    with pytest.raises(
        ValueError,
        match="SearchExecutionOptions.use_lexical_search must be a boolean",
    ):
        SearchExecutionOptions(use_lexical_search=cast(bool, "yes"))


def test_search_execution_options_validates_precomputed_dense_vector() -> None:
    with pytest.raises(
        ValueError,
        match="SearchExecutionOptions.query_vector must contain finite numbers",
    ):
        SearchExecutionOptions(query_vector=cast(list[float], [1.0, float("nan")]))


def test_search_execution_options_validates_sparse_vector_channels() -> None:
    error = (
        "SearchExecutionOptions.query_sparse_vectors must map non-empty "
        "channel names to SparseVector values"
    )

    with pytest.raises(ValueError, match=error):
        SearchExecutionOptions(
            query_sparse_vectors=cast(dict[str, SparseVector], {"": SparseVector([], [])})
        )

    with pytest.raises(ValueError, match=error):
        SearchExecutionOptions(
            query_sparse_vectors=cast(dict[str, SparseVector], {"bm25": object()})
        )


def test_empty_document_scope_returns_no_results_without_provider_search() -> None:
    async def _run() -> RecordingVectorStore:
        store = RecordingVectorStore(search_results=[make_search_result(id="leaked")])
        embedding = FakeEmbeddingProvider()
        sparse = FakeSparseEmbedder()
        pipeline_runner = _pipeline_runner(store=store, embedding=embedding, sparse=sparse)

        results = await pipeline_runner.search(
            SearchRequest(
                query="query",
                collections=["corpus-1"],
                namespace="space-1",
                document_ids=[],
            )
        )

        assert results == []
        assert embedding.embed_query_calls == []
        assert sparse.embed_query_multi_calls == []
        return store

    store = asyncio.run(_run())
    assert store.search_calls == []


def test_empty_allowlist_scope_returns_no_results_without_provider_search(
) -> None:
    async def _run(request: SearchRequest) -> RecordingVectorStore:
        store = RecordingVectorStore(search_results=[make_search_result(id="leaked")])
        embedding = FakeEmbeddingProvider()
        sparse = FakeSparseEmbedder()
        pipeline_runner = _pipeline_runner(store=store, embedding=embedding, sparse=sparse)

        results = await pipeline_runner.search(request)

        assert results == []
        assert embedding.embed_query_calls == []
        assert sparse.embed_query_multi_calls == []
        return store

    for request in (
        SearchRequest(query="query", collections=[], namespace="space-1"),
        SearchRequest(
            query="query",
            collections=["corpus-1"],
            namespace="space-1",
            content_types=[],
        ),
    ):
        store = asyncio.run(_run(request))
        assert store.search_calls == []


def test_search_uses_first_sparse_channel_when_bm25_missing() -> None:
    async def _run() -> None:
        store = RecordingVectorStore()
        pipeline_runner = _pipeline_runner(store=store)
        splade = SparseVector(indices=[9], values=[3.0])

        await pipeline_runner.search(
            SearchRequest(
                query="query",
                collections=["corpus-1"],
                namespace="space-1",
                execution=SearchExecutionOptions(
                    query_vector=[1.0, 2.0, 3.0, 4.0],
                    query_sparse_vectors={"splade": splade},
                ),
            )
        )

        assert store.search_calls[0].sparse_vector == splade

    asyncio.run(_run())


def test_search_downgrades_when_no_sparse_vector_is_available() -> None:
    async def _run() -> None:
        store = RecordingVectorStore(search_results=[make_search_result()])
        pipeline_runner = _pipeline_runner(
            store=store,
            sparse=FakeSparseEmbedder(empty_query_multi=True),
        )

        await pipeline_runner.search(
            SearchRequest(
                query="query",
                collections=["corpus-1"],
                namespace="space-1",
                execution=SearchExecutionOptions(query_vector=[1.0, 2.0, 3.0, 4.0]),
            )
        )

        assert store.search_calls[0].query_plan is not None
        assert store.search_calls[0].query_plan.search_profile is None
        assert store.search_calls[0].sparse_vector.indices == []

    asyncio.run(_run())


def test_dense_only_plan_does_not_require_sparse_query_vector() -> None:
    async def _run() -> None:
        store = RecordingVectorStore(search_results=[make_search_result()])
        sparse = FakeSparseEmbedder(empty_query_multi=True)
        pipeline_runner = _pipeline_runner(store=store, sparse=sparse)

        await pipeline_runner.search(
            SearchRequest(
                query="query",
                collections=["corpus-1"],
                namespace="space-1",
                execution=SearchExecutionOptions(
                    query_vector=[1.0, 2.0, 3.0, 4.0],
                    query_plan=query_plan_preset("dense_only", limit=5),
                ),
            )
        )

        assert store.search_calls[0].sparse_vector.indices == []
        assert sparse.embed_query_multi_calls == []

    asyncio.run(_run())


def test_sparse_only_plan_discards_unused_precomputed_dense_vector() -> None:
    async def _run() -> None:
        store = RecordingVectorStore(search_results=[make_search_result()])
        embedding = FakeEmbeddingProvider()
        sparse_vectors = {"bm25": SparseVector(indices=[2], values=[2.0])}
        pipeline_runner = _pipeline_runner(store=store, embedding=embedding)

        await pipeline_runner.search(
            SearchRequest(
                query="query",
                collections=["corpus-1"],
                namespace="space-1",
                execution=SearchExecutionOptions(
                    query_vector=[1.0],
                    query_sparse_vectors=sparse_vectors,
                    query_plan=query_plan_preset("sparse_only", limit=5),
                ),
            )
        )

        assert store.search_calls[0].dense_vector == []
        assert store.search_calls[0].sparse_vector == sparse_vectors["bm25"]
        assert embedding.embed_query_calls == []

    asyncio.run(_run())


def test_search_reranks_results_and_falls_back_on_error() -> None:
    async def _run() -> None:
        doc_a = make_search_result(id="doc-a", text="fox alpha", score=0.7)
        doc_b = make_search_result(id="doc-b", text="query beta", score=0.2)
        reranker = FakeReranker(
            results=[
                RerankResult(index=1, score=0.95, text=doc_b.text),
                RerankResult(index=0, score=0.90, text=doc_a.text),
            ]
        )
        pipeline_runner = _pipeline_runner(
            store=RecordingVectorStore(search_results=[doc_a, doc_b]), reranker=reranker
        )

        results = await pipeline_runner.search(
            SearchRequest(
                query="query", collections=["corpus-1"], namespace="space-1", rerank=True
            )
        )
        assert [r.id for r in results] == ["doc-b", "doc-a"]
        assert [r.score for r in results] == [0.2, 0.7]
        rerank_metadata = [
            cast(dict[str, object], result.metadata["rerank"]) for result in results
        ]
        assert [metadata["provider_score"] for metadata in rerank_metadata] == [
            0.95,
            0.90,
        ]
        assert [metadata["search_score"] for metadata in rerank_metadata] == [0.2, 0.7]
        assert reranker.calls == [("query", [doc_a.text, doc_b.text], 2)]

        failing = _pipeline_runner(
            store=RecordingVectorStore(search_results=[doc_a]),
            reranker=FakeReranker(error=RuntimeError("rerank failed")),
        )
        fallback = await failing.search(
            SearchRequest(
                query="query", collections=["corpus-1"], namespace="space-1", rerank=True
            )
        )
        assert [r.id for r in fallback] == ["doc-a"]

    asyncio.run(_run())


def test_search_merges_sidecar_results_before_vector_results() -> None:
    async def _run() -> None:
        semantic = make_search_result(id="doc-semantic", text="fox context")
        exact = make_search_result(
            id="doc-exact",
            text="fox query text",
            title="Fox Query",
            score=1.0,
        )
        sidecar = FakeSearchSidecar(results=[exact])
        pipeline_runner = _pipeline_runner(
            store=RecordingVectorStore(search_results=[semantic]), sidecar=sidecar
        )

        results = await pipeline_runner.search(
            SearchRequest(
                query="fox query", collections=["corpus-1"], namespace="space-1"
            )
        )

        assert [r.id for r in results] == ["doc-exact", "doc-semantic"]
        assert sidecar.calls[0].namespace == "space-1"
        assert sidecar.calls[0].collections == ["corpus-1"]

    asyncio.run(_run())


def test_search_dedupes_sidecar_and_vector_results_by_id_keeping_richer_fields() -> None:
    """Same id from both sources collapses; observable fields fall back to the richer side."""

    async def _run() -> None:
        vector_hit = make_search_result(
            id="doc-1",
            text="semantic hit",
            score=0.6,
            document_key="/docs/guide.txt",
            content_sha256="hash-1",
            section_title="Overview",
        )
        sidecar_hit = make_search_result(id="doc-1", text="exact hit", score=0.0)
        pipeline_runner = _pipeline_runner(
            store=RecordingVectorStore(search_results=[vector_hit]),
            sidecar=FakeSearchSidecar(results=[sidecar_hit]),
        )

        results = await pipeline_runner.search(
            SearchRequest(
                query="fox query", collections=["corpus-1"], namespace="space-1"
            )
        )

        assert [r.id for r in results] == ["doc-1"]
        merged = results[0]
        assert merged.text == "exact hit"
        assert merged.score == 0.6
        assert merged.document_key == "/docs/guide.txt"
        assert merged.content_sha256 == "hash-1"
        assert merged.section_title == "Overview"

    asyncio.run(_run())


def test_search_can_disable_sidecar_per_request() -> None:
    async def _run() -> None:
        sidecar = FakeSearchSidecar(results=[make_search_result(id="doc-exact", score=1.0)])
        pipeline_runner = _pipeline_runner(
            store=RecordingVectorStore(
                search_results=[make_search_result(id="doc-store")]
            ),
            sidecar=sidecar,
        )

        results = await pipeline_runner.search(
            SearchRequest(
                query="fox query",
                collections=["corpus-1"],
                namespace="space-1",
                execution=SearchExecutionOptions(use_lexical_search=False),
            )
        )

        assert [r.id for r in results] == ["doc-store"]
        assert sidecar.calls == []

    asyncio.run(_run())


def test_search_propagates_scope_and_metadata_filters_to_sidecar() -> None:
    """Scope (collection/content_type/document_ids) and metadata_filter both reach the sidecar
    AND filter out sidecar results that ignored the scope."""

    async def _run() -> None:
        sidecar = FakeSearchSidecar(
            results=[
                make_search_result(
                    id="doc-support",
                    document_id="doc-allowed",
                    metadata={"team": "support"},
                ),
                make_search_result(
                    id="doc-sales",
                    document_id="doc-allowed",
                    metadata={"team": "sales"},
                ),
                make_search_result(
                    id="doc-other-corpus",
                    document_id="doc-allowed",
                    collection="other",
                    metadata={"team": "support"},
                ),
                make_search_result(
                    id="doc-other-type",
                    document_id="doc-allowed",
                    content_type="code",
                    metadata={"team": "support"},
                ),
                make_search_result(
                    id="doc-other-document",
                    document_id="doc-other",
                    metadata={"team": "support"},
                ),
                make_search_result(
                    id="doc-other-namespace",
                    namespace="space-2",
                    document_id="doc-allowed",
                    metadata={"team": "support"},
                ),
            ]
        )
        pipeline_runner = _pipeline_runner(
            store=RecordingVectorStore(
                search_results=[make_search_result(id="doc-vector", document_id="doc-allowed")]
            ),
            sidecar=sidecar,
        )
        metadata_filter = Term(field="team", value="support")

        results = await pipeline_runner.search(
            SearchRequest(
                query="billing policy",
                collections=["corpus-1"],
                namespace="space-1",
                content_types=["document"],
                document_ids=["doc-allowed"],
                metadata_filter=metadata_filter,
            )
        )

        assert [r.id for r in results] == ["doc-support", "doc-vector"]
        call = sidecar.calls[0]
        assert call.collections == ["corpus-1"]
        assert call.content_types == ["document"]
        assert call.document_ids == ["doc-allowed"]
        assert call.metadata_filter == metadata_filter

    asyncio.run(_run())


def test_portable_lexical_sidecar_promotes_exact_and_trigram_matches() -> None:
    async def _run() -> None:
        exact = make_search_result(id="doc-exact", title="Fox Query", text="semantic text")
        trigram = make_search_result(id="doc-trigram", title="Foks Queri", text="semantic text")
        sidecar = PortableLexicalSidecar(
            records=[
                LexicalSidecarRecord(namespace="space-1", result=trigram),
                LexicalSidecarRecord(namespace="space-1", result=exact),
            ]
        )
        pipeline_runner = _pipeline_runner(sidecar=sidecar)

        exact_results = await pipeline_runner.search(
            SearchRequest(
                query="fox query", collections=["corpus-1"], namespace="space-1"
            )
        )
        trigram_results = await pipeline_runner.search(
            SearchRequest(
                query="fox quary", collections=["corpus-1"], namespace="space-1"
            )
        )

        assert exact_results[0].id == "doc-exact"
        assert exact_results[0].score == 1.0
        assert trigram_results[0].score > 0.35
        exact_meta = cast(dict[str, object], exact_results[0].metadata["search_sidecar"])
        trigram_meta = cast(dict[str, object], trigram_results[0].metadata["search_sidecar"])
        assert exact_meta["strategy"] == "exact"
        assert trigram_meta["strategy"] == "trigram"

    asyncio.run(_run())


def test_trigram_sidecar_does_not_outrank_stronger_vector_hit() -> None:
    async def _run() -> None:
        sidecar = PortableLexicalSidecar(
            records=[
                LexicalSidecarRecord(
                    namespace="space-1",
                    result=make_search_result(
                        id="side",
                        title="alphx",
                        score=0.1,
                        namespace="space-1",
                        collection="corpus-1",
                    ),
                )
            ]
        )
        store = RecordingVectorStore(
            search_results=[
                make_search_result(
                    id="vec",
                    score=0.95,
                    namespace="space-1",
                    collection="corpus-1",
                )
            ]
        )
        pipeline_runner = _pipeline_runner(store=store, sidecar=sidecar)

        results = await pipeline_runner.search(
            SearchRequest(query="alpha", collections=["corpus-1"], namespace="space-1")
        )

        assert [result.id for result in results[:2]] == ["vec", "side"]
        sidecar_meta = cast(dict[str, object], results[1].metadata["search_sidecar"])
        assert sidecar_meta["strategy"] == "trigram"

    asyncio.run(_run())


def test_portable_lexical_sidecar_applies_metadata_filter() -> None:
    async def _run() -> None:
        sidecar = PortableLexicalSidecar(
            records=[
                LexicalSidecarRecord(
                    namespace="space-1",
                    result=make_search_result(
                        id="doc-support",
                        title="Billing Policy",
                        metadata={"team": "support"},
                    ),
                ),
                LexicalSidecarRecord(
                    namespace="space-1",
                    result=make_search_result(
                        id="doc-sales",
                        title="Billing Policy",
                        metadata={"team": "sales"},
                    ),
                ),
            ]
        )

        results = await sidecar.search(
            SearchSidecarQuery(
                query="billing policy",
                namespace="space-1",
                collections=["corpus-1"],
                metadata_filter=Term(field="team", value="support"),
            )
        )

        assert [r.id for r in results] == ["doc-support"]

    asyncio.run(_run())
