"""Contracts for the single runner; real ranking is covered by local semantic evals."""

from __future__ import annotations
import asyncio
from dataclasses import replace
import pytest
from rag_core.core import Engine
from tests.support import KeywordSparseEmbedder
from rag_core.events import EventBuffer
from rag_core.search.filters import Term
from rag_core.search.provider_protocols import StoreCapabilities
from rag_core.search.request_models import RerankBudget, RerankResult
from rag_core.search.runner import SearchRequest, SearchRunner
from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
    make_test_config,
)


@pytest.mark.parametrize("mode", ["dense", "hybrid", "lexical"])
def test_runner_embeds_only_channels_the_mode_uses(mode):
    async def run():
        embedding, sparse = FakeEmbeddingProvider(), FakeSparseEmbedder()
        store = RecordingVectorStore(search_results=[make_search_result()])
        runner = SearchRunner(embedding, sparse, store)
        result = await runner.search(
            SearchRequest(
                query="invoice",
                tenant_id="space-1",
                collections=["corpus-1"],
                mode=mode,
            )
        )
        assert len(result) == 1
        assert bool(embedding.embed_query_calls) == (mode != "lexical")
        assert bool(sparse.embed_query_calls) == (mode != "dense")
        assert store.search_calls[0].mode == mode

    asyncio.run(run())


@pytest.mark.parametrize("field", ["collections", "document_ids", "content_types"])
def test_empty_scope_does_no_provider_work(field):
    async def run():
        embedding, sparse = FakeEmbeddingProvider(), FakeSparseEmbedder()
        store = RecordingVectorStore()
        request = SearchRequest(
            query="q",
            tenant_id="tenant",
            collections=[] if field == "collections" else ["docs"],
            document_ids=[] if field == "document_ids" else None,
            content_types=[] if field == "content_types" else None,
        )
        assert await SearchRunner(embedding, sparse, store).search(request) == []
        assert not embedding.embed_query_calls and not sparse.embed_query_calls
        assert not store.search_calls

    asyncio.run(run())


def test_unsupported_mode_fails_before_embedding():
    async def run():
        embedding = FakeEmbeddingProvider()
        store = RecordingVectorStore()
        store.capabilities = StoreCapabilities(per_point_delete=True)
        with pytest.raises(ValueError, match="does not support hybrid"):
            await SearchRunner(embedding, None, store).search(
                SearchRequest(
                    query="q", tenant_id="tenant", collections=["docs"], mode="hybrid"
                )
            )
        assert not embedding.embed_query_calls and not store.search_calls

    asyncio.run(run())


def test_rerank_widens_candidates_and_preserves_scope_and_trace():
    class Reverse:
        async def rerank(self, query, documents, top_k=10):
            return [
                RerankResult(index=2, score=0.9, text="ignored"),
                RerankResult(index=2, score=0.8, text="duplicate"),
                RerankResult(index=99, score=0.7, text="invalid"),
            ]

    async def run():
        events = EventBuffer()
        hits = [make_search_result(id=f"hit-{i}", text=f"chunk {i}") for i in range(3)]
        store = RecordingVectorStore(search_results=hits)
        runner = SearchRunner(
            FakeEmbeddingProvider(), FakeSparseEmbedder(), store, Reverse(), events
        )
        request = SearchRequest(
            query="q",
            tenant_id="tenant",
            collections=["docs"],
            document_ids=["doc"],
            content_types=["document"],
            metadata_filter=Term("team", "a"),
            limit=1,
        )
        result = await runner.search_with_trace(request)
        assert result.results[0].id == "hit-2"
        assert result.results[0].text == "chunk 2"
        assert store.search_calls[0].limit > request.limit
        assert store.search_calls[0].document_ids == ["doc"]
        assert store.search_calls[0].metadata_filter == Term("team", "a")
        assert all(
            getattr(e, "search_id", result.search_id) == result.search_id
            for e in events.events
        )
        completed = next(e for e in events.events if e.event_type == "search.completed")
        assert completed.applied_rerank and completed.succeeded

    asyncio.run(run())


def test_rerank_timeout_falls_back_only_when_requested():
    class Slow:
        async def rerank(self, *args, **kwargs):
            await asyncio.sleep(1)

    async def run():
        hit = make_search_result()
        runner = SearchRunner(
            FakeEmbeddingProvider(),
            None,
            RecordingVectorStore(search_results=[hit]),
            Slow(),
        )
        request = SearchRequest(
            query="q",
            tenant_id="tenant",
            collections=["docs"],
            rerank_budget=RerankBudget(timeout_seconds=0.01, fallback_on_error=True),
        )
        assert await runner.search(request) == [hit]
        with pytest.raises(TimeoutError):
            await runner.search(
                replace(
                    request,
                    rerank_budget=RerankBudget(
                        timeout_seconds=0.01, fallback_on_error=False
                    ),
                )
            )

    asyncio.run(run())


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["dense", "hybrid", "lexical"])
def test_real_qdrant_modes_and_scope(mode):
    async def run():
        config = replace(make_test_config(embedding_dimensions=4), hybrid=True)
        async with Engine(
            config,
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=KeywordSparseEmbedder(("invoices", "bank", "private")),
        ) as engine:
            first = await engine.add_bytes(
                file_bytes=b"Invoices accept bank transfer.",
                filename="billing.md",
                mime_type="text/markdown",
                tenant_id="acme",
                collection="docs",
            )
            await engine.add_bytes(
                file_bytes=b"Invoices accept private payment.",
                filename="private.md",
                mime_type="text/markdown",
                tenant_id="other",
                collection="docs",
            )
            hits = await engine.search(
                query="Invoices", tenant_id="acme", collection="docs", mode=mode
            )
            assert hits and all(h.document_id == first.document_id for h in hits)
            assert (
                await engine.search(
                    query="Invoices",
                    tenant_id="acme",
                    collection="docs",
                    document_ids=[],
                    mode=mode,
                )
                == []
            )
            await engine.delete_document(
                document_id=first.document_id, tenant_id="acme", collection="docs"
            )
            assert (
                await engine.search(
                    query="Invoices", tenant_id="acme", collection="docs", mode=mode
                )
                == []
            )

    asyncio.run(run())
