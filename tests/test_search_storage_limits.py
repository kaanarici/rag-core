"""Work and memory contracts around actual adapter translation, without hosted calls."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID
import weakref
import pytest
from rag_core.search.context_pack import build_context_pack
from rag_core.search.indexer import DocumentIndexer, IndexRequest
from rag_core.search.providers import qdrant_write
from rag_core.search.providers.qdrant_payloads import WriteLatencyTracker
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore
from rag_core.search.request_models import RerankBudget
from rag_core.search.runner import SearchRequest, SearchRunner
from rag_core.search.vector_models import VectorPoint
from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
)


def _point(index: int, *, bad_payload: bool = False) -> VectorPoint:
    return VectorPoint(
        id=str(UUID(int=index + 1)),
        dense_vector=[1.0, 0.0, 0.0, 0.0],
        payload={"bad": object()}
        if bad_payload
        else {"tenant_id": "acme", "text": "Invoice policy."},
    )


@pytest.mark.parametrize("backend", ["qdrant", "turbopuffer"])
@pytest.mark.parametrize("candidate_count", [None, 12])
def test_hybrid_rerank_budget_is_applied_once(
    backend: str, candidate_count: int | None
) -> None:
    class Capture:
        calls: list[Any]

        def __init__(self) -> None:
            self.calls = []

        async def query_points(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(points=[])

        async def multi_query(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(
                results=[SimpleNamespace(rows=[]), SimpleNamespace(rows=[])]
            )

    class Reranker:
        async def rerank(
            self, query: str, documents: list[str], top_k: int = 10
        ) -> list[Any]:
            return []

    async def run() -> None:
        capture = Capture()
        if backend == "qdrant":
            store: Any = QdrantVectorStore(
                url=None,
                api_key=None,
                location=":memory:",
                collection_name="test",
                dense_dimensions=4,
                sparse_enabled=True,
            )
            store.bind_embedding_space(FakeEmbeddingProvider().model_name)
            await store._client.close()
            store._client, store._ready = capture, True
        else:
            store = TurboPufferVectorStore(
                namespace="test", dense_dimensions=4, namespace_client=capture
            )
        if backend == "turbopuffer":
            store._identity_ready = (
                True  # Query-budget fixture; handshake has separate SDK coverage.
            )
        await SearchRunner(
            FakeEmbeddingProvider(), FakeSparseEmbedder(), store, Reranker()
        ).search(
            SearchRequest(
                query="invoices",
                tenant_id="acme",
                collections=["docs"],
                document_ids=["doc"],
                mode="hybrid",
                limit=10,
                rerank_budget=RerankBudget(candidate_count=candidate_count),
            )
        )
        assert len(capture.calls) == 1
        call = capture.calls[0]
        expected = candidate_count or 40
        if backend == "qdrant":
            assert call["limit"] == expected
            assert [query.limit for query in call["prefetch"]] == [expected, expected]
            assert all(
                query.filter == call["query_filter"] for query in call["prefetch"]
            )
        else:
            assert [query["limit"] for query in call["queries"]] == [expected, expected]
            assert call["queries"][0]["filters"] == call["queries"][1]["filters"]

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["qdrant", "turbopuffer"])
def test_invalid_final_point_is_rejected_before_any_batch_write(backend: str) -> None:
    class ForbiddenClient:
        async def upsert(self, **kwargs: Any) -> None:
            raise AssertionError("invalid input must fail before any write")

        async def write(self, **kwargs: Any) -> None:
            raise AssertionError("invalid input must fail before any write")

    async def run() -> None:
        if backend == "qdrant":
            store: Any = QdrantVectorStore(
                url=None,
                api_key=None,
                location=":memory:",
                collection_name="test",
                dense_dimensions=4,
            )
            await store._client.close()
            store._client, store._ready, store._max_batch_size = (
                ForbiddenClient(),
                True,
                10,
            )
        else:
            store = TurboPufferVectorStore(
                namespace="test",
                dense_dimensions=4,
                namespace_client=ForbiddenClient(),
                write_batch_size=10,
            )
        points = [_point(i) for i in range(30)] + [_point(30, bad_payload=True)]
        with pytest.raises(ValueError, match="unsupported value type"):
            await store.upsert(points)

    asyncio.run(run())


def test_qdrant_materializes_only_the_active_write_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        live = peak = 0
        original = qdrant_write._build_point

        def release() -> None:
            nonlocal live
            live -= 1

        def tracked(*args: Any, **kwargs: Any) -> Any:
            nonlocal live, peak
            point = original(*args, **kwargs)
            live += 1
            peak = max(peak, live)
            weakref.finalize(point, release)
            return point

        monkeypatch.setattr(qdrant_write, "_build_point", tracked)

        class Client:
            ids: set[Any] = set()

            async def upsert(self, **kwargs: Any) -> None:
                self.ids.update(point.id for point in kwargs["points"])
                await asyncio.sleep(0)

        client = Client()
        await qdrant_write.upsert_qdrant_point_batches(
            client=cast(Any, client),
            collection_name="test",
            dimensions=4,
            latency=WriteLatencyTracker(),
            max_batch_size=10,
            max_concurrent=2,
            write_sem=asyncio.Semaphore(2),
            points=[_point(i) for i in range(500)],
            available_sparse_vector_names=set(),
        )
        assert len(client.ids) == 500
        assert peak <= 20
        assert live == 0

    asyncio.run(run())


def test_explicit_empty_chunks_publish_without_rechunking_or_embedding() -> None:
    async def run() -> None:
        embedding, store = FakeEmbeddingProvider(), RecordingVectorStore()
        await DocumentIndexer(embedding, None, store).index_document(
            IndexRequest(
                document_id="doc",
                collection="docs",
                tenant_id="acme",
                text="This text must not be rechunked.",
                filename="doc.md",
                mime_type="text/markdown",
                source_type="file",
                prepared_chunks=[],
            )
        )
        assert not embedding.embed_texts_calls
        assert store.upsert_calls == [[]]
        assert not store.delete_calls
        record = store.document_records[("acme", "docs", "doc")]
        assert record.is_complete and record.chunk_count == 0

    asyncio.run(run())


def test_context_computes_each_dedupe_key_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_core.search.context_pack as context

    count = 0
    original = context._source_dedupe_key

    def tracked(result: Any) -> Any:
        nonlocal count
        count += 1
        return original(result)

    monkeypatch.setattr(context, "_source_dedupe_key", tracked)
    results = [
        make_search_result(id=f"hit-{i}", document_id=f"doc-{i}", chunk_index=0)
        for i in range(10)
    ]
    pack = build_context_pack(results + results, query="invoice", max_snippets=20)
    assert count == 20
    assert len(pack.snippets) == 10 and pack.dropped_count == 10


def test_dimension_mismatch_does_no_embedding_work() -> None:
    from rag_core.search.provider_protocols import StoreCapabilities

    async def run() -> None:
        embedding, store = FakeEmbeddingProvider(), RecordingVectorStore()
        store.capabilities = StoreCapabilities(
            per_point_delete=True, dense_vector_dimensions=8
        )
        request = IndexRequest(
            document_id="doc",
            collection="docs",
            tenant_id="acme",
            text="Invoice policy.",
            filename="doc.md",
            mime_type="text/markdown",
            source_type="file",
        )
        with pytest.raises(ValueError, match="Dense dimension mismatch"):
            await DocumentIndexer(embedding, None, store).index_document(request)
        assert not embedding.embed_texts_calls and not store.upsert_calls

    asyncio.run(run())


def test_bounded_write_cancellation_joins_active_workers() -> None:
    async def run() -> None:
        active = 0
        started = asyncio.Event()

        class Client:
            async def upsert(self, **kwargs: Any) -> None:
                nonlocal active
                active += 1
                if active == 2:
                    started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    active -= 1

        task = asyncio.create_task(
            qdrant_write.upsert_qdrant_point_batches(
                client=cast(Any, Client()),
                collection_name="test",
                dimensions=4,
                latency=WriteLatencyTracker(),
                max_batch_size=10,
                max_concurrent=2,
                write_sem=asyncio.Semaphore(2),
                points=[_point(i) for i in range(100)],
                available_sparse_vector_names=set(),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert active == 0

    asyncio.run(run())


@pytest.mark.parametrize("status", [401, 413])
def test_qdrant_split_retry_distinguishes_terminal_errors(
    status: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import httpx
    from qdrant_client.http.exceptions import UnexpectedResponse
    from rag_core.search.providers.qdrant_payloads import _build_point

    async def run() -> None:
        class Client:
            sizes: list[int] = []

            async def upsert(self, **kwargs: Any) -> None:
                self.sizes.append(len(kwargs["points"]))
                if status == 401 or len(self.sizes) == 1:
                    raise UnexpectedResponse(
                        status_code=status,
                        reason_phrase="Failure",
                        content=b"private token sk-test-secret",
                        headers=httpx.Headers(),
                    )

        client = Client()
        points = [
            _build_point(_point(i), available_sparse_vector_names=set())
            for i in range(8)
        ]
        call = qdrant_write.upsert_with_fallback(
            cast(Any, client),
            "private-collection",
            4,
            WriteLatencyTracker(),
            points,
            0,
        )
        if status == 401:
            with pytest.raises(UnexpectedResponse):
                await call
            assert client.sizes == [8]
        else:
            await call
            assert client.sizes == [8, 4, 4]
        assert (
            "private token" not in caplog.text
            and "private-collection" not in caplog.text
        )

    asyncio.run(run())
