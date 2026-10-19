"""TurboPuffer SDK wire contracts against an HTTP transport fixture, not a live service."""

from __future__ import annotations
import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace
import uuid
import pytest
from rag_core.core import Engine
from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore
from rag_core.search.request_models import DeleteFilter, SearchQuery
from rag_core.search.vector_models import VectorPoint
from tests.support import FakeEmbeddingProvider, make_test_config


@pytest.mark.integration
def test_real_sdk_writes_searches_looks_up_and_continues_delete():
    sdk = pytest.importorskip("turbopuffer")
    import httpx

    async def run():
        captured = []
        key = str(uuid.uuid4())
        payload = {
            "tenant_id": "acme",
            "collection": "docs",
            "document_id": "doc",
            "document_key": "billing.md",
            "text": "Invoices accept ACH.",
            "chunk_index": 0,
            "content_type": "document",
            "document_metadata": {"nested": ["preserved"]},
        }
        row = {"id": key, "$dist": 0.1, "rag_core_payload": json.dumps(payload)}
        delete_calls = 0

        def handler(request):
            nonlocal delete_calls
            body = json.loads(request.content)
            captured.append(body)
            if "queries" in body:
                if "aggregate_by" in body["queries"][1]:
                    projected = {key: payload.get(key) for key in body["queries"][0]["include_attributes"]}
                    return httpx.Response(200, json={"results": [{"rows": [{"id": key, **projected}]}, {"aggregations": {"chunk_count": 1}}]})
                return httpx.Response(200, json={"results": [{"rows": [row]}, {"rows": [{**row, "$dist": 3.2}]}]})
            if "aggregate_by" in body:
                return httpx.Response(200, json={"aggregations": {"chunk_count": 1}})
            if request.url.path.endswith("/query"):
                if body.get("filters", [None])[0] == "id" or "head_v2" in repr(body):
                    return httpx.Response(200, json={"rows": []})
                return httpx.Response(200, json={"rows": [row]})
            if "delete_by_filter" in body:
                delete_calls += 1
                return httpx.Response(
                    200, json={"rows_affected": 1, "rows_remaining": delete_calls == 1}
                )
            return httpx.Response(
                200, json={"rows_affected": 1, "rows_remaining": False}
            )

        client = sdk.AsyncTurbopuffer(
            api_key="test-fixture-key",
            region="gcp-us-central1",
            base_url="https://{region}.fixture.test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )
        store = TurboPufferVectorStore(
            namespace="docs", dense_dimensions=4, client=client
        )
        try:
            await store.upsert(
                [
                    VectorPoint(
                        id=key,
                        dense_vector=[1.0, 0.0, 0.0, 0.0],
                        sparse_vectors={},
                        payload=payload,
                        sparse_text="billing context Invoices accept ACH.",
                    )
                ]
            )
            write = captured[-1]
            assert json.loads(write["upsert_rows"][0]["rag_core_payload"]) == payload
            assert "document_metadata" not in write["upsert_rows"][0]
            assert write["upsert_rows"][0]["bm25_text"].startswith("billing context")
            query = SearchQuery(
                dense_vector=[1.0, 0.0, 0.0, 0.0],
                tenant_id="acme",
                collections=["docs"],
                document_ids=["doc"],
                lexical_query="Invoices",
                mode="hybrid",
            )
            hits = await store.search(query)
            assert [h.id for h in hits] == [key]
            assert hits[0].text == payload["text"]
            assert hits[0].metadata["document_metadata"] == payload["document_metadata"]
            queries = next(body["queries"] for body in reversed(captured) if "queries" in body)
            assert queries[0]["filters"] == queries[1]["filters"]
            assert queries[1]["rank_by"] == ["bm25_text", "BM25", "Invoices"]
            record = await store.get_document_record(
                tenant_id="acme", collection="docs", document_key="billing.md"
            )
            assert record is not None
            assert record.document_id == "doc" and record.chunk_count == 1
            calls_before = len(captured)
            record_by_id = await store.get_document_record(tenant_id="acme", collection="docs", document_id="doc")
            assert record_by_id == record
            assert len(captured) == calls_before + 2
            lookup = captured[-1]["queries"][0]
            assert "text" not in lookup["include_attributes"]
            assert "rag_core_payload" not in lookup["include_attributes"]
            await store.delete(DeleteFilter(tenant_id="acme", collection="docs"))
            assert delete_calls == 2
        finally:
            await client.close()

    asyncio.run(run())


def test_native_bm25_avoids_local_sparse_model(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("local BM25 must not load for TurboPuffer")

    monkeypatch.setattr(
        "rag_core.search.providers.sparse.FastEmbedSparseEmbedder", forbidden
    )
    store = TurboPufferVectorStore(
        namespace="docs", dense_dimensions=4, namespace_client=object()
    )
    engine = Engine(
        replace(make_test_config(), hybrid=True),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )
    assert engine._sparse is None


def test_partial_delete_exhaustion_raises():
    class Namespace:
        async def write(self, **kwargs):
            return SimpleNamespace(rows_remaining=True)

    async def run():
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=4,
            namespace_client=Namespace(),
            delete_continuation_limit=2,
        )
        with pytest.raises(RuntimeError, match="continuation"):
            await store.delete(DeleteFilter(tenant_id="acme", collection="docs"))

    asyncio.run(run())
