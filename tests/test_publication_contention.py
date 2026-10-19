"""Real Qdrant and SDK/HTTP-fixture storage integrity; no hosted/ranking proof."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from qdrant_client import models as rest

from rag_core.search.document_versions import (
    CHUNK, KIND, REVISION, ConcurrentDocumentUpdate,
)
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore
from rag_core.search.request_models import SearchQuery, StoredDocumentRecord
from rag_core.search.vector_models import VectorPoint
from tests.support.turbopuffer_publication_http import PublicationHTTP

pytestmark = pytest.mark.integration
Store = QdrantVectorStore | TurboPufferVectorStore
SCOPE = dict(tenant_id="contention", collection="docs", document_id="one-document")


def record(marker: str) -> StoredDocumentRecord:
    return StoredDocumentRecord(
        tenant_id=SCOPE["tenant_id"], collection=SCOPE["collection"],
        document_id=SCOPE["document_id"], document_key="one",
        chunk_count=1, content_sha256=marker,
    )


def point(marker: str, *, ranks_first: bool = False) -> VectorPoint:
    return VectorPoint(
        id="chunk-zero", dense_vector=[1., 0.] if ranks_first else [0., 1.],
        payload={
            **SCOPE, "document_key": "one", "text": marker,
            "content_type": "document", "source_type": "fixture", "chunk_index": 0,
        },
    )


def query(limit: int = 1) -> SearchQuery:
    return SearchQuery(
        dense_vector=[1., 0.], tenant_id=SCOPE["tenant_id"],
        collections=[SCOPE["collection"]], limit=limit,
    )


@dataclass
class Harness:
    store: Store
    http: PublicationHTTP | None = None

    async def chunks(self) -> list[dict[str, Any]]:
        if isinstance(self.store, QdrantVectorStore):
            rows, cursor = await self.store._client.scroll(
                collection_name="docs",
                scroll_filter=rest.Filter(must=[
                    rest.FieldCondition(key=KIND, match=rest.MatchValue(value=CHUNK)),
                ]),
                limit=32, with_payload=True, with_vectors=False,
            )
            assert cursor is None
            return [dict(row.payload or {}) for row in rows]
        assert self.http is not None
        return [
            json.loads(row["rag_core_payload"]) for row in self.http.rows.values()
            if row.get(KIND) == CHUNK
        ]


@asynccontextmanager
async def backend(provider: str, *, sdk_retries: int = 0) -> AsyncIterator[Harness]:
    if provider == "qdrant":
        store = QdrantVectorStore(
            url=None, api_key=None, location=":memory:", collection_name="docs",
            dense_dimensions=2, quantization_enabled=False,
        )
        try:
            yield Harness(store)
        finally:
            await store.close()
        return
    assert provider == "turbopuffer"
    sdk = pytest.importorskip("turbopuffer")
    fixture = PublicationHTTP()
    client = sdk.AsyncTurbopuffer(
        api_key="fixture-key", region="gcp-us-central1",
        base_url="https://{region}.fixture.invalid", max_retries=sdk_retries,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fixture)),
    )
    try:
        yield Harness(TurboPufferVectorStore(
            namespace="docs", dense_dimensions=2, client=client,
        ), fixture)
    finally:
        await client.close()


@pytest.mark.parametrize("provider", ["qdrant", "turbopuffer"])
def test_nine_writers_clean_all_eight_rejected_revisions(monkeypatch, provider):
    async def run():
        async with backend(provider) as harness:
            store = harness.store
            await store.replace_document(record("seed"), [point("seed")])
            seed = await store.get_document_record(**SCOPE)
            assert seed is not None and seed.is_complete
            original_publish = store._publish_head
            ready, winner_gate, loser_gate = asyncio.Event(), asyncio.Event(), asyncio.Event()
            arrivals: dict[str, str] = {}
            expected: set[str | None] = set()

            async def gated_publish(id, payload, expected_revision):
                marker, revision = payload["content_sha256"], payload[REVISION]
                if revision not in arrivals.values():
                    arrivals[marker] = revision
                    expected.add(expected_revision)
                    if len(arrivals) == 9:
                        ready.set()
                    await (winner_gate if marker == "winner" else loser_gate).wait()
                return await original_publish(id, payload, expected_revision)

            monkeypatch.setattr(store, "_publish_head", gated_publish)
            markers = ["winner", *(f"loser-{i}" for i in range(8))]
            tasks = [asyncio.create_task(store.replace_document(
                record(marker), [point(marker, ranks_first=marker != "winner")],
            )) for marker in markers]
            try:
                await asyncio.wait_for(ready.wait(), 5)
                assert len(arrivals) == 9 and len(set(arrivals.values())) == 9
                assert expected == {seed.revision}
                staged = await harness.chunks()
                assert len(staged) == 10  # seed plus all nine staged versions
                winner_gate.set()
                await asyncio.wait_for(tasks[0], 5)
                loser_gate.set()
                outcomes = await asyncio.wait_for(
                    asyncio.gather(*tasks[1:], return_exceptions=True), 5
                )
                assert len(outcomes) == 8
                assert all(isinstance(outcome, ConcurrentDocumentUpdate) for outcome in outcomes)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                monkeypatch.setattr(store, "_publish_head", original_publish)

            winner = await store.get_document_record(**SCOPE)
            assert winner is not None and winner.is_complete
            assert winner.content_sha256 == "winner"
            assert winner.revision == arrivals["winner"]
            for marker in markers[1:]:
                assert await store._count_revision(winner, arrivals[marker]) == 0
            chunks = await harness.chunks()
            assert [(row["text"], row[REVISION]) for row in chunks] == [("winner", winner.revision)]

            original_search = store._search_candidates
            rounds = 0

            async def counted_search(request):
                nonlocal rounds
                rounds += 1
                return await original_search(request)

            monkeypatch.setattr(store, "_search_candidates", counted_search)
            assert [hit.text for hit in await store.search(query())] == ["winner"]
            assert rounds == 1
            raw = await original_search(query(limit=32))
            assert [(hit.text, revision) for hit, revision in raw] == [("winner", winner.revision)]

    asyncio.run(run())


@pytest.mark.parametrize("existing", [False, True], ids=["insert", "replace"])
def test_turbopuffer_sdk_replay_after_applied_lost_ack_is_success(existing):
    async def run():
        async with backend("turbopuffer", sdk_retries=1) as harness:
            store, fixture = harness.store, harness.http
            assert fixture is not None
            if existing:
                await store.replace_document(record("seed"), [point("seed")])
            start = len(fixture.head_requests)
            fixture.fail_next_head_ack = True
            await store.replace_document(record("winner"), [point("winner")])
            assert fixture.head_applied.is_set()
            assert fixture.head_outcomes[start:start + 2] == [1, 0]
            assert fixture.head_requests[start] == fixture.head_requests[start + 1]
            current = await store.get_document_record(**SCOPE)
            assert current is not None and current.is_complete
            assert current.revision is not None
            assert current.content_sha256 == "winner"
            assert await store._count_revision(current, current.revision) == 1
            assert [row["text"] for row in await harness.chunks()] == ["winner"]
            assert [hit.text for hit in await store.search(query())] == ["winner"]

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["lost_ack", "cancel_after_apply"])
def test_turbopuffer_unknown_publication_preserves_applied_chunks(failure):
    async def run():
        async with backend("turbopuffer") as harness:
            store, fixture = harness.store, harness.http
            assert fixture is not None
            if failure == "lost_ack":
                sdk = pytest.importorskip("turbopuffer")
                fixture.fail_next_head_ack = True
                with pytest.raises(sdk.InternalServerError) as caught:
                    await store.replace_document(record("winner"), [point("winner")])
                assert getattr(caught.value, "status_code", None) == 503
            else:
                fixture.pause_next_head_ack = True
                task = asyncio.create_task(store.replace_document(record("winner"), [point("winner")]))
                try:
                    await asyncio.wait_for(fixture.head_applied.wait(), 5)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            assert fixture.head_outcomes == [1]
            current = await store.get_document_record(**SCOPE)
            assert current is not None and current.is_complete
            assert current.revision is not None
            assert current.content_sha256 == "winner"
            assert await store._count_revision(current, current.revision) == 1
            assert [row["text"] for row in await harness.chunks()] == ["winner"]
            assert [hit.text for hit in await store.search(query())] == ["winner"]

    asyncio.run(run())
