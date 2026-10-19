"""Remote readiness plumbing with real SDK response models; no server proof."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import httpx
from qdrant_client import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_core.search.document_versions import KIND, REVISION
from rag_core.search.policy import DEFAULT_POLICY
from rag_core.search.providers.qdrant_store import (
    QdrantVectorStore,
    collection_index_fields,
)

pytestmark = pytest.mark.plumbing
REQUIRED = {field for field, _ in collection_index_fields(DEFAULT_POLICY)}
INTERNAL = {KIND, REVISION}


class ReadinessClient:
    def __init__(self, *, fresh: bool, replication: Any = 1, writes: Any = 1):
        self.exists = not fresh
        self.create_calls = 0
        self.index_calls: list[dict[str, Any]] = []
        self.block_field: str | None = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.deny_field: str | None = None
        self.status = rest.UpdateStatus.COMPLETED
        self.params = rest.CollectionParams(
            vectors={"": rest.VectorParams(size=2, distance=rest.Distance.COSINE)},
            replication_factor=1,
            write_consistency_factor=1,
        ).model_copy(update={
            "replication_factor": replication,
            "write_consistency_factor": writes,
        })
        self.schema = {
            field: rest.PayloadIndexInfo(data_type=rest.PayloadSchemaType.KEYWORD, points=0)
            for field in REQUIRED - INTERNAL
        } if not fresh else {}
        self.schema["app_owned"] = rest.PayloadIndexInfo(
            data_type=rest.PayloadSchemaType.INTEGER, points=17
        )

    async def get_collections(self) -> Any:
        return rest.CollectionsResponse(
            collections=[rest.CollectionDescription(name="docs")] if self.exists else []
        )

    async def create_collection(self, **kwargs: Any) -> bool:
        assert "replication_factor" not in kwargs
        assert "write_consistency_factor" not in kwargs
        self.create_calls += 1
        self.exists = True
        return True

    async def get_collection(self, **kwargs: Any) -> rest.CollectionInfo:
        return rest.CollectionInfo(
            status=rest.CollectionStatus.GREEN,
            optimizer_status=rest.OptimizersStatusOneOf.OK,
            segments_count=1,
            config=rest.CollectionConfig(
                params=self.params,
                hnsw_config=rest.HnswConfig(m=16, ef_construct=100, full_scan_threshold=10000),
                optimizer_config=rest.OptimizersConfig(
                    deleted_threshold=0.2, vacuum_min_vector_number=1000,
                    default_segment_number=0, flush_interval_sec=5,
                ),
            ),
            payload_schema=dict(self.schema),
        )

    async def create_payload_index(self, **kwargs: Any) -> rest.UpdateResult:
        self.index_calls.append(kwargs)
        assert kwargs["wait"] is True
        field = kwargs["field_name"]
        if field == self.block_field:
            self.entered.set()
            await self.release.wait()
        if field == self.deny_field:
            raise UnexpectedResponse(
                status_code=403, reason_phrase="Forbidden", content=b"index denied",
                headers=httpx.Headers(),
            )
        if self.status == rest.UpdateStatus.COMPLETED:
            self.schema[field] = rest.PayloadIndexInfo(
                data_type=rest.PayloadSchemaType.KEYWORD, points=0
            )
        return rest.UpdateResult(status=self.status)


def make_store(monkeypatch: pytest.MonkeyPatch, client: ReadinessClient, *, local: bool = False):
    monkeypatch.setattr(
        "rag_core.search.providers.qdrant_store.AsyncQdrantClient", lambda **kwargs: client
    )
    return QdrantVectorStore(
        url=None if local else "https://qdrant.invalid",
        api_key=None, location=":memory:" if local else None,
        collection_name="docs", dense_dimensions=2, quantization_enabled=False,
    )


@pytest.mark.parametrize("fresh", [True, False], ids=["fresh", "reopened"])
def test_readiness_waits_for_required_indexes_and_concurrent_waiters(monkeypatch, fresh):
    async def run():
        client = ReadinessClient(fresh=fresh)
        client.block_field = "source_type" if fresh else REVISION
        store = make_store(monkeypatch, client)
        first = asyncio.create_task(store.ensure_collection())
        await asyncio.wait_for(client.entered.wait(), 2)
        second = asyncio.create_task(store.ensure_collection())
        await asyncio.sleep(0)
        assert not store._ready
        assert not first.done() and not second.done()
        client.release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 2)
        assert store._ready
        expected = REQUIRED if fresh else INTERNAL
        assert {call["field_name"] for call in client.index_calls} == expected
        assert len(client.index_calls) == len(expected)
        assert client.create_calls == int(fresh)
        assert client.schema["app_owned"].points == 17
        await store.ensure_collection()
        assert len(client.index_calls) == len(expected)

    asyncio.run(run())


@pytest.mark.parametrize("fresh", [True, False], ids=["fresh", "reopened"])
def test_index_permission_failure_leaves_unready_and_can_resume(monkeypatch, fresh):
    async def run():
        client = ReadinessClient(fresh=fresh)
        client.deny_field = REVISION
        store = make_store(monkeypatch, client)
        with pytest.raises(UnexpectedResponse) as caught:
            await store.ensure_collection()
        assert caught.value.status_code == 403
        assert not store._ready
        assert KIND in client.schema and REVISION not in client.schema
        client.deny_field = None
        await store.ensure_collection()
        assert store._ready
        assert client.create_calls == int(fresh)
        assert sum(call["field_name"] == KIND for call in client.index_calls) == 1

    asyncio.run(run())


@pytest.mark.parametrize("status", [rest.UpdateStatus.ACKNOWLEDGED, rest.UpdateStatus.WAIT_TIMEOUT])
def test_unfinished_index_operation_does_not_mark_ready(monkeypatch, status):
    async def run():
        client = ReadinessClient(fresh=False)
        client.status = status
        store = make_store(monkeypatch, client)
        with pytest.raises(RuntimeError, match="payload index.*not ready"):
            await store.ensure_collection()
        assert not store._ready

    asyncio.run(run())


@pytest.mark.parametrize("fresh", [True, False])
@pytest.mark.parametrize("replication,writes", [
    (None, None), (3, None), (None, 1), (0, 0), (-1, -1),
    (3, 1), (1, 2), (True, 1), (1, True), (1.0, 1), (1, "1"),
])
def test_remote_refuses_missing_or_unsupported_consistency_without_reconfiguration(
    monkeypatch, fresh, replication, writes,
):
    async def run():
        client = ReadinessClient(fresh=fresh, replication=replication, writes=writes)
        store = make_store(monkeypatch, client)
        with pytest.raises(ValueError, match="replication_factor.*write_consistency_factor"):
            await store.ensure_collection()
        assert not store._ready
        assert not client.index_calls
        assert client.params.replication_factor == replication
        assert client.params.write_consistency_factor == writes

    asyncio.run(run())


@pytest.mark.parametrize("factor", [1, 3])
def test_remote_accepts_equal_positive_consistency_factors(monkeypatch, factor):
    async def run():
        client = ReadinessClient(fresh=False, replication=factor, writes=factor)
        store = make_store(monkeypatch, client)
        await store.ensure_collection()
        assert store._ready

    asyncio.run(run())


@pytest.mark.parametrize("fresh", [True, False])
def test_local_skips_index_and_replication_checks(monkeypatch, fresh):
    async def run():
        client = ReadinessClient(fresh=fresh, replication=None, writes=None)
        store = make_store(monkeypatch, client, local=True)
        await store.ensure_collection()
        assert store._ready
        assert not client.index_calls

    asyncio.run(run())


def test_existing_incompatible_internal_index_fails_without_replacing_it(monkeypatch):
    async def run():
        client = ReadinessClient(fresh=False)
        client.schema[KIND] = rest.PayloadIndexInfo(
            data_type=rest.PayloadSchemaType.INTEGER, points=7
        )
        store = make_store(monkeypatch, client)
        with pytest.raises(ValueError, match="payload index.*must use keyword"):
            await store.ensure_collection()
        assert not store._ready
        assert client.schema[KIND].points == 7
        assert not client.index_calls

    asyncio.run(run())
