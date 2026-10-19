"""Local storage/lifetime plumbing with deterministic embeddings; no model proof."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import pytest

from rag_core.core import Engine
from rag_core.core_models import Config
from rag_core.demo import DemoEmbeddingProvider
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from rag_core.search.vector_models import VectorPoint

pytestmark = pytest.mark.plumbing


def make_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        url=None, api_key=None, location=":memory:", collection_name="write-lifetime",
        dense_dimensions=3072, quantization_enabled=False,
        embedding_model="demo-dense-v1",
    )


def test_engine_local_writes_can_contend_on_successive_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store()
    original_upsert = store._client.upsert

    async def yielding_upsert(**kwargs: Any) -> Any:
        # Yield like network I/O while retaining real local publication behavior.
        if kwargs["points"][0].vector:
            await asyncio.sleep(0.05)
        return await original_upsert(**kwargs)

    monkeypatch.setattr(store._client, "upsert", yielding_upsert)
    engine = Engine(
        Config.local(), vector_store=store,
        embedding_provider=DemoEmbeddingProvider(dimensions=3072),
    )

    async def batch(round_number: int) -> None:
        results = await asyncio.gather(*(
            engine.add_bytes(
                file_bytes=b"Invoices accept payment by card.",
                filename="invoice.txt", mime_type="text/plain",
                tenant_id="t", collection="c", document_id=f"{round_number}-{i}",
            )
            for i in range(5)
        ))
        assert [result.ingest_state for result in results] == ["created"] * 5

    try:
        asyncio.run(batch(1))
        asyncio.run(batch(2))
    finally:
        asyncio.run(engine.close())


def test_local_write_waiters_remain_owned_and_concurrent_loops_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_store()
    point = VectorPoint(
        id=str(UUID(int=1)), dense_vector=[1.0] + [0.0] * 3071, payload={},
    )

    async def scenario() -> None:
        await store.ensure_collection()
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def held_upsert(**kwargs: Any) -> None:
            nonlocal calls
            calls += 1
            if calls == 4:
                entered.set()
            await release.wait()

        monkeypatch.setattr(store._client, "upsert", held_upsert)
        tasks = [asyncio.create_task(store.upsert([point])) for _ in range(5)]
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert store._active_writes == 5
            tasks[-1].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[-1]
            assert store._active_writes == 4

            def other_loop() -> None:
                asyncio.run(store.upsert([point]))

            with pytest.raises(RuntimeError, match="concurrent event loops"):
                await asyncio.to_thread(other_loop)
            assert calls == 4
        finally:
            release.set()
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        assert outcomes[:4] == [None] * 4
        assert store._active_writes == 0

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(store.close())
