from __future__ import annotations

import asyncio

from rag_core import Engine

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_test_config,
)


def test_rag_core_async_context_ensures_ready_and_closes() -> None:
    async def scenario() -> RecordingVectorStore:
        store = RecordingVectorStore()
        core = Engine(
            make_test_config(
                qdrant_collection="rag_core_context_manager",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )

        async with core as opened:
            assert opened is core
            assert store.ensure_collection_calls == 1
            assert store.close_calls == 0

        return store

    store = asyncio.run(scenario())
    assert store.operations == ["ensure_collection", "close"]
    assert store.close_calls == 1
