from __future__ import annotations

import asyncio

from rag_core import Engine

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
    make_test_config,
)


def _make_core(store: RecordingVectorStore) -> Engine:
    return Engine(
        make_test_config(
            qdrant_collection="rag_core_scope_vocabulary",
            embedding_dimensions=4,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )


def test_engine_public_scope_uses_collection_and_collections() -> None:
    async def scenario() -> RecordingVectorStore:
        store = RecordingVectorStore(
            search_results=[
                make_search_result(
                    id="hit-1",
                    text="fox result",
                    score=0.88,
                    document_id="doc-7",
                    collection="docs",
                )
            ]
        )
        core = _make_core(store)
        try:
            await core.add_bytes(
                file_bytes=b"fox result",
                filename="docs.md",
                mime_type="text/markdown",
                collection="docs",
                document_id="doc-7",
            )
            await core.search(query="fox query", collection="docs", rerank=False)
            await core.context(query="fox query", collections=["docs"], rerank=False)
        finally:
            await core.close()
        return store

    store = asyncio.run(scenario())

    assert store.upsert_calls[0][0].payload["namespace"] == "default"
    assert store.upsert_calls[0][0].payload["collection"] == "docs"
    assert store.search_calls[-2].namespace == "default"
    assert store.search_calls[-2].collections == ["docs"]
    assert store.search_calls[-1].namespace == "default"
    assert store.search_calls[-1].collections == ["docs"]
