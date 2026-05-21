from __future__ import annotations

import asyncio
import logging

import pytest

from rag_core.search.indexer import IndexRequest, QdrantIndexer
from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder, RecordingVectorStore


def test_index_document_info_log_omits_document_identity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def run() -> None:
        store = RecordingVectorStore()
        indexer = QdrantIndexer(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(include_extra_channel=False),
            vector_store=store,
        )

        with caplog.at_level(logging.INFO, logger="rag_core.search.indexer"):
            result = await indexer.index_document(
                IndexRequest(
                    document_id="doc-private-sk-test-secret",
                    corpus_id="corpus-private-sk-test-secret",
                    namespace="namespace-private-sk-test-secret",
                    text="unused",
                    filename="private-roadmap.md",
                    mime_type="text/markdown",
                    source_type="file",
                    document_key="/Users/person/private-roadmap.md",
                    pre_chunked_texts=["fox query"],
                )
            )

        assert result.document_id == "doc-private-sk-test-secret"
        assert result.document_key == "/Users/person/private-roadmap.md"
        assert result.chunk_count == 1
        assert store.operations == ["upsert"]

    asyncio.run(run())

    assert "Indexed 1 chunks" in caplog.text
    assert "doc-private" not in caplog.text
    assert "corpus-private" not in caplog.text
    assert "namespace-private" not in caplog.text
    assert "private-roadmap" not in caplog.text
    assert "/Users/person" not in caplog.text
    assert "fox query" not in caplog.text
    assert "sk-test-secret" not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
