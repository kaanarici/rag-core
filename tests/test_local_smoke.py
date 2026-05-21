"""End-to-end smoke: real RAGCore + in-memory Qdrant + fake providers.

This proves the wiring between RAGCore, embedding/sparse providers, and
the vector store works without external services. Logic-level cases live
elsewhere.
"""

import asyncio

import pytest

from rag_core import RAGCore
from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder, make_test_config

pytestmark = [pytest.mark.integration]


def test_local_ingest_and_search_smoke() -> None:
    async def go() -> None:
        embedding = FakeEmbeddingProvider(vocabulary=("fox", "rag", "smoke", "tests"))
        core = RAGCore(
            make_test_config(
                qdrant_collection="rag_core_test_smoke",
                embedding_dimensions=embedding.dimensions,
            ),
            embedding_provider=embedding,
            sparse_embedder=FakeSparseEmbedder(),
        )
        try:
            ingested = await core.ingest_bytes(
                file_bytes=b"rag smoke tests keep the fox easy to find",
                filename="smoke.txt",
                mime_type="text/plain",
                namespace="test-space",
                corpus_id="test-corpus",
            )
            assert ingested.chunk_count > 0

            hits = await core.search(
                query="fox smoke",
                namespace="test-space",
                corpus_ids=["test-corpus"],
                limit=3,
                rerank=False,
            )
            assert hits
            assert hits[0].document_id == ingested.document_id
            assert "fox" in hits[0].text.lower()
        finally:
            await core.close()

    asyncio.run(go())
