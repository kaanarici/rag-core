from __future__ import annotations

import asyncio
from argparse import Namespace

import pytest

from rag_core.cli import _build_parser
from rag_core.cli_doctor import _planned_core_payload
from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.core_models import RAGCoreConfig
from rag_core.search.indexer import DocumentIndexer, IndexRequest
from rag_core.search.indexer_embeddings import prepare_index_data

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
)


def _index_request(*, chunk_count: int = 5) -> IndexRequest:
    return IndexRequest(
        document_id="doc-1",
        corpus_id="help",
        namespace="acme",
        text="unused",
        filename="guide.txt",
        mime_type="text/plain",
        source_type="file",
        pre_chunked_texts=[f"fox query {index}" for index in range(chunk_count)],
    )


def test_embedding_config_defaults_and_validates_batch_size() -> None:
    assert EmbeddingConfig().batch_size == 50
    assert EmbeddingConfig(batch_size=3).batch_size == 3
    positional = EmbeddingConfig(
        "openai",
        "text-embedding-3-small",
        1536,
        "sk-test",
        "https://api.example.test",
    )
    assert positional.api_key == "sk-test"
    assert positional.base_url == "https://api.example.test"
    assert positional.batch_size == 50

    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="batch_size"):
            EmbeddingConfig(batch_size=value)  # type: ignore[arg-type]


def test_rag_core_config_from_cli_carries_embedding_batch_size() -> None:
    config = RAGCoreConfig.from_cli(
        Namespace(
            qdrant_url=None,
            qdrant_location=":memory:",
            qdrant_api_key="",
            qdrant_collection="rag_core_chunks",
            dimension_aware_collection=True,
            vector_store="qdrant",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            embedding_batch_size=7,
            reranker_provider="none",
            reranker_model=None,
            processing_version="test-processing",
        )
    )

    assert config.embedding.batch_size == 7


def test_config_flags_accept_embedding_batch_size() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "doctor",
            "--embedding-model",
            "text-embedding-3-small",
            "--embedding-dimensions",
            "1536",
            "--embedding-batch-size",
            "8",
            "--json",
        ]
    )

    config = RAGCoreConfig.from_cli(args)
    payload = _planned_core_payload(config)

    assert config.embedding.batch_size == 8
    assert payload["embedding"] == {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "batch_size": 8,
    }


def test_config_flags_reject_non_positive_embedding_batch_size() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "doctor",
            "--embedding-batch-size",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="batch_size"):
        RAGCoreConfig.from_cli(args)


def test_prepare_index_data_batches_dense_embeddings() -> None:
    async def _run() -> None:
        embedding = FakeEmbeddingProvider()

        await prepare_index_data(
            req=_index_request(chunk_count=5),
            embedding_provider=embedding,
            sparse_embedder=FakeSparseEmbedder(include_extra_channel=False),
            embedding_batch_size=2,
        )

        assert [len(call) for call in embedding.embed_texts_calls] == [2, 2, 1]

    asyncio.run(_run())


def test_indexer_uses_configured_embedding_batch_size() -> None:
    async def _run() -> None:
        embedding = FakeEmbeddingProvider()
        indexer = DocumentIndexer(
            embedding_provider=embedding,
            sparse_embedder=FakeSparseEmbedder(include_extra_channel=False),
            vector_store=RecordingVectorStore(),
            embedding_batch_size=2,
        )

        result = await indexer.index_document(_index_request(chunk_count=5))

        assert result.chunk_count == 5
        assert [len(call) for call in embedding.embed_texts_calls] == [2, 2, 1]

    asyncio.run(_run())


def test_indexer_rejects_invalid_embedding_batch_size() -> None:
    for value in (0, 1.5):
        with pytest.raises(ValueError, match="embedding_batch_size"):
            DocumentIndexer(
                embedding_provider=FakeEmbeddingProvider(),
                sparse_embedder=FakeSparseEmbedder(include_extra_channel=False),
                vector_store=RecordingVectorStore(),
                embedding_batch_size=value,  # type: ignore[arg-type]
            )


def test_rag_core_config_still_propagates_qdrant_validation() -> None:
    with pytest.raises(ValueError, match="not both"):
        RAGCoreConfig(
            qdrant=QdrantConfig(url="http://localhost:6333", location=":memory:"),
            embedding=EmbeddingConfig(batch_size=2),
        )
