"""Subsystem configs (Qdrant, Embedding, Reranker, Chunking, Ingest) own their own validation.

``RAGCoreConfig`` is the composition of these typed subsystem configs.
"""

from __future__ import annotations

import pytest

from rag_core import RAGCoreConfig
from rag_core.config import (
    ChunkingConfig,
    EmbeddingConfig,
    IngestConfig,
    QdrantConfig,
    RerankerConfig,
    TurboPufferVectorStoreConfig,
    VectorStoreConfig,
)
from rag_core.core_models import DEFAULT_PROCESSING_VERSION


def test_qdrant_config_defaults() -> None:
    config = QdrantConfig()
    assert config.url is None
    assert config.location is None
    assert config.api_key is None
    assert config.collection == "rag_core_chunks"
    assert config.dimension_aware_collection is True


@pytest.mark.parametrize(
    ("kwargs", "field", "expected"),
    (
        ({"url": "http://localhost:6333"}, "url", "http://localhost:6333"),
        ({"location": ":memory:"}, "location", ":memory:"),
    ),
)
def test_qdrant_config_accepts_url_xor_location(
    kwargs: dict[str, object], field: str, expected: str
) -> None:
    config = QdrantConfig(**kwargs)  # type: ignore[arg-type]
    assert getattr(config, field) == expected


def test_qdrant_config_rejects_url_and_location_together() -> None:
    with pytest.raises(ValueError, match="not both"):
        QdrantConfig(url="http://localhost:6333", location=":memory:")


def test_qdrant_config_normalizes_blank_api_key() -> None:
    assert QdrantConfig(api_key="   ").api_key is None
    assert QdrantConfig(api_key=" key ").api_key == "key"


def test_embedding_config_defaults() -> None:
    config = EmbeddingConfig()
    assert config.provider == "openai"
    assert config.model == "text-embedding-3-large"
    assert config.dimensions is None
    assert config.api_key is None
    assert config.base_url is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"provider": ""}, "non-empty"),
        ({"provider": "   "}, "non-empty"),
        ({"dimensions": 0}, "positive"),
        ({"dimensions": -1}, "positive"),
    ),
)
def test_embedding_config_rejects_invalid_inputs(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        EmbeddingConfig(**kwargs)  # type: ignore[arg-type]


def test_embedding_config_accepts_explicit_dimensions() -> None:
    assert EmbeddingConfig(dimensions=1536).dimensions == 1536


def test_reranker_config_defaults() -> None:
    config = RerankerConfig()
    assert config.provider == "none"
    assert config.model is None
    assert config.api_key is None


def test_chunking_config_constructs_with_defaults() -> None:
    # Empty by design; chunking is selected by router, not config flags.
    ChunkingConfig()


def test_ingest_config_defaults() -> None:
    config = IngestConfig()
    assert config.processing_version == DEFAULT_PROCESSING_VERSION
    assert config.source_type == "file"
    assert config.enable_lexical_search is False


def test_vector_store_config_defaults_to_qdrant() -> None:
    config = VectorStoreConfig()
    assert config.provider == "qdrant"
    assert config.turbopuffer.namespace is None


def test_vector_store_config_normalizes_provider() -> None:
    assert VectorStoreConfig(provider=" TurboPuffer ").provider == "turbopuffer"


def test_vector_store_config_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="qdrant, turbopuffer"):
        VectorStoreConfig(provider="unknown")


def test_turbopuffer_vector_store_config_normalizes_values() -> None:
    config = TurboPufferVectorStoreConfig(
        namespace=" docs ",
        api_key=" key ",
        region=" aws-us-west-2 ",
        base_url=" https://example.invalid ",
        distance_metric=" cosine_distance ",
    )
    assert config.namespace == "docs"
    assert config.api_key == "key"
    assert config.region == "aws-us-west-2"
    assert config.base_url == "https://example.invalid"
    assert config.distance_metric == "cosine_distance"


def test_turbopuffer_vector_store_config_rejects_blank_metric() -> None:
    with pytest.raises(ValueError, match="distance_metric"):
        TurboPufferVectorStoreConfig(distance_metric=" ")


def test_turbopuffer_vector_store_config_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="cosine_distance, euclidean_squared"):
        TurboPufferVectorStoreConfig(distance_metric="dot_product")


def test_rag_core_config_composes_subsystem_configs() -> None:
    config = RAGCoreConfig(
        qdrant=QdrantConfig(url="http://localhost:6333", collection="product_docs"),
        embedding=EmbeddingConfig(model="text-embedding-3-small", dimensions=1536),
    )
    assert config.qdrant.url == "http://localhost:6333"
    assert config.qdrant.collection == "product_docs"
    assert config.embedding.model == "text-embedding-3-small"
    assert config.embedding.dimensions == 1536
    assert config.vector_store.provider == "qdrant"
    assert config.reranker.provider == "none"
    assert config.ingest.processing_version == DEFAULT_PROCESSING_VERSION


def test_rag_core_config_uses_default_subsystems_when_omitted() -> None:
    config = RAGCoreConfig()
    assert config.qdrant.url is None
    assert config.embedding.provider == "openai"
    assert config.vector_store.provider == "qdrant"
    assert config.reranker.provider == "none"
    assert config.ingest.source_type == "file"


def test_rag_core_config_propagates_qdrant_validation() -> None:
    with pytest.raises(ValueError, match="not both"):
        RAGCoreConfig(
            qdrant=QdrantConfig(url="http://localhost:6333", location=":memory:"),
        )


def test_rag_core_config_propagates_embedding_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        RAGCoreConfig(qdrant=QdrantConfig(), embedding=EmbeddingConfig(dimensions=0))
