"""Subsystem configs (Qdrant, Embedding, Reranker, Chunking, Ingest) own their own validation.

``Config`` is the composition of these typed subsystem configs.
"""

from __future__ import annotations

import pytest

from rag_core import Config
from rag_core.config import (
    ChunkingConfig,
    ContextualizerConfig,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_INGEST_SOURCE_TYPE,
    DEFAULT_RERANKER_PROVIDER,
    DEFAULT_VECTOR_STORE_PROVIDER,
    EmbeddingConfig,
    IngestConfig,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_PROVIDER,
    QdrantConfig,
    RerankerConfig,
    TurboPufferVectorStoreConfig,
    VectorStoreConfig,
)
from rag_core.core_models import DEFAULT_PROCESSING_VERSION
from rag_core.documents.contextualizer_provider_names import ANTHROPIC_CONTEXTUALIZER_ID


def test_qdrant_config_defaults_allow_injected_vector_store() -> None:
    config = QdrantConfig()
    assert config.url is None
    assert config.location is None
    assert config.api_key is None


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
    with pytest.raises(ValueError) as exc_info:
        QdrantConfig(url="http://localhost:6333", location=":memory:")

    message = str(exc_info.value)
    assert "got both" in message
    assert "use QdrantConfig(location=':memory:')" in message
    assert "pass url=..." in message
    assert "inject vector_store=..." in message


def test_qdrant_config_normalizes_blank_api_key() -> None:
    assert QdrantConfig(location=":memory:", api_key="   ").api_key is None
    assert QdrantConfig(location=":memory:", api_key=" key ").api_key == "key"


def test_embedding_config_defaults() -> None:
    config = EmbeddingConfig()
    assert config.provider == DEFAULT_EMBEDDING_PROVIDER
    assert config.model == DEFAULT_EMBEDDING_MODEL
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


def test_embedding_config_accepts_https_base_url() -> None:
    config = EmbeddingConfig(base_url="https://api.openai.com/v1")
    assert config.base_url == "https://api.openai.com/v1"


def test_embedding_config_normalizes_blank_base_url_to_none() -> None:
    assert EmbeddingConfig(base_url="   ").base_url is None


@pytest.mark.parametrize(
    ("base_url", "match"),
    (
        # http:// is rejected at config time, not deferred to first request.
        ("http://api.openai.com/v1", "http"),
        # private-IP literals fail closed without explicit opt-in.
        ("https://127.0.0.1/v1", "safe https URL"),
        ("https://10.0.0.1/v1", "safe https URL"),
        # embedded credentials must never reach the SDK.
        ("https://user:pass@api.example.com/v1", "safe https URL"),
        # malformed URL.
        ("not-a-url", "safe https URL"),
    ),
)
def test_embedding_config_rejects_unsafe_base_url_at_construction(
    base_url: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        EmbeddingConfig(base_url=base_url)


def test_embedding_config_region_pin_accepts_matching_host() -> None:
    config = EmbeddingConfig(
        base_url="https://eu-west-1.api.example.com/v1",
        region="eu-west-1",
    )
    assert config.region == "eu-west-1"


def test_embedding_config_region_pin_refuses_cross_region_host() -> None:
    with pytest.raises(ValueError, match="region pin"):
        EmbeddingConfig(
            base_url="https://us-east-1.api.example.com/v1",
            region="eu-west-1",
        )


def test_embedding_config_region_without_base_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires base_url"):
        EmbeddingConfig(region="us-east-1")


def test_reranker_config_defaults() -> None:
    config = RerankerConfig()
    assert config.provider == DEFAULT_RERANKER_PROVIDER
    assert config.model is None
    assert config.api_key is None
    assert config.strict_provider is False


def test_contextualizer_config_defaults_disabled() -> None:
    config = ContextualizerConfig()
    assert config.provider == ANTHROPIC_CONTEXTUALIZER_ID
    assert config.model is None
    assert config.enabled is False
    assert config.contextualizer_chunk_cap is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"provider": ""}, "non-empty"),
        ({"provider": "   "}, "non-empty"),
        ({"enabled": "yes"}, "boolean"),
        ({"model": 1}, "model"),
        ({"contextualizer_chunk_cap": 0}, "positive"),
        ({"contextualizer_chunk_cap": True}, "positive"),
    ),
)
def test_contextualizer_config_rejects_invalid_inputs(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ContextualizerConfig(**kwargs)  # type: ignore[arg-type]


def test_contextualizer_config_normalizes_provider_and_blank_model() -> None:
    config = ContextualizerConfig(provider=" Anthropic ", model="   ")
    assert config.provider == ANTHROPIC_CONTEXTUALIZER_ID
    assert config.model is None


def test_chunking_config_constructs_with_defaults() -> None:
    # Empty by design; chunking is selected by router, not config flags.
    ChunkingConfig()


def test_ingest_config_defaults() -> None:
    config = IngestConfig()
    assert config.processing_version == DEFAULT_PROCESSING_VERSION
    assert config.source_type == DEFAULT_INGEST_SOURCE_TYPE
    assert config.enable_lexical_search is False


def test_vector_store_config_defaults_to_qdrant() -> None:
    config = VectorStoreConfig()
    assert config.provider == DEFAULT_VECTOR_STORE_PROVIDER


def test_vector_store_config_normalizes_provider() -> None:
    assert VectorStoreConfig(provider=" Qdrant ").provider == "qdrant"


def test_vector_store_config_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="qdrant"):
        VectorStoreConfig(provider="unknown")


def test_turbopuffer_config_accepts_https_base_url() -> None:
    config = TurboPufferVectorStoreConfig(
        base_url="https://gcp-us-east1.turbopuffer.com"
    )
    assert config.base_url == "https://gcp-us-east1.turbopuffer.com"


def test_turbopuffer_config_rejects_http_base_url() -> None:
    with pytest.raises(ValueError, match="safe https URL"):
        TurboPufferVectorStoreConfig(base_url="http://turbopuffer.local")


def test_turbopuffer_config_rejects_private_address_base_url() -> None:
    with pytest.raises(ValueError, match="safe https URL"):
        TurboPufferVectorStoreConfig(base_url="https://127.0.0.1/tpuf")


def test_turbopuffer_config_region_pin_matches_host() -> None:
    config = TurboPufferVectorStoreConfig(
        region="gcp-us-east1",
        base_url="https://gcp-us-east1.turbopuffer.com",
    )
    assert config.region == "gcp-us-east1"


def test_turbopuffer_config_region_pin_refuses_cross_region_host() -> None:
    with pytest.raises(ValueError, match="region pin"):
        TurboPufferVectorStoreConfig(
            region="gcp-us-east1",
            base_url="https://aws-eu-west1.turbopuffer.com",
        )


def test_rag_core_config_composes_subsystem_configs() -> None:
    config = Config(
        qdrant=QdrantConfig(url="http://localhost:6333", store_collection="product_docs"),
        embedding=EmbeddingConfig(model="text-embedding-3-small", dimensions=1536),
    )
    assert config.qdrant.url == "http://localhost:6333"
    assert config.qdrant.store_collection == "product_docs"
    assert config.embedding.model == "text-embedding-3-small"
    assert config.embedding.dimensions == 1536
    assert config.vector_store.provider == DEFAULT_VECTOR_STORE_PROVIDER
    assert config.reranker.provider == DEFAULT_RERANKER_PROVIDER
    assert config.contextualizer.enabled is False
    assert config.ingest.processing_version == DEFAULT_PROCESSING_VERSION


def test_rag_core_config_uses_default_subsystems_when_omitted() -> None:
    config = Config()
    assert config.qdrant.url is None
    assert config.qdrant.location is None


def test_store_factory_teaches_when_qdrant_target_missing() -> None:
    from rag_core.search.providers.qdrant_store import create_qdrant_client

    with pytest.raises(ValueError) as exc_info:
        create_qdrant_client(url=None, api_key=None, location=None)

    message = str(exc_info.value)
    assert "use QdrantConfig(location=':memory:')" in message
    assert "pass url=..." in message
    assert "inject vector_store=..." in message


def test_rag_core_config_local_factory_builds_no_key_memory_config() -> None:
    config = Config.local()
    assert config.qdrant.location == ":memory:"
    assert config.qdrant.url is None
    assert config.embedding.provider == LOCAL_EMBEDDING_PROVIDER
    assert config.embedding.model == LOCAL_EMBEDDING_MODEL
    assert config.embedding.dimensions is None
    assert config.vector_store.provider == DEFAULT_VECTOR_STORE_PROVIDER
    assert config.reranker.provider == DEFAULT_RERANKER_PROVIDER
    assert config.contextualizer.enabled is False
    assert config.ingest.source_type == DEFAULT_INGEST_SOURCE_TYPE


def test_rag_core_config_local_factory_accepts_persist_dir(tmp_path) -> None:
    config = Config.local(persist_dir=tmp_path / "qdrant")
    assert config.qdrant.location == str(tmp_path / "qdrant")


def test_rag_core_config_qdrant_factory_uses_known_model_without_dimensions() -> None:
    config = Config.qdrant(
        url="http://localhost:6333",
        embedding_provider=LOCAL_EMBEDDING_PROVIDER,
    )
    assert config.qdrant.url == "http://localhost:6333"
    assert config.qdrant.location is None
    assert config.embedding.provider == LOCAL_EMBEDDING_PROVIDER
    assert config.embedding.model == LOCAL_EMBEDDING_MODEL
    assert config.embedding.dimensions is None
    assert config.contextualizer.enabled is False


def test_rag_core_config_qdrant_factory_requires_model_for_other_providers() -> None:
    with pytest.raises(ValueError, match="needs an explicit model"):
        Config.qdrant(url="http://localhost:6333", embedding_provider="voyage")

    config = Config.qdrant(
        url="http://localhost:6333",
        embedding_provider="voyage",
        model="voyage-4",
    )
    assert config.embedding.model == "voyage-4"


def test_rag_core_config_propagates_qdrant_validation() -> None:
    with pytest.raises(ValueError, match="got both"):
        Config(
            qdrant=QdrantConfig(url="http://localhost:6333", location=":memory:"),
        )


def test_rag_core_config_propagates_embedding_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        Config(
            qdrant=QdrantConfig(location=":memory:"),
            embedding=EmbeddingConfig(dimensions=0),
        )
