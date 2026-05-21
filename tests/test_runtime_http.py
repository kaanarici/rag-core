from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient

from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.core import RAGCore
from rag_core.core_models import RAGCoreConfig
from rag_core.demo import DemoEmbeddingProvider, DemoSparseEmbedder
from rag_core.runtime.app import create_app
from rag_core.search.providers.memory_store import InMemoryVectorStore


@pytest.fixture
def runtime_client(tmp_path: Path) -> TestClient:
    config = RAGCoreConfig(
        qdrant=QdrantConfig(location=":memory:"),
        embedding=EmbeddingConfig(model="demo", dimensions=4),
    )
    core = RAGCore(
        config,
        embedding_provider=DemoEmbeddingProvider(dimensions=4),
        sparse_embedder=DemoSparseEmbedder(),
        vector_store=InMemoryVectorStore(),
    )

    def core_factory(cfg: RAGCoreConfig) -> RAGCore:
        assert cfg is config
        return core

    app = create_app(
        config=config,
        core_factory=core_factory,
        job_db_path=tmp_path / "jobs.sqlite3",
    )
    return TestClient(app)


def test_runtime_health_and_runtime_endpoints(runtime_client: TestClient) -> None:
    health = runtime_client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"ok": True}

    runtime = runtime_client.get("/v1/runtime")
    assert runtime.status_code == 200
    payload = runtime.json()
    assert "collection_name" in payload


def test_runtime_search_returns_hit_list(runtime_client: TestClient) -> None:
    response = runtime_client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "acme",
            "corpus_ids": ["help"],
            "limit": 5,
        },
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)
