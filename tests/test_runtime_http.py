from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient

from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.core import RAGCore
from rag_core.core_models import RAGCoreConfig
from rag_core.demo import DemoEmbeddingProvider, DemoSparseEmbedder
from rag_core.runtime.app import create_app
from rag_core.search.providers.embedding import create_embedding_provider
from rag_core.search.providers.memory_store import InMemoryVectorStore


@pytest.fixture
def runtime_client(tmp_path: Path) -> TestClient:
    config = RAGCoreConfig(
        qdrant=QdrantConfig(location=":memory:"),
        embedding=EmbeddingConfig(
            provider="demo",
            model="demo-dense-v1",
            dimensions=4,
        ),
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


def test_demo_embedding_provider_is_registered_for_serve() -> None:
    provider = create_embedding_provider(provider="demo", dimensions=8)
    assert provider.model_name == "demo-dense-v1"
    assert provider.dimensions == 8


def test_runtime_health_and_runtime_endpoints(runtime_client: TestClient) -> None:
    health = runtime_client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"ok": True, "status": "ok", "live": True}

    ready = runtime_client.get("/health/ready")
    assert ready.status_code == 200
    ready_payload = ready.json()
    assert ready_payload["ready"] is True
    assert ready_payload["checks"]["core"]["status"] == "ok"
    assert ready_payload["checks"]["vector_store"]["healthy"] is True

    runtime = runtime_client.get("/v1/runtime")
    assert runtime.status_code == 200
    payload = runtime.json()
    assert "collection_name" in payload


def test_runtime_api_error_shape(runtime_client: TestClient) -> None:
    missing_fields = runtime_client.post("/v1/search", json={"query": "billing"})
    assert missing_fields.status_code == 400
    body = missing_fields.json()
    assert body["error"]["code"] == "invalid_request"
    assert "message" in body["error"]

    invalid_json = runtime_client.post(
        "/v1/search",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert invalid_json.status_code == 400
    assert invalid_json.json()["error"]["code"] == "invalid_json"

    missing_job = runtime_client.get("/v1/ingest/does-not-exist")
    assert missing_job.status_code == 404
    assert missing_job.json()["error"]["code"] == "not_found"
    assert missing_job.json()["error"]["details"]["job_id"] == "does-not-exist"


def test_runtime_unknown_route_returns_api_error(runtime_client: TestClient) -> None:
    response = runtime_client.get("/v1/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def _wait_for_job(client: TestClient, job_id: str, *, timeout_s: float = 5.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = client.get(f"/v1/ingest/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        status = payload.get("status")
        if status == "completed":
            return dict(payload)
        if status == "failed":
            pytest.fail(f"ingest job failed: {payload}")
        time.sleep(0.05)
    pytest.fail(f"ingest job timed out: {job_id}")


def test_runtime_ingest_search_and_retrieve_context_journey(
    runtime_client: TestClient,
    tmp_path: Path,
) -> None:
    doc = tmp_path / "billing.md"
    doc.write_text(
        "Invoices are due monthly. Customers can pay by card, ACH, or wire transfer.\n",
        encoding="utf-8",
    )

    created = runtime_client.post(
        "/v1/ingest",
        json={
            "path": str(doc),
            "namespace": "acme",
            "corpus_id": "help",
        },
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    finished = _wait_for_job(runtime_client, job_id)
    result = finished.get("result")
    assert isinstance(result, dict)
    assert result.get("chunk_count", 0) > 0

    search = runtime_client.post(
        "/v1/search",
        json={
            "query": "How can invoices be paid?",
            "namespace": "acme",
            "corpus_ids": ["help"],
            "limit": 5,
        },
    )
    assert search.status_code == 200
    hits = search.json()
    assert isinstance(hits, list)
    assert hits
    assert "text" in hits[0]
    assert "document_id" in hits[0]

    context = runtime_client.post(
        "/v1/retrieve-context",
        json={
            "query": "invoice payment methods",
            "namespace": "acme",
            "corpus_ids": ["help"],
            "limit": 5,
        },
    )
    assert context.status_code == 200
    pack = context.json()
    assert isinstance(pack.get("context_text"), str)
    assert pack["context_text"]


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
