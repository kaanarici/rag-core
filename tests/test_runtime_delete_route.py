"""HTTP runtime tests for the right-to-forget DELETE route.

the caller's gateway calls ``DELETE /v1/documents/{document_id}`` to purge
a single document across vector store, lexical sidecar, embedding cache,
chunk-context cache, and manifest. The route fails closed when ``namespace``
or ``corpus_id`` is missing, the result body reports honest per-surface
state, and the canonical retrieval surface (vector store) is purged before
any other side effect runs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient

from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.core import RAGCore
from rag_core.core_models import RAGCoreConfig
from rag_core.demo import DemoEmbeddingProvider, DemoSparseEmbedder
from rag_core.runtime.app import create_app
from rag_core.search.policy import CorpusPolicy
from rag_core.search.providers.memory_store import InMemoryVectorStore

pytestmark = [pytest.mark.integration]


def _make_runtime_client(tmp_path: Path) -> tuple[TestClient, RAGCore]:
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
        ingest_roots=(tmp_path,),
    )
    return TestClient(app), core


def _ingest_one(core: RAGCore) -> str:
    async def _run() -> str:
        document = await core.ingest_bytes(
            file_bytes=b"hello world",
            filename="doc.md",
            mime_type="text/markdown",
            namespace="workspace-alpha",
            corpus_id="public",
            path="/doc.md",
        )
        return document.document_id

    return asyncio.run(_run())


def test_delete_route_returns_honest_per_surface_result(tmp_path: Path) -> None:
    client, core = _make_runtime_client(tmp_path)
    try:
        document_id = _ingest_one(core)
        response = client.request(
            "DELETE",
            f"/v1/documents/{document_id}",
            json={"namespace": "workspace-alpha", "corpus_id": "public"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["document_id"] == document_id
        assert body["namespace"] == "workspace-alpha"
        assert body["corpus_id"] == "public"
        # Vector store ack drives both index_deleted and vector_store_acked.
        assert body["index_deleted"] is True
        assert body["vector_store_acked"] is True
        # No sidecar / cache wired in this runtime build -> honest None.
        assert body["lexical_sidecar_purged"] is None
        assert body["embedding_cache_purged"] is None
        assert body["chunk_context_cache_purged"] is None
        # No manifest directory configured for this in-memory runtime.
        assert body["manifest_entry_deleted"] is None
        assert body["manifest_removed"] is None
    finally:
        asyncio.run(core.close())


def test_delete_route_requires_namespace_and_corpus_id(tmp_path: Path) -> None:
    client, core = _make_runtime_client(tmp_path)
    try:
        response = client.request(
            "DELETE",
            "/v1/documents/doc-1",
            json={"namespace": "workspace-alpha"},
        )
        assert response.status_code == 400
        details = response.json()["error"]["details"]
        assert "corpus_id" in details["missing_fields"]

        response = client.request(
            "DELETE",
            "/v1/documents/doc-1",
            json={"corpus_id": "public"},
        )
        assert response.status_code == 400
        details = response.json()["error"]["details"]
        assert "namespace" in details["missing_fields"]
    finally:
        asyncio.run(core.close())


def test_delete_route_rejects_blank_document_id(tmp_path: Path) -> None:
    client, core = _make_runtime_client(tmp_path)
    try:
        # Starlette path parameters never match an empty path segment, but a
        # whitespace-only document_id used to slip through. The contract
        # parser must refuse it at the seam so right-to-forget cannot widen
        # silently.
        response = client.request(
            "DELETE",
            "/v1/documents/%20",
            json={"namespace": "workspace-alpha", "corpus_id": "public"},
        )
        assert response.status_code == 400
        body = response.json()
        assert "document_id" in body["error"]["details"]["missing_fields"]
    finally:
        asyncio.run(core.close())


def test_delete_route_rejects_unknown_fields(tmp_path: Path) -> None:
    client, core = _make_runtime_client(tmp_path)
    try:
        response = client.request(
            "DELETE",
            "/v1/documents/doc-1",
            json={
                "namespace": "workspace-alpha",
                "corpus_id": "public",
                "force": True,
            },
        )
        assert response.status_code == 400
        assert "force" in response.json()["error"]["details"]["fields"]
    finally:
        asyncio.run(core.close())


def test_delete_route_enforces_bound_namespace(tmp_path: Path) -> None:
    """When the process is bound to a single tenant, a DELETE that names a
    different namespace is refused at the HTTP boundary. Symmetric with
    /v1/ingest and /v1/search."""
    config = RAGCoreConfig(
        qdrant=QdrantConfig(location=":memory:"),
        embedding=EmbeddingConfig(
            provider="demo",
            model="demo-dense-v1",
            dimensions=4,
        ),
        corpus_policy=CorpusPolicy(bound_namespace="workspace-alpha"),
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
        ingest_roots=(tmp_path,),
    )
    client = TestClient(app)
    try:
        response = client.request(
            "DELETE",
            "/v1/documents/doc-1",
            json={"namespace": "workspace-omega", "corpus_id": "public"},
        )
        assert response.status_code == 400
        body = response.json()
        assert body["error"]["code"] == "invalid_request"
        assert body["error"]["details"] == {"field": "namespace"}
    finally:
        asyncio.run(core.close())
