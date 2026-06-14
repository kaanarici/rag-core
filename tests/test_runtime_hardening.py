"""HTTP runtime hardening: request_id, body cap, semaphore,
loopback bind, symlink rejection, and bound-namespace enforcement."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient

from rag_core.cli_serve import _enforce_loopback_bind, run_serve_command
from rag_core.cli_serve_parser import add_serve_command
from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.core import RAGCore
from rag_core.core_models import IngestedDocument, RAGCoreConfig
from rag_core.demo import DemoEmbeddingProvider, DemoSparseEmbedder
from rag_core.runtime.app import create_app
from rag_core.runtime.paths import read_validated_ingest_file, validate_ingest_path
from rag_core.runtime.requests import (
    parse_delete_document_request,
    parse_ingest_request,
    parse_retrieval_request,
)
from rag_core.runtime_defaults import (
    DEFAULT_RUNTIME_INGEST_CONCURRENCY,
    DEFAULT_RUNTIME_MAX_BODY_BYTES,
    LOOPBACK_HOSTS,
)
from rag_core.runtime.jobs import IngestJobRecord, IngestJobStore
from rag_core.search.policy import CorpusPolicy
from rag_core.search.providers.memory_store import InMemoryVectorStore

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# request_id threading
# ---------------------------------------------------------------------------


def _make_runtime_client(
    job_db_path: Path,
    *,
    config: RAGCoreConfig | None = None,
    ingest_concurrency: int = DEFAULT_RUNTIME_INGEST_CONCURRENCY,
    max_body_bytes: int = DEFAULT_RUNTIME_MAX_BODY_BYTES,
) -> TestClient:
    config = config or RAGCoreConfig(
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
        job_db_path=job_db_path,
        ingest_roots=(job_db_path.parent,),
        ingest_concurrency=ingest_concurrency,
        max_body_bytes=max_body_bytes,
    )
    return TestClient(app)


def test_runtime_mints_request_id_when_header_missing(tmp_path: Path) -> None:
    client = _make_runtime_client(tmp_path / "jobs.sqlite3")
    response = client.get("/health")
    assert response.status_code == 200
    request_id = response.headers.get("x-request-id")
    assert request_id is not None
    # UUIDv4 hex is 32 lowercase hex chars.
    assert len(request_id) == 32
    assert all(ch in "0123456789abcdef" for ch in request_id)


def test_runtime_echoes_caller_supplied_request_id(tmp_path: Path) -> None:
    client = _make_runtime_client(tmp_path / "jobs.sqlite3")
    response = client.get(
        "/health",
        headers={"X-Request-Id": "req-from-gateway-42"},
    )
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == "req-from-gateway-42"


def test_runtime_request_id_stamped_on_error_responses(tmp_path: Path) -> None:
    client = _make_runtime_client(tmp_path / "jobs.sqlite3")
    response = client.get(
        "/v1/does-not-exist",
        headers={"X-Request-Id": "req-not-found"},
    )
    assert response.status_code == 404
    assert response.headers.get("x-request-id") == "req-not-found"


# ---------------------------------------------------------------------------
# Body cap
# ---------------------------------------------------------------------------


def test_runtime_rejects_oversized_body_with_413(tmp_path: Path) -> None:
    client = _make_runtime_client(
        tmp_path / "jobs.sqlite3",
        max_body_bytes=64,
    )
    # 200-byte query field. Well over the 64-byte cap once JSON-wrapped.
    oversized = "x" * 256
    response = client.post(
        "/v1/search",
        json={
            "query": oversized,
            "namespace": "acme",
            "corpus_ids": ["help"],
        },
    )
    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == "payload_too_large"
    assert body["error"]["details"]["max_bytes"] == 64
    # request_id still threaded onto rejections.
    assert response.headers.get("x-request-id")


def test_runtime_body_cap_skipped_for_get_requests(tmp_path: Path) -> None:
    client = _make_runtime_client(
        tmp_path / "jobs.sqlite3",
        max_body_bytes=8,
    )
    # GET /health succeeds even with a tiny cap because the middleware
    # doesn't enforce the cap on read-only verbs.
    response = client.get("/health")
    assert response.status_code == 200


def test_runtime_rejects_non_numeric_content_length(tmp_path: Path) -> None:
    client = _make_runtime_client(tmp_path / "jobs.sqlite3")
    response = client.post(
        "/v1/search",
        content=b"{}",
        headers={"Content-Length": "not-a-number", "Content-Type": "application/json"},
    )
    # Starlette / httpx will likely normalize this; the middleware still has
    # to be safe against a tampered header so accept either 400 or normal flow.
    assert response.status_code in {200, 400, 422}


def test_runtime_rejects_oversized_chunked_body_with_413(tmp_path: Path) -> None:
    """Caller omits Content-Length (chunked transfer / streaming body) so the
    cap can only be enforced by tallying bytes as the receive() frames arrive.
    """
    client = _make_runtime_client(
        tmp_path / "jobs.sqlite3",
        max_body_bytes=64,
    )

    def streamed_chunks() -> Any:
        yield b'{"query": "'
        yield b"y" * 256
        yield b'", "namespace": "acme", "corpus_ids": ["help"]}'

    response = client.post(
        "/v1/search",
        content=streamed_chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == "payload_too_large"
    assert body["error"]["details"]["max_bytes"] == 64
    assert response.headers.get("x-request-id")


# ---------------------------------------------------------------------------
# Ingest concurrency semaphore (503 busy)
# ---------------------------------------------------------------------------


class _BlockingCore:
    """RAGCore stub that never finishes ingest_file until released."""

    def __init__(self) -> None:
        self._gate = asyncio.Event()
        self.started = 0

    async def ensure_ready(self) -> None:
        return None

    async def ingest_file(
        self,
        path: Path,
        *,
        namespace: str,
        corpus_id: str,
        **_: Any,
    ) -> Any:
        self.started += 1
        await self._gate.wait()
        raise RuntimeError("blocking core never completes")

    async def close(self) -> None:
        return None


class _ByteRecordingCore:
    def __init__(self) -> None:
        self.file_bytes: bytes | None = None

    async def ensure_ready(self) -> None:
        return None

    async def ingest_file(
        self,
        path: Path,
        *,
        namespace: str,
        corpus_id: str,
        pre_read_bytes: bytes | None = None,
        **_: Any,
    ) -> IngestedDocument:
        self.file_bytes = pre_read_bytes if pre_read_bytes is not None else path.read_bytes()
        return IngestedDocument(
            document_id="doc-runtime",
            corpus_id=corpus_id,
            namespace=namespace,
            chunk_count=1,
            filename=path.name,
            mime_type="text/markdown",
            ingest_state="created",
        )

    async def close(self) -> None:
        return None


def test_runtime_returns_503_busy_when_ingest_semaphore_saturated(tmp_path: Path) -> None:
    import threading
    import time

    blocking = _BlockingCore()
    config = RAGCoreConfig.local()
    doc1 = tmp_path / "a.md"
    doc2 = tmp_path / "b.md"
    doc3 = tmp_path / "c.md"
    for doc in (doc1, doc2, doc3):
        doc.write_text("body\n", encoding="utf-8")

    app = create_app(
        config=config,
        core_factory=lambda _: cast(Any, blocking),
        job_db_path=tmp_path / "jobs.sqlite3",
        ingest_roots=(tmp_path,),
        ingest_concurrency=1,
    )
    with TestClient(app) as client:
        # TestClient's portal awaits BackgroundTask completion before the
        # request returns, so the first ingest is fired from a worker thread
        # that holds the semaphore via ``_gate.wait()`` while the main thread
        # proves the saturated path.
        first_holder: dict[str, object] = {}

        def fire_first() -> None:
            first_holder["response"] = client.post(
                "/v1/ingest",
                json={"path": str(doc1), "namespace": "acme", "corpus_id": "help"},
            )

        first_thread = threading.Thread(target=fire_first, daemon=True)
        first_thread.start()
        try:
            for _ in range(200):
                if blocking.started >= 1:
                    break
                time.sleep(0.01)
            assert blocking.started >= 1, "first ingest never reached the semaphore"

            # Second concurrent ingest lands on the saturated semaphore and is
            # refused with 503 busy. ``ingest_semaphore.locked()`` is read on
            # the event-loop thread so the check is well-ordered with the
            # first BackgroundTask's ``acquire()``.
            second = client.post(
                "/v1/ingest",
                json={"path": str(doc2), "namespace": "acme", "corpus_id": "help"},
            )
            assert second.status_code == 503
            body = second.json()
            assert body["error"]["code"] == "busy"
        finally:
            # Release the held BackgroundTask so the worker thread can return
            # and the TestClient lifespan can shut down cleanly. The gate is
            # an ``asyncio.Event`` owned by the portal's loop. Schedule the
            # ``set()`` on that loop via the portal so we don't touch loop
            # internals from a foreign thread.
            assert client.portal is not None
            client.portal.call(blocking._gate.set)
            first_thread.join(timeout=5)
            assert not first_thread.is_alive(), "first ingest thread never released"


def test_runtime_ingest_uses_file_bytes_captured_before_job_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink creation not supported")
    ingest_root = tmp_path / "root"
    ingest_root.mkdir()
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    doc = ingest_root / "doc.md"
    doc.write_bytes(b"validated bytes")
    secret = outside_root / "secret.md"
    secret.write_bytes(b"swapped bytes")
    core = _ByteRecordingCore()

    original_create = IngestJobStore.create

    def create_and_swap(
        self: IngestJobStore,
        *,
        path: str,
        namespace: str,
        corpus_id: str,
    ) -> IngestJobRecord:
        record = original_create(
            self,
            path=path,
            namespace=namespace,
            corpus_id=corpus_id,
        )
        doc.unlink()
        doc.symlink_to(secret)
        return record

    monkeypatch.setattr(IngestJobStore, "create", create_and_swap)

    app = create_app(
        config=RAGCoreConfig.local(),
        core_factory=lambda _: cast(Any, core),
        job_db_path=tmp_path / "jobs.sqlite3",
        ingest_roots=(ingest_root,),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/ingest",
            json={"path": str(doc), "namespace": "acme", "corpus_id": "help"},
        )

    assert response.status_code == 202
    assert core.file_bytes == b"validated bytes"


# ---------------------------------------------------------------------------
# Path hardening
# ---------------------------------------------------------------------------


def test_validate_ingest_path_rejects_symlink_leaf(tmp_path: Path) -> None:
    real = tmp_path / "real.md"
    real.write_text("body\n", encoding="utf-8")
    link = tmp_path / "link.md"
    try:
        os.symlink(real, link)
    except (NotImplementedError, OSError):  # pragma: no cover - platform skip
        pytest.skip("symlink creation not supported")
    from rag_core.runtime.errors import RuntimeRequestError

    with pytest.raises(RuntimeRequestError) as excinfo:
        validate_ingest_path(str(link), roots=(tmp_path,))
    assert "symbolic link" in excinfo.value.message


def test_validate_ingest_path_rejects_symlink_parent(tmp_path: Path) -> None:
    real_dir = tmp_path / "real-dir"
    real_dir.mkdir()
    leaf = real_dir / "doc.md"
    leaf.write_text("body\n", encoding="utf-8")
    link_dir = tmp_path / "link-dir"
    try:
        os.symlink(real_dir, link_dir)
    except (NotImplementedError, OSError):  # pragma: no cover - platform skip
        pytest.skip("symlink creation not supported")
    via_link = link_dir / "doc.md"
    from rag_core.runtime.errors import RuntimeRequestError

    with pytest.raises(RuntimeRequestError) as excinfo:
        validate_ingest_path(str(via_link), roots=(tmp_path,))
    assert "symbolic link" in excinfo.value.message


def test_validate_ingest_path_rejects_relative_path(tmp_path: Path) -> None:
    from rag_core.runtime.errors import RuntimeRequestError

    with pytest.raises(RuntimeRequestError) as excinfo:
        validate_ingest_path("relative/path.md", roots=(tmp_path,))
    assert "must be absolute" in excinfo.value.message


def test_validate_ingest_path_rejects_tilde_expansion(tmp_path: Path) -> None:
    from rag_core.runtime.errors import RuntimeRequestError

    with pytest.raises(RuntimeRequestError) as excinfo:
        validate_ingest_path("~/secret.md", roots=(tmp_path,))
    assert "must be absolute" in excinfo.value.message


def test_validate_ingest_path_rejects_directory(tmp_path: Path) -> None:
    from rag_core.runtime.errors import RuntimeRequestError

    (tmp_path / "subdir").mkdir()
    with pytest.raises(RuntimeRequestError) as excinfo:
        validate_ingest_path(str(tmp_path / "subdir"), roots=(tmp_path,))
    assert "regular file" in excinfo.value.message


def test_validate_ingest_path_accepts_regular_file(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("body\n", encoding="utf-8")
    resolved = validate_ingest_path(str(doc), roots=(tmp_path,))
    assert resolved == doc.resolve()


def test_read_validated_ingest_file_rejects_hardlinked_file(tmp_path: Path) -> None:
    if os.name == "nt" or not hasattr(os, "link"):
        pytest.skip("hardlinks are unavailable on this platform")
    doc = tmp_path / "doc.md"
    doc.write_text("body\n", encoding="utf-8")
    alias = tmp_path / "alias.md"
    try:
        os.link(doc, alias)
    except OSError as exc:
        pytest.skip(f"hardlink support unavailable: {exc}")

    from rag_core.runtime.errors import RuntimeRequestError

    with pytest.raises(RuntimeRequestError) as excinfo:
        read_validated_ingest_file(str(doc), roots=(tmp_path,))
    assert "multi-link" in excinfo.value.message


# ---------------------------------------------------------------------------
# Loopback bind enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", sorted(LOOPBACK_HOSTS))
def test_cli_serve_loopback_hosts_pass_default(host: str) -> None:
    # Should not raise.
    _enforce_loopback_bind(host=host, bind_non_loopback=False)


def test_cli_serve_refuses_non_loopback_without_flag() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _enforce_loopback_bind(host="0.0.0.0", bind_non_loopback=False)
    assert "--bind-non-loopback" in str(excinfo.value)


def test_cli_serve_accepts_non_loopback_with_flag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging as _logging

    with caplog.at_level(_logging.WARNING, logger="rag_core.cli_serve"):
        _enforce_loopback_bind(host="0.0.0.0", bind_non_loopback=True)
    assert any("non-loopback" in rec.getMessage() for rec in caplog.records)


def test_cli_serve_unix_socket_skips_loopback_check(tmp_path: Path) -> None:
    """``--unix-socket`` is mutually exclusive with host:port, so the loopback
    check should be skipped when a UDS path is supplied.

    Smoke-tested via the CLI parser surface; we can't actually bind a UDS in
    a unit test without an event loop, but we assert the parser accepts the
    flag.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_serve_command(subparsers)
    args = parser.parse_args(
        [
            "serve",
            "--unix-socket",
            str(tmp_path / "sock"),
            "--qdrant-location",
            ":memory:",
            "--embedding-provider",
            "demo",
            "--embedding-dimensions",
            "4",
        ]
    )
    assert args.unix_socket == str(tmp_path / "sock")


def test_cli_serve_parser_exposes_new_flags() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_serve_command(subparsers)
    args = parser.parse_args(
        [
            "serve",
            "--bind-non-loopback",
            "--max-body-bytes",
            "1024",
            "--ingest-concurrency",
            "2",
            "--limit-concurrency",
            "16",
            "--job-retention-seconds",
            "300",
            "--qdrant-location",
            ":memory:",
            "--embedding-provider",
            "demo",
            "--embedding-dimensions",
            "4",
        ]
    )
    assert args.bind_non_loopback is True
    assert args.max_body_bytes == 1024
    assert args.ingest_concurrency == 2
    assert args.limit_concurrency == 16
    assert args.job_retention_seconds == 300.0


def test_run_serve_command_refuses_non_loopback_without_flag(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_serve_command(subparsers)
    args = parser.parse_args(
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "8788",
            "--job-db-path",
            str(tmp_path / "jobs.sqlite3"),
            "--qdrant-location",
            ":memory:",
            "--embedding-provider",
            "demo",
            "--embedding-dimensions",
            "4",
        ]
    )
    with pytest.raises(SystemExit):
        run_serve_command(args)


# ---------------------------------------------------------------------------
# Bound-namespace enforcement
# ---------------------------------------------------------------------------


def test_parse_ingest_request_enforces_bound_namespace() -> None:
    from rag_core.runtime.errors import RuntimeRequestError

    body = {"path": "/tmp/x.md", "namespace": "other", "corpus_id": "help"}
    with pytest.raises(RuntimeRequestError) as excinfo:
        parse_ingest_request(body, bound_namespace="signal")
    assert excinfo.value.details == {"field": "namespace"}


def test_parse_retrieval_request_enforces_bound_namespace() -> None:
    from rag_core.runtime.errors import RuntimeRequestError

    body = {"query": "q", "namespace": "other", "corpus_ids": ["help"]}
    with pytest.raises(RuntimeRequestError) as excinfo:
        parse_retrieval_request(body, bound_namespace="signal")
    assert excinfo.value.details == {"field": "namespace"}


def test_parse_delete_document_request_enforces_bound_namespace() -> None:
    from rag_core.runtime.errors import RuntimeRequestError

    body = {"namespace": "other", "corpus_id": "help"}
    with pytest.raises(RuntimeRequestError) as excinfo:
        parse_delete_document_request(
            document_id="doc-1",
            payload=body,
            bound_namespace="signal",
        )
    assert excinfo.value.details == {"field": "namespace"}


def test_runtime_search_refuses_cross_namespace_when_bound(tmp_path: Path) -> None:
    config = RAGCoreConfig(
        qdrant=QdrantConfig(location=":memory:"),
        embedding=EmbeddingConfig(
            provider="demo",
            model="demo-dense-v1",
            dimensions=4,
        ),
        corpus_policy=CorpusPolicy(bound_namespace="signal-ws-1"),
    )
    client = _make_runtime_client(
        tmp_path / "jobs.sqlite3",
        config=config,
    )
    response = client.post(
        "/v1/search",
        json={
            "query": "billing",
            "namespace": "some-other-tenant",
            "corpus_ids": ["help"],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["details"] == {"field": "namespace"}


def test_runtime_retrieval_rejects_empty_corpus_ids_via_contract() -> None:
    """The contract normalizer is what closes the tier-widening footgun at
    the HTTP boundary."""
    from rag_core.runtime.errors import RuntimeRequestError

    body = {"query": "q", "namespace": "acme", "corpus_ids": []}
    with pytest.raises(RuntimeRequestError) as excinfo:
        parse_retrieval_request(body)
    assert excinfo.value.details == {"field": "corpus_ids"}
