from __future__ import annotations

import asyncio
import hashlib
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from rag_core.search.providers.qdrant_lifecycle import create_qdrant_collection
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from tests.support import TEST_API_SECRET

LOGGER_NAME = "rag_core.search.providers.qdrant_store"

SECRET = TEST_API_SECRET
PRIVATE_COLLECTION = f"docs-{SECRET}"
OTHER_PRIVATE_COLLECTION = f"archive-{SECRET}"


def _store(collection_name: str = PRIVATE_COLLECTION) -> QdrantVectorStore:
    return QdrantVectorStore(
        url=None,
        api_key=None,
        collection_name=collection_name,
        location=":memory:",
        dense_dimensions=3,
        quantization_enabled=True,
    )


def _collection_info() -> object:
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(size=3),
                sparse_vectors={"bm25": object(), "splade": object()},
            )
        ),
        points_count=7,
        status="green",
    )


class _ExistingCollectionClient:
    async def get_collections(self) -> object:
        return SimpleNamespace(collections=[SimpleNamespace(name=PRIVATE_COLLECTION)])

    async def get_collection(self, *, collection_name: str) -> object:
        assert collection_name == PRIVATE_COLLECTION
        return _collection_info()


class _CreateCollectionClient:
    def __init__(self) -> None:
        self.create_collection_calls: list[dict[str, object]] = []

    async def create_collection(self, **kwargs: object) -> None:
        self.create_collection_calls.append(kwargs)


class _FailingHealthClient:
    async def get_collection(self, *, collection_name: str) -> object:
        assert collection_name == PRIVATE_COLLECTION
        raise RuntimeError(f"connection failed for {PRIVATE_COLLECTION}")


def _messages(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


def _fingerprint(collection_name: str) -> str:
    return hashlib.sha256(collection_name.encode("utf-8")).hexdigest()[:12]


def _assert_private_context_omitted(
    caplog: pytest.LogCaptureFixture,
    message: str,
) -> None:
    assert PRIVATE_COLLECTION not in message
    assert OTHER_PRIVATE_COLLECTION not in message
    assert SECRET not in message
    assert "connection failed for" not in message
    assert "Traceback" not in message
    assert all(record.exc_info is None for record in caplog.records)


def test_qdrant_store_init_log_omits_collection_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        _store()

    message = _messages(caplog)
    assert "QdrantVectorStore initialized" in message
    assert "backend=qdrant" in message
    assert "dense_dimensions=3" in message
    assert "max_concurrent=" in message
    assert "max_batch_size=" in message
    assert "quantization=True" in message
    assert "local=True" in message
    assert f"collection_fingerprint={_fingerprint(PRIVATE_COLLECTION)}" in message
    _assert_private_context_omitted(caplog, message)


def test_collection_fingerprint_distinguishes_store_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        _store(PRIVATE_COLLECTION)
        _store(OTHER_PRIVATE_COLLECTION)

    message = _messages(caplog)
    assert f"collection_fingerprint={_fingerprint(PRIVATE_COLLECTION)}" in message
    assert f"collection_fingerprint={_fingerprint(OTHER_PRIVATE_COLLECTION)}" in message
    _assert_private_context_omitted(caplog, message)


def test_existing_collection_log_omits_collection_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def run() -> None:
        store = _store()
        store._client = cast(Any, _ExistingCollectionClient())

        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            await store.ensure_collection()

    asyncio.run(run())

    message = _messages(caplog)
    assert "Qdrant collection already exists" in message
    assert "backend=qdrant" in message
    assert "dense_dimensions=3" in message
    assert "sparse_channels=2" in message
    assert f"collection_fingerprint={_fingerprint(PRIVATE_COLLECTION)}" in message
    _assert_private_context_omitted(caplog, message)


def test_created_collection_log_omits_collection_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def run() -> None:
        client = _CreateCollectionClient()
        store = _store()
        store._client = cast(Any, client)

        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            await create_qdrant_collection(
                client=cast(Any, client),
                config=store._config,
                logger=logging.getLogger(LOGGER_NAME),
            )

        assert client.create_collection_calls[0]["collection_name"] == PRIVATE_COLLECTION

    asyncio.run(run())

    message = _messages(caplog)
    assert "Created Qdrant collection" in message
    assert "backend=qdrant" in message
    assert "dense_dimensions=3" in message
    assert "quantization=INT8" in message
    assert "hnsw_ef=100" in message
    assert f"collection_fingerprint={_fingerprint(PRIVATE_COLLECTION)}" in message
    _assert_private_context_omitted(caplog, message)


def test_health_failure_log_omits_collection_and_raw_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def run() -> dict[str, object]:
        store = _store()
        store._client = cast(Any, _FailingHealthClient())

        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            return await store.check_health()

    health = asyncio.run(run())

    assert health["collection"] == PRIVATE_COLLECTION
    assert health["error"] == "RuntimeError"
    message = _messages(caplog)
    assert "Qdrant health check failed" in message
    assert "backend=qdrant" in message
    assert "error_type=RuntimeError" in message
    assert f"collection_fingerprint={_fingerprint(PRIVATE_COLLECTION)}" in message
    _assert_private_context_omitted(caplog, message)
