from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import pytest
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore


class _FailingQdrantClient:
    async def get_collection(self, *, collection_name: str) -> object:
        raise RuntimeError("private qdrant detail")

    async def close(self) -> None:
        return None


class _FailingTurbopufferNamespace:
    async def metadata(self) -> object:
        raise RuntimeError("private turbopuffer detail")


def test_qdrant_health_omits_backend_exception_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def run() -> None:
        store = QdrantVectorStore(
            url=None,
            api_key=None,
            collection_name="docs",
            location=":memory:",
            dense_dimensions=3,
        )
        store._client = cast(Any, _FailingQdrantClient())

        with caplog.at_level(
            logging.WARNING,
            logger="rag_core.search.providers.qdrant_store",
        ):
            health = await store.check_health()

        assert health["healthy"] is False
        assert health["backend"] == "qdrant"
        assert health["error"] == "RuntimeError"
        assert "private qdrant detail" not in str(health)
        assert "private qdrant detail" not in caplog.text
        assert "RuntimeError" in caplog.text

    asyncio.run(run())


def test_turbopuffer_health_omits_backend_exception_message() -> None:
    async def run() -> None:
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=3,
            namespace_client=_FailingTurbopufferNamespace(),
        )

        health = await store.check_health()

        assert health["healthy"] is False
        assert health["backend"] == "turbopuffer"
        assert health["error"] == "RuntimeError"
        assert "private turbopuffer detail" not in str(health)

    asyncio.run(run())
