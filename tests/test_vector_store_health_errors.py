from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import pytest
from rag_core.search.providers.qdrant_store import QdrantVectorStore


class _FailingQdrantClient:
    async def get_collection(self, *, collection_name: str) -> object:
        raise RuntimeError("private qdrant detail")

    async def close(self) -> None:
        return None


def test_qdrant_health_omits_adapter_exception_message(
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
        assert health["adapter"] == "qdrant"
        assert "backend" not in health
        assert health["error"] == "RuntimeError"
        assert "private qdrant detail" not in str(health)
        assert "private qdrant detail" not in caplog.text
        assert "RuntimeError" in caplog.text

    asyncio.run(run())
