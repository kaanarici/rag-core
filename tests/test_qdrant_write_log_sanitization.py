from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import httpx
import pytest
from qdrant_client import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_core.search.providers import qdrant_write
from rag_core.search.providers.qdrant_payloads import WriteLatencyTracker
from rag_core.search.providers.qdrant_write import (
    upsert_with_fallback,
)
from tests.support import TEST_API_SECRET

LOGGER_NAME = "rag_core.search.providers.qdrant_write"
SECRET = TEST_API_SECRET


class _SuccessfulUpsertClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def upsert(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _private_point() -> rest.PointStruct:
    return rest.PointStruct(
        id="point-1",
        vector={"": [0.1, 0.2, 0.3]},
        payload={
            "api_key": SECRET,
            "email": "private@example.com",
        },
    )


def _private_unexpected_response() -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=429,
        reason_phrase=f"rate limited {SECRET}",
        content=f'{{"error":"{SECRET}"}}'.encode(),
        headers=httpx.Headers(),
    )




def test_slow_qdrant_write_log_omits_collection_name(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        client = _SuccessfulUpsertClient()
        latency = WriteLatencyTracker()
        monkeypatch.setattr(
            qdrant_write, "_SLOW_WRITE_THRESHOLD_SECONDS", -1.0
        )

        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            await upsert_with_fallback(
                client=cast(Any, client),
                collection_name=f"docs-{SECRET}",
                dimensions=3,
                latency=latency,
                points=[_private_point()],
                split_depth=1,
            )

        assert client.calls[0]["collection_name"] == f"docs-{SECRET}"

    asyncio.run(run())

    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "Slow Qdrant write" in message
    assert "provider=qdrant" in message
    assert "backend=qdrant" not in message
    assert "point_count=1" in message
    assert "dense_dimensions=3" in message
    assert "split_depth=1" in message
    assert "latency_p50=" in message
    assert "latency_p95=" in message
    assert f"docs-{SECRET}" not in message
    assert "private@example.com" not in message
    assert SECRET not in message
    assert "Traceback" not in message
    assert all(record.exc_info is None for record in caplog.records)


def test_debug_qdrant_write_log_omits_collection_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def run() -> None:
        client = _SuccessfulUpsertClient()
        latency = WriteLatencyTracker()

        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            await upsert_with_fallback(
                client=cast(Any, client),
                collection_name=f"docs-{SECRET}",
                dimensions=3,
                latency=latency,
                points=[_private_point()],
                split_depth=0,
            )

        assert client.calls[0]["collection_name"] == f"docs-{SECRET}"

    asyncio.run(run())

    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "Qdrant write completed" in message
    assert "provider=qdrant" in message
    assert "backend=qdrant" not in message
    assert "point_count=1" in message
    assert "dense_dimensions=3" in message
    assert "split_depth=0" in message
    assert "latency_p50=" in message
    assert "latency_p95=" in message
    assert f"docs-{SECRET}" not in message
    assert "private@example.com" not in message
    assert SECRET not in message
    assert "Traceback" not in message
    assert all(record.exc_info is None for record in caplog.records)
