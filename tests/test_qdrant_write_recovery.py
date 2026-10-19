"""Scripted overload contracts and a real local lost-ack replay."""

from __future__ import annotations

import asyncio
from typing import Any, cast
from uuid import UUID

import httpx
import pytest
from qdrant_client import AsyncQdrantClient, models as rest
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_core.search.providers import qdrant_write
from rag_core.search.providers.qdrant_payloads import WriteLatencyTracker


def response(status: int, retry_after: str | None = None) -> UnexpectedResponse:
    return UnexpectedResponse(
        status, "private reason", b"private document and token",
        httpx.Headers({"Retry-After": retry_after} if retry_after else {}),
    )


def points(count: int = 8) -> list[rest.PointStruct]:
    return [
        rest.PointStruct(id=str(UUID(int=i + 1)), vector=[1.0, 0.0])
        for i in range(count)
    ]


async def write(client: Any, batch: list[rest.PointStruct]) -> None:
    await qdrant_write.upsert_with_fallback(
        client, "private-collection", 2, WriteLatencyTracker(), batch, 0
    )


@pytest.mark.parametrize("error", [
    response(429, "2"), response(503),
    ResourceExhaustedResponse("private token", 2),
    ResponseHandlingException(httpx.ReadTimeout("private URL")),
])
def test_overload_retries_original_batch_with_bounded_backoff(
    error: Exception, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    async def run() -> None:
        calls: list[list[rest.PointStruct]] = []
        delays: list[float] = []

        class Client:
            async def upsert(self, **kwargs: Any) -> None:
                calls.append(kwargs["points"])
                if len(calls) < 3:
                    raise error

        async def sleep(delay: float) -> None:
            delays.append(delay)

        monkeypatch.setattr(qdrant_write.asyncio, "sleep", sleep)
        batch = points()
        await write(Client(), batch)
        assert len(calls) == 3 and all(call is batch for call in calls)
        assert len(delays) == 2 and all(0 <= delay <= 8 for delay in delays)
        if isinstance(error, ResourceExhaustedResponse) or (
            isinstance(error, UnexpectedResponse) and error.status_code == 429
        ):
            assert delays == [2, 2]
        assert "private" not in caplog.text

    asyncio.run(run())


@pytest.mark.parametrize("error,expected_calls", [
    (response(503), 3), (response(401), 1),
    (response(429, "9"), 1), (ResourceExhaustedResponse("overloaded", 30), 1),
    (ResponseHandlingException(ValueError("bad serialization")), 1),
])
def test_unrecoverable_write_stops_without_request_amplification(
    error: Exception, expected_calls: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        sizes: list[int] = []

        class Client:
            async def upsert(self, **kwargs: Any) -> None:
                sizes.append(len(kwargs["points"]))
                raise error

        async def sleep(delay: float) -> None:
            pass

        monkeypatch.setattr(qdrant_write.asyncio, "sleep", sleep)
        with pytest.raises(type(error)):
            await write(Client(), points(100))
        assert sizes == [100] * expected_calls

    asyncio.run(run())


def test_payload_splitting_shares_one_overload_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        sizes: list[int] = []

        class Client:
            async def upsert(self, **kwargs: Any) -> None:
                sizes.append(len(kwargs["points"]))
                if len(sizes) == 1:
                    raise response(413)
                if len(sizes) != 3:
                    raise response(503)

        async def sleep(delay: float) -> None:
            pass

        monkeypatch.setattr(qdrant_write.asyncio, "sleep", sleep)
        with pytest.raises(UnexpectedResponse):
            await write(Client(), points())
        assert sizes == [8, 4, 4, 4, 4]

    asyncio.run(run())


def test_payload_splitting_preserves_every_point_once() -> None:
    async def run() -> None:
        written: list[Any] = []
        calls = 0

        class Client:
            async def upsert(self, **kwargs: Any) -> None:
                nonlocal calls
                calls += 1
                batch = kwargs["points"]
                if len(batch) > 1:
                    raise response(413)
                written.append(batch[0].id)

        batch = points(64)
        await write(Client(), batch)
        assert calls == 127
        assert written == [point.id for point in batch]

    asyncio.run(run())


def test_cancel_during_backoff_prevents_another_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        sleeping = asyncio.Event()
        calls = 0

        class Client:
            async def upsert(self, **kwargs: Any) -> None:
                nonlocal calls
                calls += 1
                raise response(429)

        async def sleep(delay: float) -> None:
            sleeping.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(qdrant_write.asyncio, "sleep", sleep)
        task = asyncio.create_task(write(Client(), points()))
        await asyncio.wait_for(sleeping.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == 1

    asyncio.run(run())


def test_lost_ack_replays_idempotently_in_real_local_qdrant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        client = AsyncQdrantClient(location=":memory:")
        await client.create_collection(
            "private-collection", vectors_config=rest.VectorParams(
                size=2, distance=rest.Distance.COSINE,
            ),
        )
        calls = 0

        class LoseFirstAck:
            async def upsert(self, **kwargs: Any) -> None:
                nonlocal calls
                calls += 1
                await client.upsert(**kwargs)
                if calls == 1:
                    raise ResponseHandlingException(httpx.ReadTimeout("lost ack"))

        async def sleep(delay: float) -> None:
            pass

        monkeypatch.setattr(qdrant_write.asyncio, "sleep", sleep)
        try:
            batch = points()
            await write(cast(Any, LoseFirstAck()), batch)
            assert calls == 2
            assert (await client.count("private-collection", exact=True)).count == 8
            rows = await client.retrieve("private-collection", [p.id for p in batch])
            assert {row.id for row in rows} == {p.id for p in batch}
        finally:
            await client.close()

    asyncio.run(run())
