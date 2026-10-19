"""Cancellation joins real blocking workers before releasing their owners."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from rag_core._async import run_sync
from rag_core.search.providers.local_embedding import LocalEmbeddingProvider


@pytest.mark.parametrize("worker_error", [False, True])
def test_repeated_cancellation_joins_worker_and_preserves_capacity(worker_error: bool) -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = threading.Event()
        finished = threading.Event()
        sem = asyncio.Semaphore(1)
        loop = asyncio.get_running_loop()
        later_stages: list[str] = []

        def blocking() -> None:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            finished.set()
            if worker_error:
                raise ValueError("worker failed during cancellation")

        async def operation() -> None:
            async with sem:
                await run_sync(blocking)
                later_stages.append("publish")

        task = asyncio.create_task(operation())
        try:
            await asyncio.wait_for(started.wait(), 2)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
            assert not task.done() and sem.locked() and not finished.is_set()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert finished.is_set() and not sem.locked() and later_stages == []

    asyncio.run(run())


def test_worker_result_error_and_self_cancellation() -> None:
    def success(*, value: int) -> int:
        return value

    def error() -> None:
        raise ValueError("failed")

    def cancelled() -> None:
        raise asyncio.CancelledError

    async def run() -> None:
        assert await run_sync(success, value=42) == 42
        with pytest.raises(ValueError, match="failed"):
            await run_sync(error)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_sync(cancelled), 2)

    asyncio.run(run())


def test_cancelled_embedding_callers_do_not_leave_queued_inference() -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        calls = 0
        provider = LocalEmbeddingProvider()

        class Model:
            def query_embed(self, texts: list[str]) -> list[list[float]]:
                nonlocal calls
                calls += 1
                loop.call_soon_threadsafe(started.set)
                assert release.wait(5)
                return [[1.0] * provider.dimensions]

        provider._model = Model()
        tasks = [asyncio.create_task(provider.embed_query(str(i))) for i in range(12)]
        try:
            await asyncio.wait_for(started.wait(), 2)
            for task in tasks:
                task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert calls == 1
            assert sum(not task.done() for task in tasks) == 1
        finally:
            release.set()
        outcomes = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
        assert all(isinstance(result, asyncio.CancelledError) for result in outcomes)
        assert calls == 1
        # The cancelled owner releases the model only after the worker has finished.
        assert len(await provider.embed_query("next request")) == provider.dimensions
        assert calls == 2

    asyncio.run(run())


def test_inference_failure_is_not_reported_as_a_download_failure() -> None:
    class Model:
        def query_embed(self, texts: list[str]) -> Any:
            raise ValueError("inference failed")

    async def run() -> None:
        provider = LocalEmbeddingProvider()
        provider._model = Model()
        with pytest.raises(ValueError, match="inference failed"):
            await provider.embed_query("question")

    asyncio.run(run())


def test_local_model_serialization_survives_successive_sync_event_loops() -> None:
    provider = LocalEmbeddingProvider()

    class Model:
        def query_embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] * provider.dimensions]

    provider._model = Model()

    async def batch() -> None:
        results = await asyncio.gather(*(provider.embed_query(str(i)) for i in range(4)))
        assert len(results) == 4

    asyncio.run(batch())
    asyncio.run(batch())
