"""Regression tests for the async->sync bridge.

The worker-thread path (used when a loop is already running) must propagate
cancellation/BaseException to the caller, never swallow it into a false ``None``
success, and must still return a legitimate ``None`` result.
"""

from __future__ import annotations

import asyncio

import pytest

from rag_core._sync import run_coro_blocking


def test_run_coro_blocking_propagates_baseexception_from_worker_thread() -> None:
    class Boom(BaseException):
        pass

    async def raiser() -> int:
        raise Boom("boom")

    async def outer() -> None:
        # Inside a running loop, run_coro_blocking uses the worker-thread bridge.
        run_coro_blocking(raiser())

    with pytest.raises(Boom):
        asyncio.run(outer())


def test_run_coro_blocking_propagates_cancellation_from_worker_thread() -> None:
    async def cancels() -> int:
        raise asyncio.CancelledError("cancelled")

    async def outer() -> int | None:
        return run_coro_blocking(cancels())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(outer())


def test_run_coro_blocking_returns_legitimate_none_result() -> None:
    async def returns_none() -> None:
        return None

    async def outer() -> None:
        # A real None result must come back as None, not raise.
        assert run_coro_blocking(returns_none()) is None

    asyncio.run(outer())


def test_run_coro_blocking_returns_value_outside_running_loop() -> None:
    async def returns_value() -> int:
        return 42

    assert run_coro_blocking(returns_value()) == 42
