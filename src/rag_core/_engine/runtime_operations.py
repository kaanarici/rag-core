"""Engine lifetime and bounded, whole-operation resource reservations."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY

if TYPE_CHECKING:
    from rag_core.core import Engine

P = ParamSpec("P")
T = TypeVar("T")


class EngineClosedError(RuntimeError):
    """The Engine has begun closing and cannot admit another operation."""


class EngineBusyError(RuntimeError):
    """The Engine's bounded admission queue is full."""


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class RuntimeOperations:
    def __init__(
        self,
        *,
        max_concurrent_ingests: int = 8,
        max_concurrent_searches: int = 32,
        max_pending_operations: int = 64,
    ) -> None:
        self._ingest_limit = _positive_int(max_concurrent_ingests, "max_concurrent_ingests")
        self._search_limit = _positive_int(max_concurrent_searches, "max_concurrent_searches")
        if (
            isinstance(max_pending_operations, bool)
            or not isinstance(max_pending_operations, int)
            or max_pending_operations < 0
        ):
            raise ValueError("max_pending_operations must be a non-negative integer")
        self._pending_limit = max_pending_operations
        self._condition = asyncio.Condition()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending: dict[str, deque[object]] = {
            "ingest": deque(), "search": deque(), "other": deque()
        }
        self._pending_count = 0
        self._ingests = self._searches = self._active = 0
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    def snapshot(self) -> dict[str, object]:
        state = "open"
        if self._closing:
            state = "closed" if self._close_task is not None and self._close_task.done() else "closing"
        return {
            "state": state,
            "max_concurrent_ingests": self._ingest_limit,
            "max_concurrent_searches": self._search_limit,
            "max_pending_operations": self._pending_limit,
            "active_ingest_reservations": self._ingests,
            "active_searches": self._searches,
            "pending_operations": self._pending_count,
            "active_operations": self._active,
        }

    def _bind_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            if self._active or self._pending_count or (
                self._close_task is not None and not self._close_task.done()
            ):
                raise RuntimeError("Engine cannot be used concurrently across event loops")
            self._condition = asyncio.Condition()
            self._loop = loop

    @asynccontextmanager
    async def operation(
        self, *, ingest_slots: int = 0, search_slots: int = 0
    ) -> AsyncIterator[None]:
        self._bind_loop()
        if ingest_slots:
            _positive_int(ingest_slots, "max_concurrency")
        if ingest_slots > self._ingest_limit:
            raise ValueError("max_concurrency exceeds max_concurrent_ingests")

        def available() -> bool:
            return (
                self._ingests + ingest_slots <= self._ingest_limit
                and self._searches + search_slots <= self._search_limit
            )

        lane = "ingest" if ingest_slots else "search" if search_slots else "other"
        pending = self._pending[lane]
        async with self._condition:
            if self._closing:
                raise EngineClosedError("Engine is closing or closed")
            if pending or not available():
                if self._pending_count >= self._pending_limit:
                    raise EngineBusyError("Engine pending operation limit reached")
                ticket = object()
                pending.append(ticket)
                self._pending_count += 1
                try:
                    await self._condition.wait_for(
                        lambda: self._closing
                        or (pending[0] is ticket and available())
                    )
                    if self._closing:
                        raise EngineClosedError("Engine is closing or closed")
                finally:
                    pending.remove(ticket)
                    self._pending_count -= 1
                    self._condition.notify_all()
            self._ingests += ingest_slots
            self._searches += search_slots
            self._active += 1
        try:
            yield
        finally:
            async with self._condition:
                self._ingests -= ingest_slots
                self._searches -= search_slots
                self._active -= 1
                self._condition.notify_all()

    async def close(self, close_store: Callable[[], Awaitable[None]]) -> None:
        """Drain external operations; call outside operation-owned callbacks."""
        self._bind_loop()
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._finish_close(close_store))
        task = self._close_task
        if task.done():
            task.result()
            return
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            if not task.cancelled():
                task.exception()
            raise

    async def _finish_close(self, close_store: Callable[[], Awaitable[None]]) -> None:
        async with self._condition:
            self._condition.notify_all()
            await self._condition.wait_for(
                lambda: self._active == 0 and self._pending_count == 0
            )
        await close_store()


def operation(
    *, ingest: bool = False, batch: bool = False, search: bool = False
) -> Callable[
    [Callable[Concatenate[Engine, P], Awaitable[T]]],
    Callable[Concatenate[Engine, P], Coroutine[Any, Any, T]],
]:
    def decorate(
        method: Callable[Concatenate[Engine, P], Awaitable[T]],
    ) -> Callable[Concatenate[Engine, P], Coroutine[Any, Any, T]]:
        @wraps(method)
        async def admitted(self: Engine, *args: P.args, **kwargs: P.kwargs) -> T:
            slots = int(ingest)
            if batch:
                slots = _positive_int(
                    kwargs.get("max_concurrency", DEFAULT_INGEST_MAX_CONCURRENCY),
                    "max_concurrency",
                )
            async with self._operations.operation(
                ingest_slots=slots, search_slots=int(search)
            ):
                return await method(self, *args, **kwargs)

        admitted.__name__ = method.__name__.removeprefix("_")
        return admitted

    return decorate
