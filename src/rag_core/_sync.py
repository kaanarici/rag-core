"""Run an async coroutine to completion from sync code.

Single owner for the async->sync bridge used by the easy facade and the
optional integration adapters. If a loop is already running, the coroutine is
driven on a short-lived worker thread so callers in notebooks / nested loops
keep working.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from threading import Thread
from typing import Any, TypeVar, cast


T = TypeVar("T")


def run_coro_blocking(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, object] = {}

    def _runner() -> None:
        try:
            box["result"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover - delegated back to caller
            box["error"] = exc

    thread = Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    error = box.get("error")
    if error is not None:
        raise cast(Exception, error)
    return cast(T, box.get("result"))


__all__ = ["run_coro_blocking"]
