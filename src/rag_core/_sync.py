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

_MISSING = object()


def run_coro_blocking(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # A loop is already running, so drive the coroutine on a worker thread.
    result: Any = _MISSING
    error: BaseException | None = None

    def _runner() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coro)
        # BaseException, not Exception: CancelledError/KeyboardInterrupt/SystemExit
        # must propagate to the caller, not be swallowed into a false None success.
        except BaseException as exc:  # noqa: BLE001
            error = exc

    thread = Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if error is not None:
        raise error
    if result is _MISSING:
        # The worker thread ended without a result or an error (e.g. interpreter
        # teardown). Fail loudly instead of returning a wrong None.
        raise RuntimeError("run_coro_blocking worker produced neither result nor error")
    return cast(T, result)


__all__ = ["run_coro_blocking"]
