"""Runtime helpers for optional LangChain adapters."""

from __future__ import annotations

import asyncio
from threading import Thread
from typing import Any, cast


class LangChainNotInstalledError(ImportError):
    """Raised when a LangChain adapter is used without langchain-core installed."""


def run_coro_blocking(coro: Any) -> Any:
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
    return box.get("result")


def require_langchain_symbol(module_name: str, symbol: str) -> Any:
    try:
        module = __import__(module_name, fromlist=[symbol])
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LangChainNotInstalledError(
            "LangChain integration requires langchain-core. "
            "Install rag-core with the `langchain` extra enabled."
        ) from exc
    try:
        return getattr(module, symbol)
    except AttributeError as exc:  # pragma: no cover - environment dependent
        raise LangChainNotInstalledError(
            "LangChain integration requires langchain-core with expected adapter symbols. "
            f"Missing `{symbol}` in `{module_name}`."
        ) from exc


__all__ = [
    "LangChainNotInstalledError",
    "require_langchain_symbol",
    "run_coro_blocking",
]
