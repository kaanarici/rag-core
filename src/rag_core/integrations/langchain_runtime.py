"""Runtime helpers for optional LangChain adapters."""

from __future__ import annotations

from typing import Any

from rag_core._sync import run_coro_blocking


class LangChainNotInstalledError(ImportError):
    """Raised when a LangChain adapter is used without langchain-core installed."""


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
