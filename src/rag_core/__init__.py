from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import Engine as Engine
    from .core_models import Config as Config
    from .easy import Index as Index
    from .easy import index as index
    from .search import Context as Context
    from .search import SearchResult as SearchResult

__all__ = [
    "index",
    "Index",
    "Engine",
    "Config",
    "Context",
    "SearchResult",
]

_EXPORTS = {
    "index": ("rag_core.easy", "index"),
    "Index": ("rag_core.easy", "Index"),
    "Engine": ("rag_core.core", "Engine"),
    "Config": ("rag_core.core_models", "Config"),
    "Context": ("rag_core.search", "Context"),
    "SearchResult": ("rag_core.search", "SearchResult"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'rag_core' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
