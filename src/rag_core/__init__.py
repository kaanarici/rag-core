from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core_models import Document as Document
    from .core_models import IngestResult as IngestResult
    from .core import Engine as Engine
    from .core_models import Config as Config
    from .easy import Index as Index
    from .easy import index as index
    from .scope import Scope as Scope
    from .search import Answerability as Answerability
    from .search import Evidence as Evidence
    from .search import RetrievalResult as RetrievalResult
    from .search import SearchOptions as SearchOptions
    from .search import format_evidence as format_evidence

__all__ = [
    "Answerability",
    "Config",
    "Document",
    "Engine",
    "Evidence",
    "Index",
    "IngestResult",
    "RetrievalResult",
    "Scope",
    "SearchOptions",
    "format_evidence",
    "index",
]

_EXPORTS = {
    "Engine": ("rag_core.core", "Engine"),
    "Config": ("rag_core.core_models", "Config"),
    "Document": ("rag_core.core_models", "Document"),
    "Index": ("rag_core.easy", "Index"),
    "IngestResult": ("rag_core.core_models", "IngestResult"),
    "Scope": ("rag_core.scope", "Scope"),
    "Answerability": ("rag_core.search", "Answerability"),
    "Evidence": ("rag_core.search", "Evidence"),
    "RetrievalResult": ("rag_core.search", "RetrievalResult"),
    "SearchOptions": ("rag_core.search", "SearchOptions"),
    "format_evidence": ("rag_core.search", "format_evidence"),
    "index": ("rag_core.easy", "index"),
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
