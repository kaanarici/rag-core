from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .rag import Document as Document
    from .rag import Evidence as Evidence
    from .rag import IngestResult as IngestResult
    from .rag import RAGCore as RAGCore
    from .rag import RetrievalResult as RetrievalResult
    from .core_models import Config as Config

__all__ = [
    "RAGCore",
    "Config",
    "Document",
    "Evidence",
    "IngestResult",
    "RetrievalResult",
]

_EXPORTS = {
    "RAGCore": ("rag_core.rag", "RAGCore"),
    "Config": ("rag_core.core_models", "Config"),
    "Document": ("rag_core.rag", "Document"),
    "Evidence": ("rag_core.rag", "Evidence"),
    "IngestResult": ("rag_core.rag", "IngestResult"),
    "RetrievalResult": ("rag_core.rag", "RetrievalResult"),
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
