from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import RAGCore as RAGCore
    from .core_models import CorpusManifest as CorpusManifest
    from .core_models import CorpusManifestEntry as CorpusManifestEntry
    from .core_models import DeleteDocumentResult as DeleteDocumentResult
    from .core_models import IngestedDocument as IngestedDocument
    from .core_models import OcrMetadata as OcrMetadata
    from .core_models import OcrRoutingSignal as OcrRoutingSignal
    from .core_models import ParsedDocument as ParsedDocument
    from .core_models import PreparedChunk as PreparedChunk
    from .core_models import PreparedDocument as PreparedDocument
    from .core_models import ProcessingFingerprint as ProcessingFingerprint
    from .core_models import RAGCoreConfig as RAGCoreConfig
    from .search import ContextSnippet as ContextSnippet
    from .search import ContextPack as ContextPack
    from .search import SearchResult as SearchResult
    from .search import SourceLocator as SourceLocator
    from .search import SourcePreview as SourcePreview
    from .search import SourceReference as SourceReference

__all__ = [
    "ContextSnippet",
    "CorpusManifest",
    "CorpusManifestEntry",
    "DeleteDocumentResult",
    "IngestedDocument",
    "ContextPack",
    "OcrMetadata",
    "OcrRoutingSignal",
    "ParsedDocument",
    "PreparedChunk",
    "PreparedDocument",
    "ProcessingFingerprint",
    "RAGCore",
    "RAGCoreConfig",
    "SearchResult",
    "SourceLocator",
    "SourcePreview",
    "SourceReference",
]

_EXPORTS = {
    "ContextSnippet": ("rag_core.search", "ContextSnippet"),
    "CorpusManifest": ("rag_core.core_models", "CorpusManifest"),
    "CorpusManifestEntry": ("rag_core.core_models", "CorpusManifestEntry"),
    "DeleteDocumentResult": ("rag_core.core_models", "DeleteDocumentResult"),
    "IngestedDocument": ("rag_core.core_models", "IngestedDocument"),
    "ContextPack": ("rag_core.search", "ContextPack"),
    "OcrMetadata": ("rag_core.core_models", "OcrMetadata"),
    "OcrRoutingSignal": ("rag_core.core_models", "OcrRoutingSignal"),
    "ParsedDocument": ("rag_core.core_models", "ParsedDocument"),
    "PreparedChunk": ("rag_core.core_models", "PreparedChunk"),
    "PreparedDocument": ("rag_core.core_models", "PreparedDocument"),
    "ProcessingFingerprint": ("rag_core.core_models", "ProcessingFingerprint"),
    "RAGCore": ("rag_core.core", "RAGCore"),
    "RAGCoreConfig": ("rag_core.core_models", "RAGCoreConfig"),
    "SearchResult": ("rag_core.search", "SearchResult"),
    "SourceLocator": ("rag_core.search", "SourceLocator"),
    "SourcePreview": ("rag_core.search", "SourcePreview"),
    "SourceReference": ("rag_core.search", "SourceReference"),
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
    return sorted(set(globals()) | set(__all__))
