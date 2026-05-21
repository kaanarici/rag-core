from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import RAGCore as RAGCore
    from .core_models import CorpusManifest as CorpusManifest
    from .core_models import CorpusManifestEntry as CorpusManifestEntry
    from .core_models import IngestedDocument as IngestedDocument
    from .core_models import OcrMetadata as OcrMetadata
    from .core_models import OcrRoutingSignal as OcrRoutingSignal
    from .core_models import ParsedDocument as ParsedDocument
    from .core_models import PreparedChunk as PreparedChunk
    from .core_models import PreparedDocument as PreparedDocument
    from .core_models import ProcessingFingerprint as ProcessingFingerprint
    from .core_models import RAGCoreConfig as RAGCoreConfig
    from .search.context_pack import ContextSnippet as ContextSnippet
    from .search.context_pack import ModelContextPack as ModelContextPack
    from .search.context_pack import SourceLocator as SourceLocator
    from .search.context_pack import SourcePreview as SourcePreview
    from .search.context_pack import SourceReference as SourceReference
    from .search.types import SearchResult as SearchResult

__all__ = [
    "ContextSnippet",
    "CorpusManifest",
    "CorpusManifestEntry",
    "IngestedDocument",
    "ModelContextPack",
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
    "ContextSnippet": ("rag_core.search.context_pack", "ContextSnippet"),
    "CorpusManifest": ("rag_core.core_models", "CorpusManifest"),
    "CorpusManifestEntry": ("rag_core.core_models", "CorpusManifestEntry"),
    "IngestedDocument": ("rag_core.core_models", "IngestedDocument"),
    "ModelContextPack": ("rag_core.search.context_pack", "ModelContextPack"),
    "OcrMetadata": ("rag_core.core_models", "OcrMetadata"),
    "OcrRoutingSignal": ("rag_core.core_models", "OcrRoutingSignal"),
    "ParsedDocument": ("rag_core.core_models", "ParsedDocument"),
    "PreparedChunk": ("rag_core.core_models", "PreparedChunk"),
    "PreparedDocument": ("rag_core.core_models", "PreparedDocument"),
    "ProcessingFingerprint": ("rag_core.core_models", "ProcessingFingerprint"),
    "RAGCore": ("rag_core.core", "RAGCore"),
    "RAGCoreConfig": ("rag_core.core_models", "RAGCoreConfig"),
    "SearchResult": ("rag_core.search.types", "SearchResult"),
    "SourceLocator": ("rag_core.search.context_pack", "SourceLocator"),
    "SourcePreview": ("rag_core.search.context_pack", "SourcePreview"),
    "SourceReference": ("rag_core.search.context_pack", "SourceReference"),
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
