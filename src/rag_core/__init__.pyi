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
from .easy import Rag as Rag
from .easy import index as index
from .search import ContextSnippet as ContextSnippet
from .search import ContextPack as ContextPack
from .search import SearchResult as SearchResult
from .search import SourceLocator as SourceLocator
from .search import SourcePreview as SourcePreview
from .search import SourceReference as SourceReference

# Day-one surface only. The data-shape types above remain importable from
# ``rag_core`` (the door) but are kept off ``__all__`` so the obvious path
# stays obvious.
__all__: list[str] = [
    "index",
    "Rag",
    "RAGCore",
    "RAGCoreConfig",
    "ContextPack",
    "SearchResult",
]
