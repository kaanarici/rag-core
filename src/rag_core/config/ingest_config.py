from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROCESSING_VERSION = "rag_core_processing_v1"


@dataclass(frozen=True)
class IngestConfig:
    processing_version: str = DEFAULT_PROCESSING_VERSION
    source_type: str = "file"
    enable_lexical_search: bool = False
    manifest_directory: Path | None = None
    """Directory under which manifest JSONL files are written, one per corpus.

    When unset, the engine does not persist a manifest. The CLI defaults to
    ``./.rag-core/manifest`` so ``list-corpora`` style commands have data.
    """
    lexical_search_provider: str | None = None
    """Name of a lexical search provider registered in ``SEARCH_SIDECARS``.

    ``None`` keeps the default of no lexical index unless
    ``enable_lexical_search`` is True. Constructor injection
    (``RAGCore(search_sidecar=...)``) takes precedence.
    """
    embedding_cache_provider: str | None = None
    """Name of an EmbeddingCache registered in ``EMBEDDING_CACHES``.

    ``None`` means no cache wrapping is applied to the embedding provider.
    Constructor injection (``RAGCore(embedding_cache=...)``) takes precedence.
    """
    embedding_cache_path: Path | None = None
    """SQLite embedding-cache path when ``embedding_cache_provider="sqlite"``."""
