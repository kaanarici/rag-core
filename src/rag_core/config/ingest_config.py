from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROCESSING_VERSION = "rag_core_processing_v1"
INGEST_SOURCE_TYPE_FILE = "file"
INGEST_SOURCE_TYPE_URL = "url"
INGEST_SOURCE_TYPE_ARCHIVE = "archive"
DEFAULT_INGEST_SOURCE_TYPE = INGEST_SOURCE_TYPE_FILE
STANDARD_INGEST_SOURCE_TYPES = (
    INGEST_SOURCE_TYPE_FILE,
    INGEST_SOURCE_TYPE_URL,
    INGEST_SOURCE_TYPE_ARCHIVE,
)
DEFAULT_CLI_MANIFEST_DIRECTORY = ".rag-core/manifest"
DEFAULT_INGEST_MAX_CONCURRENCY = 1
CLI_MANIFEST_DIR_ENV = "RAG_CORE_MANIFEST_DIR"
PROCESSING_VERSION_ENV = "RAG_CORE_PROCESSING_VERSION"


@dataclass(frozen=True)
class IngestConfig:
    processing_version: str = DEFAULT_PROCESSING_VERSION
    source_type: str = DEFAULT_INGEST_SOURCE_TYPE
    enable_lexical_search: bool = False
    manifest_directory: Path | None = None
    """Directory under which manifest JSONL files are written, one per corpus.

    When unset, the engine does not persist a manifest. The CLI defaults to
    ``DEFAULT_CLI_MANIFEST_DIRECTORY`` so manifest inspection commands have data.
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
