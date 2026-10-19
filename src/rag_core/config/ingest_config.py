from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_PROCESSING_VERSION = "rag_core_processing_v9"
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
SkipUnchangedMode = Literal["fast", "materialize"]
SKIP_UNCHANGED_FAST: SkipUnchangedMode = "fast"
SKIP_UNCHANGED_MATERIALIZE: SkipUnchangedMode = "materialize"


@dataclass(frozen=True)
class IngestConfig:
    processing_version: str = DEFAULT_PROCESSING_VERSION
    source_type: str = DEFAULT_INGEST_SOURCE_TYPE
    max_document_bytes: int = 64 * 1024 * 1024
    """Maximum source size before parsing; local reads allocate at most limit + 1."""
    build_sparse_index: bool = False
    """Build sparse vectors for explicit lexical or hybrid retrieval."""
    manifest_directory: Path | None = None
    skip_unchanged: SkipUnchangedMode = SKIP_UNCHANGED_FAST
    """How unchanged documents are handled during skip-by-hash reingest.

    ``"fast"`` returns stored document metadata without parsing or chunking.
    ``"materialize"`` preserves the older behavior of parsing the file to
    return freshly prepared metadata even though indexing is skipped.
    """

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_document_bytes, bool)
            or not isinstance(self.max_document_bytes, int)
            or self.max_document_bytes <= 0
        ):
            raise ValueError("IngestConfig.max_document_bytes must be a positive integer")
        if not isinstance(self.build_sparse_index, bool):
            raise ValueError("IngestConfig.build_sparse_index must be a boolean")
        if self.skip_unchanged not in {
            SKIP_UNCHANGED_FAST,
            SKIP_UNCHANGED_MATERIALIZE,
        }:
            raise ValueError(
                "IngestConfig.skip_unchanged must be 'fast' or 'materialize'"
            )
