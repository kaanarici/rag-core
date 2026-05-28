from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from rag_core.config import (
    ChunkingConfig,
    EmbeddingConfig,
    IngestConfig,
    MARKDOWN_CHUNKING_STRATEGY,
    QdrantConfig,
    RerankerConfig,
    VectorStoreConfig,
)
from rag_core.config.ingest_config import DEFAULT_PROCESSING_VERSION
from rag_core.search.policy import DEFAULT_POLICY, VectorStorePolicy

__all__ = [
    "CorpusManifest",
    "CorpusManifestEntry",
    "DEFAULT_PROCESSING_VERSION",
    "DeleteDocumentResult",
    "IngestedDocument",
    "OcrMetadata",
    "OcrRoutingSignal",
    "ParsedDocument",
    "PreparedChunk",
    "PreparedDocument",
    "ProcessingFingerprint",
    "RAGCoreConfig",
]


@dataclass(frozen=True)
class RAGCoreConfig:
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    policy: VectorStorePolicy = DEFAULT_POLICY

    @classmethod
    def from_cli(
        cls,
        args: argparse.Namespace,
        *,
        manifest_dir: Path | None = None,
    ) -> "RAGCoreConfig":
        from rag_core.core_config_cli import build_rag_core_config_from_cli_args

        return build_rag_core_config_from_cli_args(
            cls,
            args,
            manifest_dir=manifest_dir,
        )


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    mime_type: str
    markdown: str
    metadata: dict[str, Any] = field(default_factory=dict)
    path: str | None = None


@dataclass(frozen=True)
class IngestedDocument:
    document_id: str
    corpus_id: str
    namespace: str
    chunk_count: int
    filename: str
    mime_type: str
    document_key: str | None = None
    content_sha256: str | None = None
    ingest_state: str = "created"
    replaced_existing: bool = False
    collection_name: str | None = None
    embedding_model: str | None = None
    processing_version: str | None = None
    ocr: "OcrRoutingSignal" = field(default_factory=lambda: OcrRoutingSignal())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeleteDocumentResult:
    document_id: str
    namespace: str
    corpus_id: str
    index_deleted: bool
    sidecar_deleted: bool | None = None
    manifest_entry_deleted: bool | None = None


@dataclass(frozen=True)
class CorpusManifestEntry:
    document_id: str
    namespace: str
    corpus_id: str
    document_key: str | None
    content_sha256: str | None
    filename: str
    mime_type: str
    chunk_count: int
    parser: str | None = None
    needs_ocr: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedChunk:
    """The canonical chunk type used end-to-end by chunking, prepare, and indexing."""

    chunk_index: int
    text: str
    embedding_text: str
    word_count: int
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    chunking_strategy: str = MARKDOWN_CHUNKING_STRATEGY
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OcrRoutingSignal:
    needed: bool = False
    page_indices: list[int] = field(default_factory=list)
    confidence: float | None = None
    parser: str | None = None


@dataclass(frozen=True)
class PreparedDocument:
    filename: str
    mime_type: str
    markdown: str
    chunks: list[PreparedChunk]
    metadata: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    ocr: OcrRoutingSignal = field(default_factory=OcrRoutingSignal)


@dataclass(frozen=True)
class CorpusManifest:
    namespace: str
    corpus_id: str
    collection_name: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    document_count: int
    chunk_count: int
    source_document_ids: tuple[str, ...]
    ocr_document_count: int
    ocr_page_count: int
    documents: tuple[IngestedDocument, ...]
    entries: tuple[CorpusManifestEntry, ...] = ()


@dataclass(frozen=True)
class OcrMetadata:
    """Outcome of an OCR step on one document.

    Stamped onto ``IngestedDocument.metadata["ocr"]`` (via ``asdict``) and read
    back through this typed shape rather than via individual magic-string keys.
    """

    provider: str | None = None
    model: str | None = None
    pages_used: tuple[int, ...] = ()
    page_count: int = 0
    merge_mode: str | None = None


@dataclass(frozen=True)
class ProcessingFingerprint:
    """Identity of the processing pipeline that produced a document's chunks.

    Persisted as the JSON form of ``serialize()`` on the vector-store payload
    and on manifest entries; drift between the runtime fingerprint and the
    stored fingerprint triggers an automatic reindex.
    """

    base_version: str
    source_type: str
    contextualizer_id: str | None = None

    def serialize(self) -> str:
        payload = {"base_version": self.base_version, "source_type": self.source_type}
        if self.contextualizer_id is not None:
            payload["contextualizer_id"] = self.contextualizer_id
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def parse(cls, raw: str) -> "ProcessingFingerprint":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("ProcessingFingerprint expects valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"ProcessingFingerprint expects a JSON object, got: {raw!r}")
        base_version = payload.get("base_version")
        source_type = payload.get("source_type")
        if not isinstance(base_version, str) or not base_version:
            raise ValueError("ProcessingFingerprint.base_version must be a non-empty string")
        if not isinstance(source_type, str) or not source_type:
            raise ValueError("ProcessingFingerprint.source_type must be a non-empty string")
        contextualizer_id = payload.get("contextualizer_id")
        if contextualizer_id is not None and not isinstance(contextualizer_id, str):
            raise ValueError("ProcessingFingerprint.contextualizer_id must be a string")
        return cls(
            base_version=base_version,
            source_type=source_type,
            contextualizer_id=contextualizer_id,
        )
