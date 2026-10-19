from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from rag_core.config import (
    ChunkingConfig,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION,
    EmbeddingConfig,
    IngestConfig,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_PROVIDER,
    MARKDOWN_CHUNKING_STRATEGY,
    QdrantConfig,
    VectorStoreConfig,
)
from rag_core.config.embedding_config import default_embedding_model
from rag_core.config.ingest_config import DEFAULT_PROCESSING_VERSION
from rag_core.search.policy import CollectionPolicy, DEFAULT_POLICY, VectorStorePolicy

_TOKEN_UNIT_RE = re.compile(r"[A-Za-z0-9_]+|[^\s]", re.UNICODE)

__all__ = [
    "Config",
    "CollectionManifestEntry",
    "DEFAULT_PROCESSING_VERSION",
    "DeleteDocumentResult",
    "IngestedDocument",
    "OcrMetadata",
    "OcrRoutingSignal",
    "ParsedDocument",
    "PreparedChunk",
    "PreparedDocument",
    "ProcessingFingerprint",
]


@dataclass(frozen=True, kw_only=True)
class Config:
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    hybrid: bool = False
    policy: VectorStorePolicy = DEFAULT_POLICY
    # Optional per-process tenant_id/collection fence. ``None`` = unrestricted.
    collection_policy: CollectionPolicy | None = None

    @classmethod
    def qdrant(
        cls: type[Config],
        *,
        url: str,
        embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
        model: str | None = None,
        embedding_dimensions: int | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        qdrant_api_key: str | None = None,
        store_collection: str = DEFAULT_QDRANT_COLLECTION,
        dimension_aware_collection: bool = DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION,
        hybrid: bool = False,
    ) -> Config:
        return cls(
            hybrid=hybrid,
            vector_store=VectorStoreConfig(
                qdrant=QdrantConfig(
                    url=url,
                    api_key=qdrant_api_key,
                    store_collection=store_collection,
                    dimension_aware_collection=dimension_aware_collection,
                ),
            ),
            embedding=EmbeddingConfig(
                provider=embedding_provider,
                model=model or default_embedding_model(embedding_provider),
                dimensions=embedding_dimensions,
                api_key=embedding_api_key,
                base_url=embedding_base_url,
            ),
        )

    @classmethod
    def local(
        cls, persist_dir: str | Path | None = None, *, hybrid: bool = False
    ) -> "Config":
        return cls(
            vector_store=VectorStoreConfig(
                qdrant=QdrantConfig(
                    location=":memory:" if persist_dir is None else str(persist_dir),
                ),
            ),
            embedding=EmbeddingConfig(
                provider=LOCAL_EMBEDDING_PROVIDER,
                model=LOCAL_EMBEDDING_MODEL,
            ),
            hybrid=hybrid,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.hybrid, bool):
            raise ValueError("Config.hybrid must be a boolean")


@dataclass(frozen=True, kw_only=True)
class ParsedDocument:
    filename: str
    mime_type: str
    markdown: str
    metadata: dict[str, Any] = field(default_factory=dict)
    path: str | None = None


@dataclass(frozen=True, kw_only=True)
class IngestedDocument:
    document_id: str
    collection: str
    tenant_id: str
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


@dataclass(frozen=True, kw_only=True)
class DeleteDocumentResult:
    document_id: str
    tenant_id: str
    collection: str
    vector_store_acked: bool
    manifest_removed: bool | None = None


@dataclass(frozen=True, kw_only=True)
class CollectionManifestEntry:
    document_id: str
    tenant_id: str
    collection: str
    document_key: str | None
    content_sha256: str | None
    filename: str
    mime_type: str
    chunk_count: int
    parser: str | None = None
    needs_ocr: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class PreparedChunk:
    """The canonical chunk type used end-to-end by chunking, prepare, and indexing."""

    chunk_index: int
    text: str
    embedding_text: str
    word_count: int
    start_char: int | None = None
    end_char: int | None = None
    token_count: int = 0
    chunking_strategy: str = MARKDOWN_CHUNKING_STRATEGY
    metadata: Mapping[str, object] = field(default_factory=dict)
    sparse_text: str | None = None


def estimate_token_count(text: str) -> int:
    """Deterministic model-neutral estimate for chunk budgeting and metadata."""
    if not text:
        return 0
    count = 0
    for match in _TOKEN_UNIT_RE.finditer(text):
        unit = match.group(0)
        if unit.isascii() and (unit[0].isalnum() or unit[0] == "_"):
            count += max(1, (len(unit) + 3) // 4)
        else:
            count += len(unit)
    return max(1, count)


@dataclass(frozen=True, kw_only=True)
class OcrRoutingSignal:
    needed: bool = False
    page_indices: list[int] = field(default_factory=list)
    confidence: float | None = None
    parser: str | None = None


@dataclass(frozen=True, kw_only=True)
class PreparedDocument:
    filename: str
    mime_type: str
    markdown: str
    chunks: list[PreparedChunk]
    metadata: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    ocr: OcrRoutingSignal = field(default_factory=OcrRoutingSignal)


@dataclass(frozen=True, kw_only=True)
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


@dataclass(frozen=True, kw_only=True)
class ProcessingFingerprint:
    """Identity of the processing pipeline that produced a document's chunks.

    Persisted as the JSON form of ``serialize()`` on the vector-store payload
    and on manifest entries; drift between the runtime fingerprint and the
    stored fingerprint triggers an automatic reindex.
    """

    base_version: str
    source_type: str
    contextualizer_id: str | None = None
    contextualizer_chunk_cap: int | None = None
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    sparse_index: bool = False
    sparse_model: str | None = None

    def serialize(self) -> str:
        payload: dict[str, object] = {
            "base_version": self.base_version,
            "source_type": self.source_type,
            "chunking": asdict(self.chunking),
            "sparse_index": self.sparse_index,
        }
        if self.sparse_model is not None:
            payload["sparse_model"] = self.sparse_model
        if self.contextualizer_id is not None:
            payload["contextualizer_id"] = self.contextualizer_id
        if self.contextualizer_chunk_cap is not None:
            payload["contextualizer_chunk_cap"] = self.contextualizer_chunk_cap
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
            raise ValueError(
                f"ProcessingFingerprint expects a JSON object, got: {raw!r}"
            )
        base_version = payload.get("base_version")
        source_type = payload.get("source_type")
        if not isinstance(base_version, str) or not base_version:
            raise ValueError(
                "ProcessingFingerprint.base_version must be a non-empty string"
            )
        if not isinstance(source_type, str) or not source_type:
            raise ValueError(
                "ProcessingFingerprint.source_type must be a non-empty string"
            )
        contextualizer_id = payload.get("contextualizer_id")
        if contextualizer_id is not None and not isinstance(contextualizer_id, str):
            raise ValueError("ProcessingFingerprint.contextualizer_id must be a string")
        raw_chunk_cap = payload.get("contextualizer_chunk_cap")
        if raw_chunk_cap is None:
            contextualizer_chunk_cap = None
        elif (
            isinstance(raw_chunk_cap, bool)
            or not isinstance(raw_chunk_cap, int)
            or raw_chunk_cap <= 0
        ):
            raise ValueError(
                "ProcessingFingerprint.contextualizer_chunk_cap must be a positive integer"
            )
        else:
            contextualizer_chunk_cap = raw_chunk_cap
        chunking_data = payload.get("chunking", {})
        if not isinstance(chunking_data, dict):
            raise ValueError("ProcessingFingerprint.chunking must be an object")
        try:
            chunking = ChunkingConfig(**chunking_data)
        except (TypeError, ValueError) as exc:
            raise ValueError("ProcessingFingerprint.chunking is invalid") from exc
        sparse_index = payload.get("sparse_index", False)
        if not isinstance(sparse_index, bool):
            raise ValueError("ProcessingFingerprint.sparse_index must be a boolean")
        sparse_model = payload.get("sparse_model")
        if sparse_model is not None and (not isinstance(sparse_model, str) or not sparse_model.strip()):
            raise ValueError("ProcessingFingerprint.sparse_model must be a non-empty string")
        return cls(
            sparse_model=sparse_model,
            base_version=base_version,
            source_type=source_type,
            contextualizer_id=contextualizer_id,
            contextualizer_chunk_cap=contextualizer_chunk_cap,
            chunking=chunking,
            sparse_index=sparse_index,
        )
