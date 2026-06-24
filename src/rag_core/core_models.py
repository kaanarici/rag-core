from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, cast

from rag_core.config import (
    ChunkingConfig,
    ContextualizerConfig,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION,
    DEMO_EMBEDDING_MODEL,
    DEMO_EMBEDDING_PROVIDER,
    EmbeddingConfig,
    IngestConfig,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_PROVIDER,
    MARKDOWN_CHUNKING_STRATEGY,
    QdrantConfig,
    RerankerConfig,
    VectorStoreConfig,
)
from rag_core.config.ingest_config import DEFAULT_PROCESSING_VERSION
from rag_core.search.policy import CollectionPolicy, DEFAULT_POLICY, VectorStorePolicy

__all__ = [
    "Config",
    "CollectionManifest",
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


class _ConfigMeta(type):
    def qdrant(
        cls,
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
    ) -> "Config":
        return _config_qdrant(
            cast("type[Config]", cls),
            url=url,
            embedding_provider=embedding_provider,
            model=model,
            embedding_dimensions=embedding_dimensions,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            qdrant_api_key=qdrant_api_key,
            store_collection=store_collection,
            dimension_aware_collection=dimension_aware_collection,
        )


@dataclass(frozen=True, kw_only=True)
class Config(metaclass=_ConfigMeta):
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    contextualizer: ContextualizerConfig = field(default_factory=ContextualizerConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    policy: VectorStorePolicy = DEFAULT_POLICY
    # Optional per-process collection/tier fence. ``None`` = unrestricted behavior.
    # Tiered deployments can pass a CollectionPolicy so rerank, lexical sidecar,
    # and cross-namespace requests fail at the seam before provider calls.
    collection_policy: CollectionPolicy | None = None

    @classmethod
    def local(cls, persist_dir: str | Path | None = None) -> "Config":
        return cls(
            qdrant=QdrantConfig(
                location=":memory:" if persist_dir is None else str(persist_dir),
            ),
            embedding=EmbeddingConfig(
                provider=LOCAL_EMBEDDING_PROVIDER,
                model=LOCAL_EMBEDDING_MODEL,
            ),
        )

    @classmethod
    def from_cli(
        cls,
        args: argparse.Namespace,
        *,
        manifest_dir: Path | None = None,
    ) -> "Config":
        from rag_core._engine.core_config_cli import build_rag_core_config_from_cli_args

        return build_rag_core_config_from_cli_args(
            cls,
            args,
            manifest_dir=manifest_dir,
        )


def _config_qdrant(
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
) -> Config:
    return cls(
        qdrant=QdrantConfig(
            url=url,
            api_key=qdrant_api_key,
            store_collection=store_collection,
            dimension_aware_collection=dimension_aware_collection,
        ),
        embedding=EmbeddingConfig(
            provider=embedding_provider,
            model=model or _default_embedding_model(embedding_provider),
            dimensions=embedding_dimensions,
            api_key=embedding_api_key,
            base_url=embedding_base_url,
        ),
    )


def _default_embedding_model(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == LOCAL_EMBEDDING_PROVIDER:
        return LOCAL_EMBEDDING_MODEL
    if normalized == DEMO_EMBEDDING_PROVIDER:
        return DEMO_EMBEDDING_MODEL
    if normalized == DEFAULT_EMBEDDING_PROVIDER:
        return DEFAULT_EMBEDDING_MODEL
    raise ValueError(
        f"Config.qdrant(embedding_provider={provider!r}) needs an explicit "
        f"model=...; only {DEFAULT_EMBEDDING_PROVIDER!r}, "
        f"{LOCAL_EMBEDDING_PROVIDER!r}, and {DEMO_EMBEDDING_PROVIDER!r} have "
        "factory defaults."
    )


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


@dataclass(frozen=True, kw_only=True)
class DeleteDocumentResult:
    """Honest accounting of which surfaces actually purged the document.

    ``index_deleted`` reflects the vector-store ack, not an optimistic "we
    asked the store" assumption. The cache and sidecar booleans are tri-state
    so callers can distinguish "purge succeeded" (``True``), "purge failed
    and a recovery journal entry was written" (``False``), and "no such
    surface configured on this Engine" (``None``).
    """

    document_id: str
    namespace: str
    collection: str
    index_deleted: bool
    sidecar_deleted: bool | None = None
    manifest_entry_deleted: bool | None = None
    # Right-to-forget completeness fields. ``None`` = surface not wired on
    # this Engine (e.g. NoCache); ``True`` = scoped purge succeeded; ``False``
    # = purge failed and a delete-recovery journal entry now tracks it.
    vector_store_acked: bool = False
    lexical_sidecar_purged: bool | None = None
    embedding_cache_purged: bool | None = None
    chunk_context_cache_purged: bool | None = None
    manifest_removed: bool | None = None


@dataclass(frozen=True, kw_only=True)
class CollectionManifestEntry:
    document_id: str
    namespace: str
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
    """Chars/4 token estimate, matching the context-pack budgeting heuristic."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


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
class CollectionManifest:
    namespace: str
    collection: str
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
    entries: tuple[CollectionManifestEntry, ...] = ()


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

    def serialize(self) -> str:
        payload: dict[str, object] = {
            "base_version": self.base_version,
            "source_type": self.source_type,
        }
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
        return cls(
            base_version=base_version,
            source_type=source_type,
            contextualizer_id=contextualizer_id,
            contextualizer_chunk_cap=contextualizer_chunk_cap,
        )
