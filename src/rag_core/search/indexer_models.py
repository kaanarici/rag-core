from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_core.core_models import PreparedChunk


@dataclass
class IndexRequest:
    document_id: str
    corpus_id: str
    namespace: str
    text: str
    filename: str
    mime_type: str
    source_type: str
    document_key: str | None = None
    content_sha256: str | None = None
    processing_version: str | None = None
    existing_chunk_count: int | None = None
    path: Optional[str] = None
    document_path: Optional[str] = None
    section_mappings: Optional[list[dict[str, object]]] = None
    content_bytes: Optional[bytes] = None
    document_metadata: Optional[dict[str, object]] = None
    extra_fields: Optional[dict[str, str]] = None
    chunker_strategy: Optional[str] = None
    embedding_model: Optional[str] = None
    pre_chunked_texts: Optional[list[str]] = field(default=None)
    embedding_chunk_texts: Optional[list[str]] = field(default=None)
    chunk_metadata: Optional[list[dict[str, object]]] = field(default=None)
    prepared_chunks: "Sequence[PreparedChunk] | None" = field(default=None)


@dataclass(frozen=True)
class IndexResult:
    document_id: str
    chunk_count: int
    point_ids: list[str]
    point_payloads: list[dict[str, object]]
    document_key: str | None = None
    content_sha256: str | None = None


@dataclass(frozen=True)
class DeleteAck:
    """Vector-store ack for a per-document delete.

    the caller's right-to-forget contract needs ``DeleteDocumentResult.index_deleted``
    to reflect what the store actually did, not the engine's optimism. Adapters
    that cannot report point counts (no per-id read-back) should set
    ``deleted_point_count=-1`` and rely on ``succeeded=True`` only when the
    underlying ``store.delete`` returned without raising.
    """

    succeeded: bool
    deleted_point_count: int = -1
