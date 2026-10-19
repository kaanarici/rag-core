"""Document processing, embedding, and indexing event records."""

from __future__ import annotations

from dataclasses import dataclass
from rag_core.events.record import EventRecord
from typing import Literal

from rag_core.events.event_types import (
    CHUNK_PRODUCED_EVENT,
    CONTEXTUALIZE_COMPLETED_EVENT,
    CONTEXTUALIZE_STARTED_EVENT,
    EMBED_COMPLETED_EVENT,
    EMBED_REQUESTED_EVENT,
    INDEX_DELETED_EVENT,
    INDEX_UPSERTED_EVENT,
    OCR_APPLIED_EVENT,
    PARSE_COMPLETED_EVENT,
)
from rag_core.retrieval_channels import DENSE_RETRIEVAL_CHANNEL, RetrievalChannel


@dataclass(frozen=True)
class ParseCompleted(EventRecord):
    filename: str = ""
    mime_type: str = ""
    parser: str = ""
    needs_ocr: bool = False
    quality_verdict: str = ""
    quality_details: str = ""
    char_count: int = 0
    meaningful_ratio: float = 0.0
    mojibake_ratio: float = 0.0
    text_to_page_ratio: float = 0.0
    page_count: int = 0
    ocr_page_count: int = 0
    ocr_page_indices: tuple[int, ...] = ()
    extraction_ratio: float | None = None
    duration_ms: float = 0.0
    event_type: Literal["parse.completed"] = PARSE_COMPLETED_EVENT


@dataclass(frozen=True)
class OcrApplied(EventRecord):
    filename: str = ""
    provider: str = ""
    pages_processed: int = 0
    duration_ms: float = 0.0
    event_type: Literal["ocr.applied"] = OCR_APPLIED_EVENT


@dataclass(frozen=True)
class ChunkProduced(EventRecord):
    filename: str = ""
    chunk_count: int = 0
    chunking_strategy: str = ""
    event_type: Literal["chunk.produced"] = CHUNK_PRODUCED_EVENT


@dataclass(frozen=True)
class ContextualizeStarted(EventRecord):
    chunk_count: int = 0
    model: str = ""
    contextualized_chunk_count: int = 0
    skipped_chunk_count: int = 0
    chunk_cap: int | None = None
    event_type: Literal["contextualize.started"] = CONTEXTUALIZE_STARTED_EVENT


@dataclass(frozen=True)
class ContextualizeCompleted(EventRecord):
    chunk_count: int = 0
    model: str = ""
    duration_ms: float = 0.0
    succeeded: bool = True
    contextualized_chunk_count: int = 0
    skipped_chunk_count: int = 0
    chunk_cap: int | None = None
    event_type: Literal["contextualize.completed"] = CONTEXTUALIZE_COMPLETED_EVENT


@dataclass(frozen=True)
class EmbedRequested(EventRecord):
    provider: str = ""
    model: str = ""
    text_count: int = 0
    role: RetrievalChannel = DENSE_RETRIEVAL_CHANNEL
    event_type: Literal["embed.requested"] = EMBED_REQUESTED_EVENT


@dataclass(frozen=True)
class EmbedCompleted(EventRecord):
    provider: str = ""
    model: str = ""
    text_count: int = 0
    role: RetrievalChannel = DENSE_RETRIEVAL_CHANNEL
    duration_ms: float = 0.0
    event_type: Literal["embed.completed"] = EMBED_COMPLETED_EVENT


@dataclass(frozen=True)
class IndexUpserted(EventRecord):
    tenant_id: str = ""
    collection: str = ""
    document_id: str = ""
    point_count: int = 0
    duration_ms: float = 0.0
    event_type: Literal["index.upserted"] = INDEX_UPSERTED_EVENT


@dataclass(frozen=True)
class IndexDeleted(EventRecord):
    tenant_id: str = ""
    collection: str = ""
    document_id: str = ""
    event_type: Literal["index.deleted"] = INDEX_DELETED_EVENT
