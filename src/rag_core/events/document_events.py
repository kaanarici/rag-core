"""Document processing, embedding, and indexing event records."""

from __future__ import annotations

from dataclasses import dataclass
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
class ParseCompleted:
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
class OcrApplied:
    filename: str = ""
    provider: str = ""
    pages_processed: int = 0
    duration_ms: float = 0.0
    event_type: Literal["ocr.applied"] = OCR_APPLIED_EVENT


@dataclass(frozen=True)
class ChunkProduced:
    filename: str = ""
    chunk_count: int = 0
    chunking_strategy: str = ""
    event_type: Literal["chunk.produced"] = CHUNK_PRODUCED_EVENT


@dataclass(frozen=True)
class ContextualizeStarted:
    chunk_count: int = 0
    model: str = ""
    event_type: Literal["contextualize.started"] = CONTEXTUALIZE_STARTED_EVENT


@dataclass(frozen=True)
class ContextualizeCompleted:
    chunk_count: int = 0
    model: str = ""
    duration_ms: float = 0.0
    succeeded: bool = True
    event_type: Literal["contextualize.completed"] = CONTEXTUALIZE_COMPLETED_EVENT


@dataclass(frozen=True)
class EmbedRequested:
    provider: str = ""
    model: str = ""
    text_count: int = 0
    role: RetrievalChannel = DENSE_RETRIEVAL_CHANNEL
    event_type: Literal["embed.requested"] = EMBED_REQUESTED_EVENT


@dataclass(frozen=True)
class EmbedCompleted:
    provider: str = ""
    model: str = ""
    text_count: int = 0
    role: RetrievalChannel = DENSE_RETRIEVAL_CHANNEL
    duration_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    cache_bypasses: int = 0
    event_type: Literal["embed.completed"] = EMBED_COMPLETED_EVENT


@dataclass(frozen=True)
class IndexUpserted:
    namespace: str = ""
    corpus_id: str = ""
    document_id: str = ""
    point_count: int = 0
    duration_ms: float = 0.0
    event_type: Literal["index.upserted"] = INDEX_UPSERTED_EVENT


@dataclass(frozen=True)
class IndexDeleted:
    namespace: str = ""
    corpus_id: str = ""
    document_id: str = ""
    event_type: Literal["index.deleted"] = INDEX_DELETED_EVENT
