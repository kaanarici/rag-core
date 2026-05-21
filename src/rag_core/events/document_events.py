"""Document processing, embedding, and indexing event records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
    event_type: Literal["parse.completed"] = "parse.completed"


@dataclass(frozen=True)
class OcrApplied:
    filename: str = ""
    provider: str = ""
    pages_processed: int = 0
    duration_ms: float = 0.0
    event_type: Literal["ocr.applied"] = "ocr.applied"


@dataclass(frozen=True)
class ChunkProduced:
    filename: str = ""
    chunk_count: int = 0
    chunking_strategy: str = ""
    event_type: Literal["chunk.produced"] = "chunk.produced"


@dataclass(frozen=True)
class ContextualizeStarted:
    chunk_count: int = 0
    model: str = ""
    event_type: Literal["contextualize.started"] = "contextualize.started"


@dataclass(frozen=True)
class ContextualizeCompleted:
    chunk_count: int = 0
    model: str = ""
    duration_ms: float = 0.0
    succeeded: bool = True
    event_type: Literal["contextualize.completed"] = "contextualize.completed"


@dataclass(frozen=True)
class EmbedRequested:
    provider: str = ""
    model: str = ""
    text_count: int = 0
    role: Literal["dense", "sparse"] = "dense"
    event_type: Literal["embed.requested"] = "embed.requested"


@dataclass(frozen=True)
class EmbedCompleted:
    provider: str = ""
    model: str = ""
    text_count: int = 0
    role: Literal["dense", "sparse"] = "dense"
    duration_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    cache_bypasses: int = 0
    event_type: Literal["embed.completed"] = "embed.completed"


@dataclass(frozen=True)
class IndexUpserted:
    namespace: str = ""
    corpus_id: str = ""
    document_id: str = ""
    point_count: int = 0
    duration_ms: float = 0.0
    event_type: Literal["index.upserted"] = "index.upserted"


@dataclass(frozen=True)
class IndexDeleted:
    namespace: str = ""
    corpus_id: str = ""
    document_id: str = ""
    event_type: Literal["index.deleted"] = "index.deleted"
