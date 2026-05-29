from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_core._engine.core_prepare_metadata import (
    coerce_float,
    coerce_float_or_zero,
    coerce_int_or_zero,
    coerce_str,
    normalize_page_indices,
    parse_quality_metadata,
)
from rag_core.events.emit import emit_event
from rag_core.events.types import ChunkProduced, OcrApplied, ParseCompleted

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


def emit_parse_completed(
    sink: "EventSink | None",
    *,
    filename: str,
    mime_type: str,
    metadata: dict[str, Any],
    duration_ms: float,
) -> None:
    parser_label = coerce_str(metadata.get("parser")) or mime_type
    quality = parse_quality_metadata(metadata)
    ocr_page_indices = normalize_page_indices(
        metadata.get("ocr_page_indices_telemetry", metadata.get("ocr_page_indices"))
    )
    emit_event(
        sink,
        ParseCompleted(
            filename=filename,
            mime_type=mime_type,
            parser=parser_label,
            needs_ocr=bool(metadata.get("needs_ocr")),
            quality_verdict=coerce_str(quality.get("verdict")) or "",
            quality_details=coerce_str(quality.get("details")) or "",
            char_count=coerce_int_or_zero(quality.get("char_count")),
            meaningful_ratio=coerce_float_or_zero(quality.get("meaningful_ratio")),
            mojibake_ratio=coerce_float_or_zero(quality.get("mojibake_ratio")),
            text_to_page_ratio=coerce_float_or_zero(quality.get("text_to_page_ratio")),
            page_count=coerce_int_or_zero(
                metadata.get("page_count", quality.get("page_count"))
            ),
            ocr_page_count=coerce_int_or_zero(metadata.get("ocr_page_count")),
            ocr_page_indices=tuple(ocr_page_indices),
            extraction_ratio=coerce_float(metadata.get("extraction_ratio")),
            duration_ms=duration_ms,
        ),
    )


def emit_chunk_produced(
    sink: "EventSink | None",
    *,
    filename: str,
    chunk_count: int,
    chunking_strategy: str,
) -> None:
    emit_event(
        sink,
        ChunkProduced(
            filename=filename,
            chunk_count=chunk_count,
            chunking_strategy=chunking_strategy,
        ),
    )


def emit_ocr_applied(
    sink: "EventSink | None",
    *,
    filename: str,
    provider: str,
    pages_processed: int,
    duration_ms: float,
) -> None:
    emit_event(
        sink,
        OcrApplied(
            filename=filename,
            provider=provider,
            pages_processed=pages_processed,
            duration_ms=duration_ms,
        ),
    )


__all__ = [
    "emit_chunk_produced",
    "emit_ocr_applied",
    "emit_parse_completed",
]
