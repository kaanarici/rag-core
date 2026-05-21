from __future__ import annotations

from dataclasses import dataclass

from rag_core.documents.pdf_inspector_payloads import (
    optional_bool,
    optional_float,
    optional_non_negative_int,
    parse_analysis_fields,
    require_markdown,
    require_pages_needing_ocr,
    require_positive_int,
    require_string,
)


@dataclass(frozen=True)
class PdfInspectorDetectionResult:
    """Normalized detection result from pdf-inspector."""

    pdf_type: str
    page_count: int
    pages_needing_ocr: list[int]
    confidence: float | None
    has_encoding_issues: bool
    processing_time_ms: int | None
    is_complex: bool | None = None
    pages_with_tables: list[int] | None = None
    pages_with_columns: list[int] | None = None


@dataclass(frozen=True)
class PdfInspectorExtractionResult:
    """Normalized extraction result from pdf-inspector."""

    pdf_type: str
    page_count: int
    pages_needing_ocr: list[int]
    has_encoding_issues: bool
    processing_time_ms: int | None
    markdown: str
    is_complex: bool | None = None
    pages_with_tables: list[int] | None = None
    pages_with_columns: list[int] | None = None


def detection_result_from_payload(
    payload: dict[str, object],
) -> PdfInspectorDetectionResult:
    (
        is_complex,
        pages_with_tables,
        pages_with_columns,
    ) = parse_analysis_fields(payload)
    return PdfInspectorDetectionResult(
        pdf_type=require_string(payload, "pdf_type"),
        page_count=require_positive_int(payload, "page_count"),
        pages_needing_ocr=require_pages_needing_ocr(payload),
        confidence=optional_float(payload.get("confidence")),
        has_encoding_issues=optional_bool(
            payload.get("has_encoding_issues"), default=False
        ),
        processing_time_ms=optional_non_negative_int(payload.get("processing_time_ms")),
        is_complex=is_complex,
        pages_with_tables=pages_with_tables,
        pages_with_columns=pages_with_columns,
    )


def extraction_result_from_payload(
    payload: dict[str, object],
) -> PdfInspectorExtractionResult:
    (
        is_complex,
        pages_with_tables,
        pages_with_columns,
    ) = parse_analysis_fields(payload)
    return PdfInspectorExtractionResult(
        pdf_type=require_string(payload, "pdf_type"),
        page_count=require_positive_int(payload, "page_count"),
        pages_needing_ocr=require_pages_needing_ocr(payload),
        has_encoding_issues=optional_bool(
            payload.get("has_encoding_issues"), default=False
        ),
        processing_time_ms=optional_non_negative_int(payload.get("processing_time_ms")),
        markdown=require_markdown(payload),
        is_complex=is_complex,
        pages_with_tables=pages_with_tables,
        pages_with_columns=pages_with_columns,
    )
