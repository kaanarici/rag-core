"""PDF Inspector conversion result assembly."""

from __future__ import annotations

from typing import Any

from ..pdf_inspector import PdfInspectorExtractionResult
from .base import (
    ConversionResult,
    QualityScore,
    QualityVerdict,
    score_text_quality,
)
from .pdf_converter_inspector import (
    InspectorResult,
    _apply_inspector_analysis_metadata,
    _get_inspector_field,
    _get_inspector_metadata,
    _get_inspector_page_count,
    _normalize_inspector_ocr_page_indices,
    _telemetry_page_indices,
)

_INSPECTOR_PATH = "local:pdf_inspector"


def inspector_ocr_only_result(
    *,
    detection: InspectorResult | None,
    page_count: int,
) -> ConversionResult:
    ocr_page_indices = _normalize_inspector_ocr_page_indices(
        _get_inspector_field(detection, "pages_needing_ocr"),
        page_count=page_count,
        default_all_pages=True,
    )
    metadata: dict[str, Any] = {}
    metadata.update(_get_inspector_metadata(detection))
    metadata.update(
        {
            "parser": _INSPECTOR_PATH,
            "page_count": page_count,
            "needs_ocr": True,
            "extraction_ratio": 0.0,
            "ocr_page_count": len(ocr_page_indices),
        }
    )
    if ocr_page_indices:
        metadata["ocr_page_indices"] = ocr_page_indices
        telemetry_indices = _telemetry_page_indices(ocr_page_indices)
        if len(telemetry_indices) < len(ocr_page_indices):
            metadata["ocr_page_indices_telemetry"] = telemetry_indices
    _apply_inspector_analysis_metadata(
        metadata,
        detection=detection,
        extraction=None,
        ocr_page_indices=ocr_page_indices,
    )

    return ConversionResult(
        content="",
        metadata=metadata,
        quality=QualityScore(
            page_count=page_count,
            verdict=QualityVerdict.EMPTY,
            details="pdf inspector classified document as OCR-only",
        ),
        needs_ocr=True,
        ocr_page_indices=ocr_page_indices or None,
    )


def inspector_text_result(
    *,
    markdown: str,
    detection: InspectorResult | None,
    extraction: PdfInspectorExtractionResult | None,
    page_count: int,
) -> ConversionResult:
    resolved_page_count = _get_inspector_page_count(extraction, detection) or page_count
    quality = score_text_quality(
        markdown,
        page_count=resolved_page_count,
    )
    if quality.verdict == QualityVerdict.GOOD:
        quality.details = "pdf inspector canonical extraction"
    else:
        quality.details = "pdf inspector canonical extraction: %s" % quality.details

    metadata = _inspector_base_metadata(
        detection=detection,
        extraction=extraction,
        page_count=resolved_page_count,
        needs_ocr=False,
        extraction_ratio=1.0,
        ocr_page_indices=[],
    )
    return ConversionResult(
        content=markdown,
        metadata=metadata,
        quality=quality,
    )


def inspector_mixed_result(
    *,
    markdown: str,
    detection: InspectorResult | None,
    extraction: PdfInspectorExtractionResult | None,
    page_count: int,
) -> ConversionResult:
    resolved_page_count = _get_inspector_page_count(extraction, detection) or page_count
    extraction_ocr_page_indices = _normalize_inspector_ocr_page_indices(
        _get_inspector_field(extraction, "pages_needing_ocr"),
        page_count=resolved_page_count,
    )
    detection_ocr_page_indices = _normalize_inspector_ocr_page_indices(
        _get_inspector_field(detection, "pages_needing_ocr"),
        page_count=resolved_page_count,
    )
    ocr_page_indices = extraction_ocr_page_indices or detection_ocr_page_indices
    quality = score_text_quality(
        markdown,
        page_count=resolved_page_count,
        min_chars=max(20, resolved_page_count * 10),
        min_chars_per_page=10.0,
    )
    quality.details = "pdf inspector mixed extraction: %s" % quality.details
    if quality.verdict == QualityVerdict.EMPTY:
        quality.verdict = QualityVerdict.POOR

    metadata = _inspector_base_metadata(
        detection=detection,
        extraction=extraction,
        page_count=resolved_page_count,
        needs_ocr=bool(ocr_page_indices),
        extraction_ratio=max(
            0.0,
            (
                (resolved_page_count - len(ocr_page_indices)) / resolved_page_count
                if resolved_page_count
                else 0.0
            ),
        ),
        ocr_page_indices=ocr_page_indices,
    )
    return ConversionResult(
        content=markdown,
        metadata=metadata,
        quality=quality,
        needs_ocr=bool(ocr_page_indices),
        ocr_page_indices=ocr_page_indices or None,
    )


def _inspector_base_metadata(
    *,
    detection: InspectorResult | None,
    extraction: PdfInspectorExtractionResult | None,
    page_count: int,
    needs_ocr: bool,
    extraction_ratio: float,
    ocr_page_indices: list[int],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    metadata.update(_get_inspector_metadata(detection))
    metadata.update(_get_inspector_metadata(extraction))
    metadata.update(
        {
            "parser": _INSPECTOR_PATH,
            "page_count": page_count,
            "needs_ocr": needs_ocr,
            "extraction_ratio": extraction_ratio,
            "ocr_page_count": len(ocr_page_indices),
        }
    )
    if ocr_page_indices:
        metadata["ocr_page_indices"] = ocr_page_indices
        telemetry_indices = _telemetry_page_indices(ocr_page_indices)
        if len(telemetry_indices) < len(ocr_page_indices):
            metadata["ocr_page_indices_telemetry"] = telemetry_indices
    _apply_inspector_analysis_metadata(
        metadata,
        detection=detection,
        extraction=extraction,
        ocr_page_indices=ocr_page_indices,
    )
    return metadata
