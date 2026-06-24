"""Inspector detect/extract calls and the conversion-result builders for inspector routes."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from rag_core.documents.exception_names import exception_type
from ..pdf_inspector import PdfInspectorDetectionResult, PdfInspectorExtractionResult
from .pdf_converter_inspector import _get_inspector_page_count, _get_inspector_route
from typing import Any
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
    _normalize_inspector_ocr_page_indices,
    _telemetry_page_indices,
)


_INSPECTOR_PATH = "local:pdf_inspector"

@dataclass(frozen=True)
class InspectorDetection:
    result: PdfInspectorDetectionResult | None
    route: str
    page_count: int

@dataclass(frozen=True)
class InspectorExtraction:
    result: PdfInspectorExtractionResult | None
    failed: bool = False

async def detect_with_pdf_inspector(
    file_bytes: bytes,
    *,
    enabled: Callable[[], bool],
    detect: Callable[[bytes], PdfInspectorDetectionResult | None],
    logger: logging.Logger,
) -> InspectorDetection | None:
    try:
        if not enabled():
            return None
    except Exception as exc:
        logger.warning(
            "PDF Inspector availability check failed; inspector_path=%s error_type=%s",
            _INSPECTOR_PATH,
            exception_type(exc),
        )
        return None

    try:
        detection = await asyncio.to_thread(detect, file_bytes)
    except Exception as exc:
        logger.warning(
            "PDF Inspector detection failed; inspector_path=%s error_type=%s",
            _INSPECTOR_PATH,
            exception_type(exc),
        )
        return None
    return InspectorDetection(
        result=detection,
        route=_get_inspector_route(detection),
        page_count=_get_inspector_page_count(detection) or 1,
    )

async def extract_with_pdf_inspector(
    file_bytes: bytes,
    *,
    extract: Callable[[bytes], PdfInspectorExtractionResult | None],
    route: str,
    page_count: int,
    logger: logging.Logger,
) -> InspectorExtraction:
    try:
        return InspectorExtraction(result=await asyncio.to_thread(extract, file_bytes))
    except Exception as exc:
        logger.warning(
            "PDF Inspector extraction failed; inspector_path=%s route=%s page_count=%d error_type=%s",
            _INSPECTOR_PATH,
            route or "unknown",
            page_count,
            exception_type(exc),
        )
        return InspectorExtraction(result=None, failed=True)


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
