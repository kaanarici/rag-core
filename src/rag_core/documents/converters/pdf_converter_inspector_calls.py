"""PDF Inspector call wrappers with sanitized fallback logging."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from ..pdf_inspector import PdfInspectorDetectionResult, PdfInspectorExtractionResult
from .pdf_converter_inspector import _get_inspector_page_count, _get_inspector_route

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


def _exception_type(exc: Exception) -> str:
    return type(exc).__name__


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
            _exception_type(exc),
        )
        return None

    try:
        detection = await asyncio.to_thread(detect, file_bytes)
    except Exception as exc:
        logger.warning(
            "PDF Inspector detection failed; inspector_path=%s error_type=%s",
            _INSPECTOR_PATH,
            _exception_type(exc),
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
            _exception_type(exc),
        )
        return InspectorExtraction(result=None, failed=True)
