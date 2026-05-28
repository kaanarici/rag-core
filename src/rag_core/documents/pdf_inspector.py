"""Subprocess wrapper for Firecrawl's pdf-inspector CLI."""

from __future__ import annotations

import logging

from rag_core.documents.pdf_inspector_results import (
    PdfInspectorDetectionResult,
    PdfInspectorExtractionResult,
    detection_result_from_payload,
    extraction_result_from_payload,
)
from rag_core.documents.exception_names import exception_type
from rag_core.documents import pdf_inspector_runtime

logger = logging.getLogger(__name__)


def detect_pdf_with_inspector(file_bytes: bytes) -> PdfInspectorDetectionResult | None:
    """Detect PDF type via pdf-inspector.

    Returns ``None`` when the integration is disabled, unavailable, or the CLI
    returns unusable output.
    """

    payload = pdf_inspector_runtime.run_pdf_inspector(
        ["detect-pdf", "--analyze", "--json"],
        file_bytes,
    )
    if payload is None:
        return None

    try:
        return detection_result_from_payload(payload)
    except ValueError as exc:
        logger.warning(
            "pdf-inspector detection payload was invalid: %s",
            exception_type(exc),
        )
        return None


def extract_pdf_with_inspector(file_bytes: bytes) -> PdfInspectorExtractionResult | None:
    """Extract PDF markdown via pdf-inspector."""

    payload = pdf_inspector_runtime.run_pdf_inspector(["pdf2md", "--json"], file_bytes)
    if payload is None:
        return None

    try:
        return extraction_result_from_payload(payload)
    except ValueError as exc:
        logger.warning(
            "pdf-inspector extraction payload was invalid: %s",
            exception_type(exc),
        )
        return None


def pdf_inspector_enabled() -> bool:
    return pdf_inspector_runtime.pdf_inspector_enabled()


def describe_pdf_inspector_runtime() -> dict[str, object]:
    return pdf_inspector_runtime.describe_pdf_inspector_runtime()
