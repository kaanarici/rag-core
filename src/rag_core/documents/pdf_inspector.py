"""Adapters for Firecrawl's pdf-inspector wheel and CLI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
import logging

from rag_core.documents.pdf_inspector_payloads import require_positive_int
from rag_core.documents.pdf_inspector_payloads import (
    PdfInspectorDetectionResult,
    PdfInspectorExtractionResult,
    detection_result_from_payload,
    extraction_result_from_payload,
)
from rag_core.documents.exception_names import exception_type
from rag_core.documents.page_indices import normalize_page_indices
from rag_core.documents import pdf_inspector_runtime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfInspectorProcessResult:
    """Normalized in-process pdf-inspector wheel result."""

    detection: PdfInspectorDetectionResult
    extraction: PdfInspectorExtractionResult
    has_explicit_ocr_page_info: bool


def process_pdf_with_inspector_wheel(file_bytes: bytes) -> PdfInspectorProcessResult | None:
    module = _import_pdf_inspector_wheel()
    if module is None:
        return None

    process_pdf_bytes = getattr(module, "process_pdf_bytes", None)
    if not callable(process_pdf_bytes):
        logger.warning("pdf-inspector wheel is missing process_pdf_bytes")
        return None

    try:
        return _process_wheel_result(process_pdf_bytes(file_bytes))
    except ValueError as exc:
        logger.warning(
            "pdf-inspector wheel payload was invalid: %s",
            exception_type(exc),
        )
        return None
    except Exception as exc:
        logger.warning(
            "pdf-inspector wheel processing failed: %s",
            exception_type(exc),
        )
        return None


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
    description = pdf_inspector_runtime.describe_pdf_inspector_runtime()
    description["wheel_available"] = _import_pdf_inspector_wheel() is not None
    description["adapter_order"] = ["wheel", "cli", "pymupdf"]
    return description


def _import_pdf_inspector_wheel() -> object | None:
    try:
        return importlib.import_module("pdf_inspector")
    except ImportError:
        return None


def _process_wheel_result(result: object) -> PdfInspectorProcessResult:
    page_count = require_positive_int(
        {"page_count": _get_wheel_field(result, "page_count")},
        "page_count",
    )
    pages_needing_ocr, has_explicit_ocr_page_info = _wheel_ocr_page_numbers(
        result,
        page_count=page_count,
    )
    is_complex = _get_wheel_field(result, "is_complex_layout")
    if is_complex is None:
        is_complex = _get_wheel_field(result, "is_complex")
    payload: dict[str, object] = {
        "pdf_type": _get_wheel_field(result, "pdf_type"),
        "page_count": page_count,
        "pages_needing_ocr": pages_needing_ocr,
        "confidence": _get_wheel_field(result, "confidence"),
        "has_encoding_issues": _get_wheel_field(result, "has_encoding_issues"),
        "processing_time_ms": _get_wheel_field(result, "processing_time_ms"),
        "is_complex": is_complex,
        "pages_with_tables": _optional_wheel_page_numbers(
            result,
            "pages_with_tables",
            page_count=page_count,
        ),
        "pages_with_columns": _optional_wheel_page_numbers(
            result,
            "pages_with_columns",
            page_count=page_count,
        ),
    }
    detection = detection_result_from_payload(payload)
    extraction = extraction_result_from_payload(
        {
            **payload,
            "markdown": _get_wheel_field(result, "markdown"),
        }
    )
    return PdfInspectorProcessResult(
        detection=detection,
        extraction=extraction,
        has_explicit_ocr_page_info=has_explicit_ocr_page_info,
    )


def _wheel_ocr_page_numbers(
    result: object,
    *,
    page_count: int,
) -> tuple[list[int], bool]:
    for field_name in (
        "pages_needing_ocr",
        "ocr_page_indices",
        "pages_requiring_ocr",
        "ocr_required_pages",
        "ocr_pages",
    ):
        raw_value = _get_wheel_field(result, field_name)
        if isinstance(raw_value, list):
            return _page_numbers(
                normalize_page_indices(raw_value, page_count=page_count)
            ), True

    pdf_type = _get_wheel_field(result, "pdf_type")
    if not isinstance(pdf_type, str):
        return [], False
    route_key = "".join(char for char in pdf_type.lower() if char.isalnum())
    if route_key in {"scanned", "imagebased", "imageheavy", "imageonly"}:
        return _page_numbers(list(range(page_count))), False
    return [], False


def _optional_wheel_page_numbers(
    result: object,
    field_name: str,
    *,
    page_count: int,
) -> list[int] | None:
    raw_value = _get_wheel_field(result, field_name)
    if raw_value is None:
        return None
    if not isinstance(raw_value, list):
        raise ValueError(f"{field_name} must be a list when present")
    return _page_numbers(normalize_page_indices(raw_value, page_count=page_count))


def _get_wheel_field(result: object, name: str) -> object | None:
    if isinstance(result, Mapping):
        return result.get(name)
    return getattr(result, name, None)


def _page_numbers(indices: list[int]) -> list[int]:
    return [index + 1 for index in indices]
