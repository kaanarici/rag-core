"""PDF Inspector payload + result value types (detection/extraction data the inspector returns)."""

from __future__ import annotations

import math
from dataclasses import dataclass


def require_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value

def require_markdown(payload: dict[str, object]) -> str:
    value = payload.get("markdown")
    if not isinstance(value, str):
        raise ValueError("markdown must be a string")
    return value

def require_positive_int(payload: dict[str, object], key: str) -> int:
    value = _require_int(payload, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value

def optional_non_negative_int(value: object) -> int | None:
    parsed = _optional_int(value)
    if parsed is None or parsed < 0:
        return None
    return parsed

def optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None

def parse_analysis_fields(
    payload: dict[str, object],
) -> tuple[bool | None, list[int] | None, list[int] | None]:
    pages_with_tables = _optional_page_indices(
        payload.get("pages_with_tables"),
        field_name="pages_with_tables",
    )
    pages_with_columns = _optional_page_indices(
        payload.get("pages_with_columns"),
        field_name="pages_with_columns",
    )
    is_complex = _optional_nullable_bool(payload.get("is_complex"))

    if is_complex is None and pages_with_tables is not None and pages_with_columns is not None:
        is_complex = bool(pages_with_tables or pages_with_columns)

    return is_complex, pages_with_tables, pages_with_columns

def optional_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default

def require_pages_needing_ocr(payload: dict[str, object]) -> list[int]:
    raw_pages = payload.get("pages_needing_ocr", [])
    if not isinstance(raw_pages, list):
        raise ValueError("pages_needing_ocr must be a list")

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_page in raw_pages:
        page_number = _require_positive_page_number(
            raw_page,
            field_name="pages_needing_ocr",
        )
        page_index = page_number - 1
        if page_index in seen:
            continue
        seen.add(page_index)
        normalized.append(page_index)
    return normalized

def _require_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    coerced = _coerce_int(value)
    if coerced is None:
        raise ValueError(f"{key} must be an integer")
    return coerced

def _optional_int(value: object) -> int | None:
    return _coerce_int(value)

def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None

def _optional_nullable_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None

def _optional_page_indices(value: object, *, field_name: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list when present")

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_page in value:
        page_number = _require_positive_page_number(raw_page, field_name=field_name)
        page_index = page_number - 1
        if page_index in seen:
            continue
        seen.add(page_index)
        normalized.append(page_index)
    return normalized

def _require_positive_page_number(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} entries must be positive integers")
    return value


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
