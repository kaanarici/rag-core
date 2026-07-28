from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_core.core_models import OcrRoutingSignal
from rag_core.documents.pdf_page_locators import split_canonical_pdf_page_sections

if TYPE_CHECKING:
    from rag_core.documents.ocr import OcrResult


def build_ocr_signal(metadata: dict[str, Any]) -> OcrRoutingSignal:
    return OcrRoutingSignal(
        needed=bool(metadata.get("needs_ocr")),
        page_indices=normalize_page_indices(metadata.get("ocr_page_indices")),
        confidence=coerce_float(metadata.get("confidence")),
        parser=coerce_str(metadata.get("parser")),
    )


def normalize_page_indices(raw_indices: Any) -> list[int]:
    if not isinstance(raw_indices, list):
        return []
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_index in raw_indices:
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index < 0
            or raw_index in seen
        ):
            continue
        seen.add(raw_index)
        normalized.append(raw_index)
    return sorted(normalized)


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def coerce_float_or_zero(value: Any) -> float:
    result = coerce_float(value)
    if result is None:
        return 0.0
    return result


def coerce_int_or_zero(value: Any) -> int:
    result = coerce_int(value)
    if result is None:
        return 0
    return result


def parse_quality_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    quality = metadata.get("quality")
    if isinstance(quality, dict):
        return quality
    return {}


def resolve_ocr_pages_used(
    *,
    parsed_metadata: dict[str, Any],
    ocr_result: OcrResult,
    requested_page_indices: list[int],
) -> list[int]:
    processed_pages = normalize_page_indices(ocr_result.pages_processed)
    if processed_pages:
        return processed_pages
    if bool(ocr_result.metadata.get("ocr_processed_entire_document")):
        page_count = resolve_document_page_count(
            parsed_metadata=parsed_metadata,
            ocr_metadata=ocr_result.metadata,
        )
        if page_count is not None:
            return list(range(page_count))
        return []
    return list(requested_page_indices)


def resolve_ocr_page_count(
    *,
    parsed_metadata: dict[str, Any],
    ocr_result: OcrResult,
    ocr_pages_used: list[int],
    requested_page_indices: list[int],
) -> int:
    if ocr_pages_used:
        return len(ocr_pages_used)
    explicit_count = coerce_int(ocr_result.metadata.get("ocr_pages_used_count"))
    if explicit_count is not None and explicit_count >= 0:
        return explicit_count
    if bool(ocr_result.metadata.get("ocr_processed_entire_document")):
        page_count = resolve_document_page_count(
            parsed_metadata=parsed_metadata,
            ocr_metadata=ocr_result.metadata,
        )
        if page_count is not None:
            return page_count
        return 0
    return len(requested_page_indices)


def resolve_document_page_count(
    *,
    parsed_metadata: dict[str, Any],
    ocr_metadata: dict[str, Any],
) -> int | None:
    for raw_value in (
        parsed_metadata.get("page_count"),
        ocr_metadata.get("page_count"),
        ocr_metadata.get("ocr_page_count"),
    ):
        page_count = coerce_int(raw_value)
        if page_count is not None and page_count >= 0:
            return page_count
    return None


def merge_markdown(base_markdown: str, ocr_result: OcrResult) -> str:
    if not ocr_result.markdown.strip():
        return base_markdown
    ocr_markdown = ocr_result.markdown.strip("\r\n")
    if ocr_result.merge_mode == "replace":
        return ocr_markdown
    if not base_markdown.strip():
        return ocr_markdown
    base = base_markdown.strip("\r\n")
    merged = _replace_page_sections(base, ocr_markdown)
    if merged is not None:
        return merged
    return f"{base}\n\n{ocr_markdown}"


def _replace_page_sections(base_markdown: str, ocr_markdown: str) -> str | None:
    base_prefix, base_sections = split_canonical_pdf_page_sections(base_markdown)
    _, ocr_sections = split_canonical_pdf_page_sections(ocr_markdown)
    if not base_sections or not ocr_sections:
        return None

    base_by_page = {page_number: section for page_number, section in base_sections}
    ocr_by_page = {page_number: section for page_number, section in ocr_sections}

    merged_sections: list[str] = []
    merged_page_numbers = sorted(set(base_by_page) | set(ocr_by_page))
    for page_number in merged_page_numbers:
        if page_number in ocr_by_page:
            merged_sections.append(ocr_by_page[page_number])
        else:
            merged_sections.append(base_by_page[page_number])

    parts = []
    if base_prefix:
        parts.append(base_prefix)
    parts.extend(merged_sections)
    return "\n\n".join(parts)
