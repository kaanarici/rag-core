"""PDF inspector helpers for routing and metadata."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, TypeAlias

from rag_core.documents.page_indices import normalize_page_indices
from rag_core.documents.pdf_page_locators import (
    render_owned_pdf_page,
)
from rag_core.documents.pdf_limits import validate_pdf_page_count

from ..pdf_inspector import PdfInspectorDetectionResult, PdfInspectorExtractionResult

logger = logging.getLogger(__name__)

_MAX_INSPECTOR_OCR_PAGE_INDICES_TELEMETRY = 400
_INSPECTOR_PAGE_COMMENT_RE = re.compile(
    r"^<!--[ \t]*Page\b.*?-->[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_TRUSTED_INSPECTOR_PAGE_MARKER_RE = re.compile(
    r"^<!--[ \t]*Page[ \t]+([1-9][0-9]*)[ \t]*-->[ \t]*$",
    re.MULTILINE,
)
InspectorResult: TypeAlias = PdfInspectorDetectionResult | PdfInspectorExtractionResult


class InspectorRouteKind(Enum):
    TEXT = "text"
    MIXED = "mixed"
    OCR_ONLY = "ocr_only"


@dataclass(frozen=True)
class InspectorOcrRouting:
    page_indices: tuple[int, ...]
    result_kind: InspectorRouteKind


def _normalize_inspector_route(route: str) -> str:
    return "".join(char for char in route.lower() if char.isalnum())


def _get_inspector_field(result: InspectorResult | None, name: str) -> object | None:
    if result is None:
        return None
    return getattr(result, name, None)


def _get_inspector_page_indices(result: InspectorResult | None, name: str) -> list[int]:
    raw_value = _get_inspector_field(result, name)
    if not isinstance(raw_value, list):
        return []
    return [
        value
        for value in raw_value
        if isinstance(value, int) and not isinstance(value, bool)
    ]


def _get_inspector_metadata(result: InspectorResult | None) -> Dict[str, object]:
    metadata: Dict[str, object] = {}
    confidence = _get_inspector_field(result, "confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        metadata["confidence"] = float(confidence)
    return metadata


def _get_inspector_page_count(*results: InspectorResult | None) -> int | None:
    for result in results:
        if result is not None and result.page_count > 0:
            return result.page_count
    return None


def _get_inspector_route(result: InspectorResult | None) -> str:
    if result is None:
        return ""
    return result.pdf_type.strip().lower()


def _inspector_route_kind(
    result: InspectorResult | None,
) -> InspectorRouteKind | None:
    route_key = _normalize_inspector_route(_get_inspector_route(result))
    if route_key in {
        "text",
        "textbased",
        "textnative",
        "nativetext",
        "digitaltext",
        "textual",
    }:
        return InspectorRouteKind.TEXT
    if route_key in {"mixed", "mixedcontent"}:
        return InspectorRouteKind.MIXED
    if route_key in {
        "ocronly",
        "scanned",
        "scannedsimple",
        "scannedcomplex",
        "imagebased",
        "imageheavy",
        "imageonly",
    }:
        return InspectorRouteKind.OCR_ONLY
    return None


def _get_inspector_markdown(result: PdfInspectorExtractionResult | None) -> str:
    if result is None:
        return ""
    if not result.markdown.strip():
        return ""
    return result.markdown.strip("\r\n")


def _canonicalize_inspector_markdown(
    markdown: str,
    *,
    page_count: int,
    trusted_page_markers: bool,
) -> str | None:
    if not markdown.strip():
        return ""
    stripped = markdown.strip("\r\n")
    if page_count == 1:
        if trusted_page_markers:
            markers = list(_TRUSTED_INSPECTOR_PAGE_MARKER_RE.finditer(stripped))
            comments = list(_INSPECTOR_PAGE_COMMENT_RE.finditer(stripped))
            if (
                len(markers) == len(comments) == 1
                and int(markers[0].group(1)) == 1
                and not stripped[: markers[0].start()].strip()
            ):
                return _render_trusted_inspector_pages(stripped, markers)
        return render_owned_pdf_page(stripped, page_number=1)
    if not trusted_page_markers:
        return None

    markers = list(_TRUSTED_INSPECTOR_PAGE_MARKER_RE.finditer(stripped))
    comments = list(_INSPECTOR_PAGE_COMMENT_RE.finditer(stripped))
    # Only parser-generated comments from the owned --pages invocation cross this seam.
    if (
        len(markers) != len(comments)
        or [int(marker.group(1)) for marker in markers]
        != list(range(1, page_count + 1))
        or not markers
        or stripped[: markers[0].start()].strip()
    ):
        return None
    return _render_trusted_inspector_pages(stripped, markers)


def _render_trusted_inspector_pages(
    markdown: str,
    markers: list[re.Match[str]],
) -> str:
    sections: list[str] = []
    for index, marker in enumerate(markers):
        body_end = (
            markers[index + 1].start() if index + 1 < len(markers) else len(markdown)
        )
        sections.append(
            render_owned_pdf_page(
                markdown[marker.end() : body_end],
                page_number=int(marker.group(1)),
            )
        )
    return "\n\n".join(sections)


def _reconciled_inspector_ocr_routing(
    *,
    detection: InspectorResult | None,
    extraction: PdfInspectorExtractionResult | None,
    page_count: int,
) -> InspectorOcrRouting | None:
    try:
        validate_pdf_page_count(page_count)
    except ValueError:
        return None
    if detection is None or detection.page_count != page_count:
        return None

    detection_kind = _inspector_route_kind(detection)
    extraction_kind = _inspector_route_kind(extraction)
    if detection_kind is None:
        return None
    if extraction is not None and (
        extraction.page_count != page_count or extraction_kind is None
    ):
        return None
    if detection_kind is InspectorRouteKind.OCR_ONLY or (
        extraction_kind is InspectorRouteKind.OCR_ONLY
    ):
        return InspectorOcrRouting(
            page_indices=tuple(range(page_count)),
            result_kind=InspectorRouteKind.OCR_ONLY,
        )
    if extraction is None:
        return None

    uses_page_routing = detection_kind is InspectorRouteKind.MIXED or (
        extraction_kind is InspectorRouteKind.MIXED
    )
    if (
        detection_kind is InspectorRouteKind.MIXED
        and not detection.has_explicit_ocr_page_info
    ) or (
        extraction_kind is InspectorRouteKind.MIXED
        and not extraction.has_explicit_ocr_page_info
    ):
        return None

    indices: set[int] = set()
    for result in (detection, extraction):
        raw_indices = _get_inspector_field(result, "pages_needing_ocr")
        normalized = normalize_page_indices(
            raw_indices,
            page_count=page_count,
        )
        if isinstance(raw_indices, list) and raw_indices and not normalized:
            return None
        indices.update(normalized)
    resolved = tuple(sorted(indices))
    return InspectorOcrRouting(
        page_indices=resolved,
        result_kind=(
            InspectorRouteKind.MIXED
            if uses_page_routing or resolved
            else InspectorRouteKind.TEXT
        ),
    )


def _apply_inspector_analysis_metadata(
    metadata: Dict[str, object],
    *,
    detection: InspectorResult | None,
    extraction: PdfInspectorExtractionResult | None,
    ocr_page_indices: List[int],
) -> None:
    raw_route = _get_inspector_route(detection)
    if raw_route:
        metadata["inspector_route"] = raw_route

    has_encoding_issues = _get_inspector_field(extraction, "has_encoding_issues")
    if not isinstance(has_encoding_issues, bool):
        has_encoding_issues = _get_inspector_field(detection, "has_encoding_issues")
    if isinstance(has_encoding_issues, bool):
        metadata["inspector_has_encoding_issues"] = has_encoding_issues

    layout_candidates = [
        value for value in (extraction, detection) if value is not None
    ]
    complex_pages = normalize_page_indices(
        [
            page_index
            for value in layout_candidates
            for page_index in (
                _get_inspector_page_indices(value, "pages_with_tables")
                + _get_inspector_page_indices(value, "pages_with_columns")
            )
        ],
        page_count=None,
    )
    if complex_pages:
        metadata["complex_ocr_page_indices"] = [
            page_index
            for page_index in complex_pages
            if page_index in set(ocr_page_indices)
        ]

    is_complex = _get_inspector_field(extraction, "is_complex")
    if not isinstance(is_complex, bool):
        is_complex = _get_inspector_field(detection, "is_complex")
    if isinstance(is_complex, bool):
        metadata["inspector_is_complex"] = is_complex

    route_key = _normalize_inspector_route(raw_route)
    if route_key in {"imagebased", "imageheavy", "imageonly"} and ocr_page_indices:
        metadata["image_only_page_indices"] = list(ocr_page_indices)


def _telemetry_page_indices(indices: List[int]) -> List[int]:
    if len(indices) <= _MAX_INSPECTOR_OCR_PAGE_INDICES_TELEMETRY:
        return list(indices)
    logger.info(
        "Capping inspector OCR telemetry page indices from %d to %d pages",
        len(indices),
        _MAX_INSPECTOR_OCR_PAGE_INDICES_TELEMETRY,
    )
    return list(indices[:_MAX_INSPECTOR_OCR_PAGE_INDICES_TELEMETRY])
