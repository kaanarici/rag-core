from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from typing import NamedTuple, TypeGuard

from rag_core.core_models import PreparedChunk

# Locked to the canonical ``## Page N`` markup pymupdf and the OCR commands
# emit so stray ``# Page N`` lines in source text cannot misalign locators.
_PDF_PAGE_HEADING_RE = re.compile(r"(?m)^## Page ([1-9]\d*)$")


class _PageRange(NamedTuple):
    page_number: int
    start_char: int
    end_char: int


def with_pdf_page_locators(
    *,
    text: str,
    chunks: Sequence[PreparedChunk],
    mime_type: str | None,
    filename: str | None,
) -> list[PreparedChunk]:
    resolved_chunks = list(chunks)
    if not _looks_like_pdf(mime_type=mime_type, filename=filename):
        return resolved_chunks
    page_ranges = _pdf_page_ranges(text)
    if not page_ranges:
        return resolved_chunks

    annotated: list[PreparedChunk] = []
    for chunk in resolved_chunks:
        if chunk.metadata.get("offset_reconstruction") == "unreliable":
            annotated.append(chunk)
            continue
        start_char = chunk.start_char
        end_char = chunk.end_char
        if not _is_int_offset(start_char) or not _is_int_offset(end_char):
            annotated.append(chunk)
            continue
        page_number = _page_number_for_span(
            start_char=start_char,
            end_char=end_char,
            page_ranges=page_ranges,
        )
        if page_number is None:
            annotated.append(chunk)
            continue
        metadata = dict(chunk.metadata)
        metadata["page_number"] = page_number
        metadata["page_index"] = page_number - 1
        annotated.append(replace(chunk, metadata=metadata))
    return annotated


def _looks_like_pdf(*, mime_type: str | None, filename: str | None) -> bool:
    if mime_type and mime_type.lower().split(";", 1)[0].strip() == "application/pdf":
        return True
    return bool(filename and filename.lower().endswith(".pdf"))


def _is_int_offset(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _pdf_page_ranges(text: str) -> list[_PageRange]:
    matches = list(_PDF_PAGE_HEADING_RE.finditer(text))
    if not matches:
        return []

    ranges: list[_PageRange] = []
    first_page_number = int(matches[0].group(1))
    first_heading_start = matches[0].start()
    if first_heading_start > 0 and text[:first_heading_start].strip():
        ranges.append(
            _PageRange(
                page_number=first_page_number,
                start_char=0,
                end_char=first_heading_start,
            )
        )
    for index, match in enumerate(matches):
        end_char = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        ranges.append(
            _PageRange(
                page_number=int(match.group(1)),
                start_char=match.start(),
                end_char=end_char,
            )
        )
    return ranges


def _page_number_for_span(
    *,
    start_char: int,
    end_char: int,
    page_ranges: Sequence[_PageRange],
) -> int | None:
    best_page: int | None = None
    best_overlap = 0
    resolved_end = max(start_char, end_char)
    for page_range in page_ranges:
        overlap = max(
            0,
            min(resolved_end, page_range.end_char)
            - max(start_char, page_range.start_char),
        )
        if overlap > best_overlap:
            best_page = page_range.page_number
            best_overlap = overlap
    if best_page is not None:
        return best_page
    for page_range in page_ranges:
        if page_range.start_char <= start_char < page_range.end_char:
            return page_range.page_number
    return None


__all__ = ["with_pdf_page_locators"]
