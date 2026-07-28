from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from typing import NamedTuple, TypeGuard

from rag_core.core_models import PreparedChunk
from rag_core.documents.markdown_fences import (
    MarkdownFenceState,
    advance_markdown_fence,
)
from rag_core.documents.markdown_headings import parse_atx_heading
from rag_core.documents.converters.registry_maps import is_registered_pdf_document

# Exact level-two page headings are reserved for converter-owned boundaries.
_PDF_PAGE_HEADING_LINE_RE = re.compile(r"## Page ([1-9]\d*)")
_SOURCE_PAGE_TITLE_RE = re.compile(r"page[^\S\r\n]+[0-9]+", re.IGNORECASE)


class _PageHeading(NamedTuple):
    page_number: int
    start_char: int
    end_char: int


class _PageRange(NamedTuple):
    page_number: int
    start_char: int
    end_char: int


def normalize_pdf_page_body(markdown: str) -> str:
    """Escape source ATX headings whose title collides with a page boundary."""
    normalized: list[str] = []
    fence: MarkdownFenceState | None = None
    for raw_line in markdown.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        ending = raw_line[len(line) :]
        transitioned, fence = advance_markdown_fence(line, fence)
        heading = None if transitioned or fence is not None else parse_atx_heading(line)
        if heading is None or not _is_pdf_page_title(heading.title):
            normalized.append(raw_line)
        else:
            marker_start = len(heading.indent)
            escaped = (
                line[:marker_start]
                + r"\#" * heading.level
                + line[marker_start + heading.level :]
            )
            normalized.append(f"{escaped}{ending}")
    result = "".join(normalized)
    if fence is not None:
        separator = "" if result.endswith(("\n", "\r")) else "\n"
        result = f"{result}{separator}{fence.marker * fence.length}"
    return result


def _is_pdf_page_title(title: str) -> bool:
    return _SOURCE_PAGE_TITLE_RE.fullmatch(title) is not None


def canonicalize_pdf_page_markdown(
    markdown: str,
    expected_page_numbers: Sequence[int],
) -> str | None:
    headings = _canonical_pdf_page_headings(markdown)
    actual = [heading.page_number for heading in headings]
    expected = list(expected_page_numbers)
    if actual != expected or (
        headings and markdown[: headings[0].start_char].strip("\r\n")
    ):
        return None

    sections: list[str] = []
    for index, heading in enumerate(headings):
        body_end = (
            headings[index + 1].start_char
            if index + 1 < len(headings)
            else len(markdown)
        )
        sections.append(
            render_owned_pdf_page(
                markdown[heading.end_char : body_end],
                page_number=heading.page_number,
            )
        )
    return "\n\n".join(sections)


def render_owned_pdf_page(
    markdown: str,
    *,
    page_number: int,
    consume_leading_boundary: bool = False,
) -> str:
    body = markdown.strip("\r\n")
    first_line, separator, remainder = body.partition("\n")
    if (
        consume_leading_boundary
        and _canonical_pdf_page_number(first_line.rstrip("\r")) == page_number
    ):
        body = remainder.lstrip("\r\n") if separator else ""
    normalized_body = normalize_pdf_page_body(body)
    heading = f"## Page {page_number}"
    return f"{heading}\n\n{normalized_body}" if normalized_body else heading


def _canonical_pdf_page_number(line: str) -> int | None:
    match = _PDF_PAGE_HEADING_LINE_RE.fullmatch(line)
    return int(match.group(1)) if match is not None else None


def split_canonical_pdf_page_sections(
    markdown: str,
) -> tuple[str, list[tuple[int, str]]]:
    headings = _canonical_pdf_page_headings(markdown)
    if not headings:
        return "", []

    prefix = markdown[: headings[0].start_char].strip("\r\n")
    sections: list[tuple[int, str]] = []
    for index, heading in enumerate(headings):
        section_end = (
            headings[index + 1].start_char
            if index + 1 < len(headings)
            else len(markdown)
        )
        sections.append(
            (
                heading.page_number,
                markdown[heading.start_char : section_end].strip("\r\n"),
            )
        )
    return prefix, sections


def _canonical_pdf_page_headings(markdown: str) -> list[_PageHeading]:
    headings: list[_PageHeading] = []
    fence: MarkdownFenceState | None = None
    offset = 0
    for raw_line in markdown.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        transitioned, fence = advance_markdown_fence(line, fence)
        if not transitioned and fence is None:
            page_number = _canonical_pdf_page_number(line)
            if page_number is not None:
                headings.append(
                    _PageHeading(
                        page_number=page_number,
                        start_char=offset,
                        end_char=offset + len(line),
                    )
                )
        offset += len(raw_line)
    return headings


def with_pdf_page_locators(
    *,
    text: str,
    chunks: Sequence[PreparedChunk],
    mime_type: str | None,
    filename: str | None,
) -> list[PreparedChunk]:
    resolved_chunks = list(chunks)
    if not is_registered_pdf_document(mime_type=mime_type, filename=filename):
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


def _is_int_offset(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _pdf_page_ranges(text: str) -> list[_PageRange]:
    headings = _canonical_pdf_page_headings(text)
    if not headings:
        return []

    ranges: list[_PageRange] = []
    first_page_number = headings[0].page_number
    first_heading_start = headings[0].start_char
    if first_heading_start > 0 and text[:first_heading_start].strip():
        ranges.append(
            _PageRange(
                page_number=first_page_number,
                start_char=0,
                end_char=first_heading_start,
            )
        )
    for index, heading in enumerate(headings):
        end_char = (
            headings[index + 1].start_char if index + 1 < len(headings) else len(text)
        )
        ranges.append(
            _PageRange(
                page_number=heading.page_number,
                start_char=heading.start_char,
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


__all__ = [
    "canonicalize_pdf_page_markdown",
    "normalize_pdf_page_body",
    "render_owned_pdf_page",
    "with_pdf_page_locators",
]
