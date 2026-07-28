"""PDF page extraction helpers for the local converter path."""

from __future__ import annotations

import asyncio
from bisect import bisect_right
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, cast

from ..pdf_text_quality import normalize_pdf_extracted_text
from ..pdf_limits import validate_pdf_page_count

logger = logging.getLogger(__name__)

_MIN_CHARS_PER_PAGE = 50
_MIN_CHARS_IMAGE_PAGE = 200


def _is_encrypted_pdf_open_error(exc: Exception) -> bool:
    """Best-effort encrypted-PDF detection at the PyMuPDF boundary."""
    exc_name = type(exc).__name__.lower()
    if "password" in exc_name or "encrypted" in exc_name:
        return True
    exc_message = str(exc).lower()
    return "password" in exc_message or "encrypted" in exc_message


class PdfPageLike(Protocol):
    def get_text(self, mode: str) -> object: ...

    def get_images(self) -> Sequence[object]: ...


@dataclass
class PageExtraction:
    """Result of text extraction from a single PDF page."""

    page_num: int
    text: str = ""
    needs_ocr: bool = False
    char_count: int = 0
    has_garbled_text: bool = False


@dataclass
class PdfExtraction:
    """Result of extracting text from an entire PDF."""

    pages: list[PageExtraction] = field(default_factory=list)
    page_count: int = 0
    is_encrypted: bool = False

    @property
    def text_pages(self) -> list[PageExtraction]:
        """Pages with sufficient extracted text."""
        return [page for page in self.pages if not page.needs_ocr]

    @property
    def ocr_page_indices(self) -> list[int]:
        """0-based indices of pages needing OCR."""
        return [page.page_num for page in self.pages if page.needs_ocr]

    @property
    def full_text(self) -> str:
        """Combined text from all extracted pages (including partial)."""
        parts = [page.text for page in self.pages if page.text]
        return "\n\n".join(parts)

    @property
    def extraction_ratio(self) -> float:
        """Fraction of pages successfully extracted."""
        if not self.pages:
            return 0.0
        return len(self.text_pages) / len(self.pages)


def _extract_page(page: PdfPageLike, page_num: int) -> PageExtraction:
    """Extract text from a single PDF page with quality check."""
    try:
        raw_text = _extract_page_layout_text(page)
        text, has_garbled_text = normalize_pdf_extracted_text(raw_text)
        char_count = len(text.strip())

        if has_garbled_text:
            return PageExtraction(
                page_num=page_num,
                text=text,
                needs_ocr=True,
                char_count=char_count,
                has_garbled_text=True,
            )

        if char_count < _MIN_CHARS_PER_PAGE:
            return PageExtraction(
                page_num=page_num,
                text=text,
                needs_ocr=True,
                char_count=char_count,
            )

        # Images present with minimal text suggests a scan.
        image_list = page.get_images()
        if image_list and char_count < _MIN_CHARS_IMAGE_PAGE:
            return PageExtraction(
                page_num=page_num,
                text=text,
                needs_ocr=True,
                char_count=char_count,
            )

        return PageExtraction(
            page_num=page_num,
            text=text,
            needs_ocr=False,
            char_count=char_count,
        )

    except Exception as exc:
        logger.warning(
            "Text extraction failed for page %d with %s",
            page_num,
            type(exc).__name__,
        )
        return PageExtraction(page_num=page_num, needs_ocr=True)


def _extract_page_layout_text(page: PdfPageLike) -> str:
    try:
        raw_blocks = cast(Any, page).get_text("blocks", sort=False)
    except TypeError:
        return str(page.get_text("text") or "")
    if not isinstance(raw_blocks, Sequence) or isinstance(raw_blocks, str | bytes):
        return str(raw_blocks or "")

    blocks: list[tuple[float, float, float, float, str, int]] = []
    for block in raw_blocks:
        if not isinstance(block, Sequence) or isinstance(block, str | bytes):
            continue
        if len(block) < 5:
            continue
        if len(block) >= 7 and block[6] != 0:
            continue
        block_text = str(block[4] or "").strip("\r\n")
        if block_text.strip():
            try:
                blocks.append(
                    (
                        float(block[0]),
                        float(block[1]),
                        float(block[2]),
                        float(block[3]),
                        block_text,
                        int(block[5]) if len(block) >= 6 else len(blocks),
                    )
                )
            except (TypeError, ValueError):
                continue
    if blocks:
        return "\n\n".join(block[4] for block in _order_text_blocks(blocks))
    return str(page.get_text("text") or "")


def _order_text_blocks(
    blocks: list[tuple[float, float, float, float, str, int]],
) -> list[tuple[float, float, float, float, str, int]]:
    reading_order = sorted(blocks, key=lambda block: (block[1], block[0], block[5]))
    if len(blocks) < 4:
        return reading_order

    min_x = min(block[0] for block in blocks)
    max_x = max(block[2] for block in blocks)
    page_width = max_x - min_x
    widths = sorted(max(0.0, block[2] - block[0]) for block in blocks)
    median_width = widths[len(widths) // 2]
    spanning = [
        block
        for block in blocks
        if page_width > 0
        and block[2] - block[0] >= page_width * 0.72
        and block[2] - block[0] >= median_width * 1.4
    ]
    if spanning:
        ordered_with_spans, used_columns = _order_regions_around_spanning_blocks(
            blocks,
            spanning,
        )
        if used_columns:
            return ordered_with_spans

    ordered, _ = _order_column_region(blocks)
    return ordered


def _order_regions_around_spanning_blocks(
    blocks: list[tuple[float, float, float, float, str, int]],
    spanning: list[tuple[float, float, float, float, str, int]],
) -> tuple[
    list[tuple[float, float, float, float, str, int]],
    bool,
]:
    spans = sorted(spanning, key=lambda block: (block[1], block[0], block[5]))
    span_ids = {id(block) for block in spans}
    remaining = [block for block in blocks if id(block) not in span_ids]
    regions: list[list[tuple[float, float, float, float, str, int]]] = [
        [] for _ in range(len(spans) + 1)
    ]
    span_centers = [(block[1] + block[3]) / 2 for block in spans]
    for block in remaining:
        center_y = (block[1] + block[3]) / 2
        region_index = bisect_right(span_centers, center_y)
        regions[region_index].append(block)

    ordered: list[tuple[float, float, float, float, str, int]] = []
    used_columns = False
    for index, span in enumerate(spans):
        region, region_used_columns = _order_column_region(regions[index])
        ordered.extend(region)
        ordered.append(span)
        used_columns = used_columns or region_used_columns
    tail, tail_used_columns = _order_column_region(regions[-1])
    ordered.extend(tail)
    return ordered, used_columns or tail_used_columns


def _order_column_region(
    blocks: list[tuple[float, float, float, float, str, int]],
) -> tuple[
    list[tuple[float, float, float, float, str, int]],
    bool,
]:
    reading_order = sorted(blocks, key=lambda block: (block[1], block[0], block[5]))
    if len(blocks) < 4:
        return reading_order, False

    by_center = sorted(
        blocks,
        key=lambda block: ((block[0] + block[2]) / 2, block[1], block[5]),
    )
    count = len(by_center)
    prefix_max_x2 = [0.0] * count
    prefix_min_y = [0.0] * count
    prefix_max_y = [0.0] * count
    suffix_min_x0 = [0.0] * count
    suffix_min_y = [0.0] * count
    suffix_max_y = [0.0] * count
    for index, block in enumerate(by_center):
        if index == 0:
            prefix_max_x2[index] = block[2]
            prefix_min_y[index] = block[1]
            prefix_max_y[index] = block[3]
        else:
            prefix_max_x2[index] = max(prefix_max_x2[index - 1], block[2])
            prefix_min_y[index] = min(prefix_min_y[index - 1], block[1])
            prefix_max_y[index] = max(prefix_max_y[index - 1], block[3])
    for index in range(count - 1, -1, -1):
        block = by_center[index]
        if index == count - 1:
            suffix_min_x0[index] = block[0]
            suffix_min_y[index] = block[1]
            suffix_max_y[index] = block[3]
        else:
            suffix_min_x0[index] = min(suffix_min_x0[index + 1], block[0])
            suffix_min_y[index] = min(suffix_min_y[index + 1], block[1])
            suffix_max_y[index] = max(suffix_max_y[index + 1], block[3])

    best_split: int | None = None
    best_gap = 0.0
    for split in range(2, count - 1):
        gap = suffix_min_x0[split] - prefix_max_x2[split - 1]
        if gap <= best_gap:
            continue
        vertical_overlap = min(
            prefix_max_y[split - 1],
            suffix_max_y[split],
        ) - max(
            prefix_min_y[split - 1],
            suffix_min_y[split],
        )
        if gap >= 18.0 and vertical_overlap > 0:
            best_gap = gap
            best_split = split

    if best_split is None:
        return reading_order, False
    left = by_center[:best_split]
    right = by_center[best_split:]
    return (
        [
            *sorted(left, key=lambda block: (block[1], block[0], block[5])),
            *sorted(right, key=lambda block: (block[1], block[0], block[5])),
        ],
        True,
    )


async def extract_pdf(file_bytes: bytes) -> PdfExtraction:
    """Extract text from PDF using PyMuPDF with per-page quality detection.

    Raises:
        ImportError: If PyMuPDF (fitz) is not installed.
    """
    import fitz

    def _extract() -> PdfExtraction:
        result = PdfExtraction()
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            if _is_encrypted_pdf_open_error(exc):
                result.is_encrypted = True
                return result
            raise

        try:
            if getattr(doc, "needs_pass", False):
                result.is_encrypted = True
                return result
            result.page_count = validate_pdf_page_count(len(doc)) if len(doc) else 0
            for page_num in range(result.page_count):
                page = doc[page_num]
                extraction = _extract_page(page, page_num)
                result.pages.append(extraction)
        finally:
            doc.close()

        return result

    return await asyncio.to_thread(_extract)
