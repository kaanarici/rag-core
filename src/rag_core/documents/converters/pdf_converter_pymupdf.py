"""PyMuPDF extraction result assembly for PDF conversion."""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    ConversionResult,
    QualityScore,
    QualityVerdict,
    score_text_quality,
    text_to_markdown,
)
from .pdf_converter_extraction import PdfExtraction

_PYMUPDF_PATH = "local:pymupdf"


def pymupdf_conversion_result(
    extraction: PdfExtraction,
    *,
    logger: logging.Logger,
) -> ConversionResult:
    metadata: dict[str, Any] = {
        "parser": _PYMUPDF_PATH,
        "page_count": extraction.page_count,
    }

    if extraction.is_encrypted:
        logger.info(
            "Encrypted PDF requires OCR; path=%s page_count=%d",
            _PYMUPDF_PATH,
            extraction.page_count,
        )
        return ConversionResult(
            needs_ocr=True,
            metadata={
                **metadata,
                "is_encrypted": True,
                "needs_ocr": True,
                "ocr_processed_entire_document": True,
            },
            quality=QualityScore(verdict=QualityVerdict.EMPTY, details="encrypted PDF"),
        )

    if not extraction.pages:
        return ConversionResult(
            metadata=metadata,
            quality=QualityScore(verdict=QualityVerdict.EMPTY, details="no pages"),
        )

    sections: list[str] = []
    for page in extraction.pages:
        if page.text:
            markdown = text_to_markdown(page.text)
            sections.append("## Page %d\n\n%s" % (page.page_num + 1, markdown))

    content = "\n\n".join(sections) if sections else ""
    quality = score_text_quality(
        content,
        page_count=extraction.page_count,
        min_chars=50,
        min_chars_per_page=20.0,
    )

    ocr_indices = extraction.ocr_page_indices
    garbled_page_indices = [
        page.page_num for page in extraction.pages if getattr(page, "has_garbled_text", False)
    ]
    metadata["needs_ocr"] = bool(ocr_indices)
    metadata["extraction_ratio"] = extraction.extraction_ratio
    metadata["ocr_page_count"] = len(ocr_indices)
    if garbled_page_indices:
        metadata["garbled_text_page_indices"] = garbled_page_indices

    if ocr_indices:
        logger.info(
            "PDF pages need OCR; path=%s ocr_page_count=%d page_count=%d partial_ocr=true",
            _PYMUPDF_PATH,
            len(ocr_indices),
            extraction.page_count,
        )

    return ConversionResult(
        content=content,
        metadata=metadata,
        quality=quality,
        needs_ocr=bool(ocr_indices),
        ocr_page_indices=ocr_indices if ocr_indices else None,
    )
