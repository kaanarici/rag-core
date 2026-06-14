"""PDF converter with hybrid text extraction and selective OCR routing."""

from __future__ import annotations

import logging
from typing import Optional

from ..pdf_inspector import (
    detect_pdf_with_inspector,
    extract_pdf_with_inspector,
    pdf_inspector_enabled,
)
from .base import (
    ConversionResult,
    HybridConverter,
    QualityScore,
    QualityVerdict,
    score_text_quality,
)
from .converter_keys import PDF_CONVERTER_KEY
from .pdf_converter_extraction import extract_pdf
from .pdf_converter_inspector import (
    _get_inspector_markdown,
    _inspector_is_ocr_only_route,
    _inspector_is_text_based,
    _inspector_supports_page_level_routing,
)
from .pdf_converter_inspector_calls import (
    detect_with_pdf_inspector,
    extract_with_pdf_inspector,
)
from .pdf_converter_inspector_results import (
    inspector_mixed_result,
    inspector_ocr_only_result,
    inspector_text_result,
)
from .pdf_converter_pymupdf import pymupdf_conversion_result

logger = logging.getLogger(__name__)

_INSPECTOR_PATH = "local:pdf_inspector"
_PYMUPDF_PATH = "local:pymupdf"


class PdfConverter(HybridConverter):
    """Converts PDFs to markdown with per-page OCR tracking.

    The converter tracks which pages need OCR and which have good text so the
    ingest pipeline can OCR only the pages that need it.
    """

    format_name = PDF_CONVERTER_KEY

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        try:
            result = await self._try_extract(file_bytes, filename, mime_type)
        except Exception as exc:
            root = exc.__cause__ if isinstance(exc.__cause__, Exception) else exc
            logger.warning(
                "%s extraction failed with %s",
                self.format_name,
                type(root).__name__,
            )
            raise ValueError("PDF parse failed (%s)" % type(root).__name__) from exc

        if (
            result.content
            and result.quality
            and result.quality.verdict == QualityVerdict.GOOD
        ):
            logger.debug(
                "%s extracted via text layer (%d chars)",
                self.format_name,
                len(result.content),
            )
            return result

        if result.quality:
            logger.debug(
                "%s extraction quality %s; recommending OCR",
                self.format_name,
                result.quality.verdict.value,
            )
        result.needs_ocr = True
        return result

    async def _try_extract(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        inspector_result = await self._try_extract_with_inspector(file_bytes, filename)
        if inspector_result is not None:
            return inspector_result

        return await self._try_extract_with_pymupdf(file_bytes, filename, mime_type)

    async def _try_extract_with_inspector(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> Optional[ConversionResult]:
        detected = await detect_with_pdf_inspector(
            file_bytes,
            enabled=pdf_inspector_enabled,
            detect=detect_pdf_with_inspector,
            logger=logger,
        )
        if detected is None:
            return None

        detection = detected.result
        route = detected.route
        page_count = detected.page_count
        if _inspector_is_ocr_only_route(detection):
            return inspector_ocr_only_result(
                detection=detection,
                page_count=page_count,
            )

        if not _inspector_is_text_based(
            detection
        ) and not _inspector_supports_page_level_routing(detection):
            logger.info(
                "PDF Inspector classified PDF as route=%s; inspector_path=%s page_count=%d fallback_path=%s",
                route or "unknown",
                _INSPECTOR_PATH,
                page_count,
                _PYMUPDF_PATH,
            )
            return None

        extracted = await extract_with_pdf_inspector(
            file_bytes,
            extract=extract_pdf_with_inspector,
            route=route,
            page_count=page_count,
            logger=logger,
        )
        if extracted.failed:
            return None

        extraction = extracted.result
        markdown = _get_inspector_markdown(extraction)
        if _inspector_is_text_based(detection):
            if not markdown:
                logger.info(
                    "PDF Inspector returned no markdown; inspector_path=%s route=%s page_count=%d fallback_path=%s",
                    _INSPECTOR_PATH,
                    route or "unknown",
                    page_count,
                    _PYMUPDF_PATH,
                )
                return None

            return inspector_text_result(
                markdown=markdown,
                detection=detection,
                extraction=extraction,
                page_count=page_count,
            )

        if not _inspector_supports_page_level_routing(detection):
            logger.info(
                "PDF Inspector classified PDF as route=%s; inspector_path=%s page_count=%d fallback_path=%s",
                route or "unknown",
                _INSPECTOR_PATH,
                page_count,
                _PYMUPDF_PATH,
            )
            return None
        if not markdown:
            logger.info(
                "PDF Inspector returned no markdown; inspector_path=%s route=%s page_count=%d fallback_path=%s",
                _INSPECTOR_PATH,
                route or "unknown",
                page_count,
                _PYMUPDF_PATH,
            )
            return None

        return inspector_mixed_result(
            markdown=markdown,
            detection=detection,
            extraction=extraction,
            page_count=page_count,
        )

    async def _try_extract_with_pymupdf(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Extract text from PDF with per-page quality scoring."""
        extraction = await extract_pdf(file_bytes)
        return pymupdf_conversion_result(extraction, logger=logger)
