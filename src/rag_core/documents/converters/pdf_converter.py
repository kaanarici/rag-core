"""PDF extraction with page provenance and selective OCR routing."""

from __future__ import annotations
import logging
from .base import BaseConverter, ConversionResult, QualityVerdict
from .pdf_converter_extraction import extract_pdf
from .pdf_converter_pymupdf import pymupdf_conversion_result
from rag_core.documents.exception_names import root_exception_type

logger = logging.getLogger(__name__)


class PdfConverter(BaseConverter):
    format_name = "pdf"

    async def convert(
        self, file_bytes: bytes, filename: str, mime_type: str
    ) -> ConversionResult:
        try:
            extraction = await extract_pdf(file_bytes)
            result = pymupdf_conversion_result(extraction, logger=logger)
        except Exception as exc:
            error_type = root_exception_type(exc)
            logger.warning("pdf extraction failed with %s", error_type)
            raise ValueError(f"PDF parse failed ({error_type})") from None
        if (
            not result.content
            or result.quality is None
            or result.quality.verdict != QualityVerdict.GOOD
        ):
            result.needs_ocr = True
        result.metadata["needs_ocr"] = result.needs_ocr
        return result
