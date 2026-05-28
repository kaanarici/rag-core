"""Base converter interfaces for document conversion.

Provides:
- BaseConverter: protocol for all converters
- HybridConverter: extract-first / OCR-fallback coordination
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag_core.documents.converters.quality import (
    QualityScore,
    QualityVerdict,
    score_text_quality as score_text_quality,
)
from rag_core.documents.converters.text_helpers import (
    detect_encoding as detect_encoding,
    render_markdown_table as render_markdown_table,
    safe_decode as safe_decode,
    text_to_markdown as text_to_markdown,
)
from rag_core.documents.exception_names import exception_type

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result from a document converter.

    Attributes:
        content: Extracted markdown/text content.
        metadata: Format-specific metadata (parser name, page count, etc.).
        quality: Quality assessment of the extraction.
        needs_ocr: Whether OCR fallback is recommended.
        ocr_page_indices: For PDFs, specific page indices needing OCR
            (enables partial OCR for only the pages that need it).
    """

    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality: Optional[QualityScore] = None
    needs_ocr: bool = False
    ocr_page_indices: Optional[List[int]] = None


class BaseConverter(ABC):
    """Base class for all document converters.

    Each converter handles a specific format and converts file bytes
    to markdown text with metadata.
    """

    format_name: str = "unknown"

    @abstractmethod
    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert file bytes to markdown text.

        Args:
            file_bytes: Raw file content.
            filename: Original filename for extension detection.
            mime_type: MIME type of the file.

        Returns:
            ConversionResult with extracted text and metadata.
        """


def sanitized_error_value(exc: Exception) -> str:
    return exception_type(exc)


def sanitized_error_metadata(*, parser: str, exc: Exception) -> Dict[str, Any]:
    return {"parser": parser, "error": sanitized_error_value(exc)}


class HybridConverter(BaseConverter):
    """Converter that tries local extraction first, then OCR fallback.

    Subclasses implement _try_extract for format-specific extraction.
    The shared convert method handles the extract-first / OCR-fallback
    coordination, including partial OCR for PDFs (per-page, not whole-doc).
    """

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert with extract-first, quality-score, OCR-fallback strategy."""
        try:
            result = await self._try_extract(file_bytes, filename, mime_type)
            if result.content and result.quality and result.quality.verdict == QualityVerdict.GOOD:
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

        except Exception as exc:
            logger.warning(
                "%s extraction failed with %s; recommending OCR",
                self.format_name,
                exception_type(exc),
            )
            return ConversionResult(
                needs_ocr=True,
                metadata=sanitized_error_metadata(
                    parser="local:%s" % self.format_name,
                    exc=exc,
                ),
            )

    @abstractmethod
    async def _try_extract(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Attempt local text extraction for a single file.

        Returns ConversionResult with quality scoring.
        The caller uses quality.verdict to decide OCR fallback.
        """
