"""Base converter interfaces for document conversion.

Provides:
- BaseConverter: protocol for all converters
- HybridConverter: extract-first / OCR-fallback coordination
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag_core.documents.converters.quality import (
    QualityScore,
    QualityVerdict,
    is_char_count_only_quality_failure,
    score_text_quality as score_text_quality,
)
from rag_core.documents.converters.text_helpers import (
    detect_encoding as detect_encoding,
    render_markdown_table as render_markdown_table,
    safe_decode as safe_decode,
    text_to_markdown as text_to_markdown,
)
from rag_core.documents.exception_names import exception_type, root_exception_type

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


class TextLikeConverter(BaseConverter):
    """Base for text-family converters (CSV, JSON, XML, HTML, ...).

    Subclasses implement ``_render`` to turn decoded text into content plus
    format metadata. The shared ``convert`` runs the common scaffold off the
    event loop: decode via :func:`safe_decode` (binary input surfaces as an
    ``error`` metadata result), gate empty text to an empty-quality result, and
    attach :func:`score_text_quality` to the rendered content.

    ``_render`` owns its parser label and any ``needs_ocr`` flag, so a subclass
    can vary the parser name per fallback path (e.g. HTML's extractor chain).
    """

    parser_label: str = "local:unknown"

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        def _convert() -> ConversionResult:
            try:
                text = safe_decode(file_bytes)
            except ValueError as exc:
                return ConversionResult(
                    metadata={"parser": self.parser_label, "error": str(exc)},
                )

            if not text.strip():
                return ConversionResult(
                    metadata={"parser": self.parser_label},
                    quality=score_text_quality(""),
                )

            rendered = self._render(text, filename, mime_type)
            if isinstance(rendered, ConversionResult):
                return rendered

            content, metadata = rendered
            return ConversionResult(
                content=content,
                metadata=metadata,
                quality=score_text_quality(content),
            )

        return await asyncio.to_thread(_convert)

    @abstractmethod
    def _render(
        self,
        text: str,
        filename: str,
        mime_type: str,
    ) -> Tuple[str, Dict[str, Any]] | ConversionResult:
        """Render decoded, non-empty text to ``(content, metadata)``.

        The base attaches ``score_text_quality(content)``. Return a complete
        ``ConversionResult`` instead when a path needs its own quality (e.g. a
        parse-error fallback that scores raw text under a different metadata
        shape).
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

    async def _convert_office(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        *,
        parse_label: str,
    ) -> ConversionResult:
        """Extract-then-gate flow shared by the DOCX and PPTX converters.

        Extraction errors surface as a sanitized ``ValueError``. Short but
        otherwise clean extractions are kept (flagged ``short_extracted_text``)
        rather than forced through OCR.
        """
        try:
            result = await self._try_extract(file_bytes, filename, mime_type)
        except Exception as exc:
            root_type = root_exception_type(exc)
            logger.warning(
                "%s extraction failed with %s",
                self.format_name,
                root_type,
            )
            raise ValueError(
                "%s parse failed (%s)" % (parse_label, root_type)
            ) from exc

        quality = result.quality
        if not result.content.strip():
            needs_ocr = True
        elif quality is not None and is_char_count_only_quality_failure(quality):
            needs_ocr = False
            result.metadata["quality_warning"] = "short_extracted_text"
        elif quality is not None and quality.verdict == QualityVerdict.GOOD:
            needs_ocr = False
        else:
            needs_ocr = True
        result.needs_ocr = needs_ocr
        result.metadata["needs_ocr"] = needs_ocr
        return result
