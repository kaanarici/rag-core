"""DOCX converter with hybrid text extraction + OCR fallback.

Uses python-docx for text extraction with heading style detection,
table extraction, and quality scoring.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any, Dict, List, Optional

from .base import (
    ConversionResult,
    HybridConverter,
    render_markdown_table,
    score_text_quality,
)
from .quality import QualityVerdict, is_char_count_only_quality_failure

logger = logging.getLogger(__name__)

_HEADING_MAP = (
    ("heading 1", "# "),
    ("heading 2", "## "),
    ("heading 3", "### "),
    ("heading", "#### "),
)


def _format_paragraph(para: Any) -> Optional[str]:
    """Convert a DOCX paragraph to a markdown line using style detection."""
    text = str(getattr(para, "text", "") or "").strip()
    if not text:
        return None

    style_name = (para.style.name or "").lower() if para.style else ""

    for keyword, prefix in _HEADING_MAP:
        if keyword in style_name:
            return "%s%s" % (prefix, text)

    if "list" in style_name:
        return "- %s" % text

    return text


def _format_table(table: Any) -> str:
    """Convert a DOCX table to markdown."""
    rows: List[List[str]] = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append(cells)
    return render_markdown_table(rows)


def _extract_ordered_text_parts(doc: Any) -> List[str]:
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parts: List[str] = []
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            line = _format_paragraph(Paragraph(child, doc))
            if line:
                parts.append(line)
        elif isinstance(child, CT_Tbl):
            md_table = _format_table(Table(child, doc))
            if md_table:
                parts.append(md_table)
    return parts


def _extract_docx_figure_items(doc: Any) -> List[Dict[str, Any]]:
    """Extract lightweight figure metadata from DOCX inline shapes."""
    figures: List[Dict[str, Any]] = []
    for idx, shape in enumerate(getattr(doc, "inline_shapes", [])):
        figure_id = "fig:docx:%d" % (idx + 1)
        label = "DOCX Figure %d" % (idx + 1)
        description = ""
        try:
            doc_pr = shape._inline.docPr
            alt_text = (
                _doc_pr_attr(doc_pr, "descr")
                or _doc_pr_attr(doc_pr, "title")
                or ""
            )
            if alt_text:
                description = str(alt_text).strip()
        except Exception:
            # Inline alt text is optional; keep the figure when descriptor lookup fails.
            pass

        figures.append(
            {
                "figure_id": figure_id,
                "label": label,
                "description": description,
                "metadata": {
                    "source": "docx:inline_shape",
                },
            }
        )
    return figures


def _doc_pr_attr(doc_pr: Any, name: str) -> object:
    value = getattr(doc_pr, name, None)
    if value:
        return value
    get = getattr(doc_pr, "get", None)
    if callable(get):
        return get(name)
    return None


def _attach_single_docx_figure_locator(
    figure_items: List[Dict[str, Any]],
    *,
    content: str,
    text_parts: List[str],
) -> None:
    if len(figure_items) != 1 or not content.strip():
        return
    anchor = next((part.strip() for part in text_parts if part.strip()), "")
    if not anchor:
        return
    start = content.find(anchor)
    if start < 0:
        return
    metadata = dict(figure_items[0].get("metadata") or {})
    metadata["text_anchor_start_char"] = start
    metadata["text_anchor_end_char"] = start + len(anchor)
    figure_items[0]["metadata"] = metadata


class DocxConverter(HybridConverter):
    """Converts DOCX files to markdown with heading style detection."""

    format_name = "docx"

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
            raise ValueError("DOCX parse failed (%s)" % type(root).__name__) from exc
        quality = result.quality
        content = result.content.strip()
        if not content:
            result.needs_ocr = True
            result.metadata["needs_ocr"] = True
            return result
        if quality is not None and is_char_count_only_quality_failure(quality):
            result.needs_ocr = False
            result.metadata["needs_ocr"] = False
            result.metadata["quality_warning"] = "short_extracted_text"
            return result
        if quality is not None and quality.verdict == QualityVerdict.GOOD:
            result.needs_ocr = False
            result.metadata["needs_ocr"] = False
            return result
        result.needs_ocr = True
        result.metadata["needs_ocr"] = True
        return result

    async def _try_extract(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Extract text from DOCX using python-docx."""
        from docx import Document

        def _extract() -> ConversionResult:
            try:
                doc = Document(io.BytesIO(file_bytes))
            except Exception as exc:
                logger.warning(
                    "Failed to open Office document: format=%s error_type=%s",
                    self.format_name,
                    type(exc).__name__,
                )
                raise ValueError("DOCX parse failed (%s)" % type(exc).__name__) from exc

            text_parts = _extract_ordered_text_parts(doc)

            figure_items = _extract_docx_figure_items(doc)
            parts = list(text_parts)
            has_extracted_text = bool("\n\n".join(text_parts).strip())

            content = "\n\n".join(parts) if has_extracted_text else ""
            _attach_single_docx_figure_locator(
                figure_items,
                content=content,
                text_parts=text_parts,
            )
            quality = score_text_quality(content)

            metadata: Dict[str, Any] = {
                "parser": "local:python-docx",
                "needs_ocr": bool(figure_items and not has_extracted_text),
                "text_char_count": len("\n\n".join(text_parts).strip()),
            }
            if figure_items:
                metadata["figure_items"] = figure_items
                metadata["figure_count"] = len(figure_items)

            return ConversionResult(
                content=content,
                metadata=metadata,
                quality=quality,
            )

        return await asyncio.to_thread(_extract)
