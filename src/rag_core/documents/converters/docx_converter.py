"""DOCX converter with hybrid text extraction + OCR fallback.

Uses python-docx for text extraction with heading style detection,
table extraction, and quality scoring.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import (
    ConversionResult,
    HybridConverter,
    render_markdown_table,
    score_text_quality,
)
from .converter_keys import DOCX_CONVERTER_KEY

logger = logging.getLogger(__name__)

_HEADING_MAP = (
    ("heading 1", "# "),
    ("heading 2", "## "),
    ("heading 3", "### "),
    ("heading", "#### "),
)
_TABLE_FIGURE_MARKER_RE = re.compile("\ufdd0([0-9]+)\ufdd1")


@dataclass(frozen=True)
class _DocxTextPart:
    body_index: int
    text: str


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


def _format_table_with_figure_offsets(
    table: Any,
) -> tuple[str, dict[str, int]]:
    rows: List[List[str]] = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            paragraphs: list[str] = []
            for paragraph in cell.paragraphs:
                text = str(getattr(paragraph, "text", "") or "")
                for doc_pr_id, offset in sorted(
                    _raw_paragraph_figure_offsets(paragraph).items(),
                    key=lambda item: item[1],
                    reverse=True,
                ):
                    marker = f"\ufdd0{doc_pr_id}\ufdd1"
                    text = f"{text[:offset]}{marker}{text[offset:]}"
                paragraphs.append(text)
            cells.append("\n".join(paragraphs).strip())
        rows.append(cells)

    rendered = render_markdown_table(rows)
    clean_parts: list[str] = []
    offsets: dict[str, int] = {}
    cursor = 0
    clean_length = 0
    for match in _TABLE_FIGURE_MARKER_RE.finditer(rendered):
        visible = rendered[cursor : match.start()]
        clean_parts.append(visible)
        clean_length += len(visible)
        offsets[match.group(1)] = clean_length
        cursor = match.end()
    clean_parts.append(rendered[cursor:])
    return "".join(clean_parts), offsets


def _extract_ordered_text_parts(
    doc: Any,
) -> tuple[List[_DocxTextPart], dict[str, tuple[int, int]]]:
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parts: List[_DocxTextPart] = []
    figure_offsets: dict[str, tuple[int, int]] = {}
    for body_index, child in enumerate(doc.element.body.iterchildren()):
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, doc)
            line = _format_paragraph(paragraph)
            if line:
                parts.append(_DocxTextPart(body_index=body_index, text=line))
            for doc_pr_id, offset in _paragraph_figure_offsets(
                paragraph,
                formatted_text=line or "",
            ).items():
                figure_offsets[doc_pr_id] = (body_index, offset)
        elif isinstance(child, CT_Tbl):
            md_table, table_figure_offsets = _format_table_with_figure_offsets(
                Table(child, doc)
            )
            if md_table:
                parts.append(_DocxTextPart(body_index=body_index, text=md_table))
            for doc_pr_id, offset in table_figure_offsets.items():
                figure_offsets[doc_pr_id] = (body_index, offset)
    return parts, figure_offsets


def _paragraph_figure_offsets(
    paragraph: Any,
    *,
    formatted_text: str,
) -> dict[str, int]:
    raw_text = str(getattr(paragraph, "text", "") or "")
    visible_text = raw_text.strip()
    leading_trim = len(raw_text) - len(raw_text.lstrip())
    prefix_length = (
        len(formatted_text) - len(visible_text)
        if visible_text and formatted_text.endswith(visible_text)
        else 0
    )
    offsets: dict[str, int] = {}
    for doc_pr_id, raw_offset in _raw_paragraph_figure_offsets(paragraph).items():
        visible_offset = max(
            0,
            min(len(visible_text), raw_offset - leading_trim),
        )
        offsets[doc_pr_id] = prefix_length + visible_offset
    return offsets


def _raw_paragraph_figure_offsets(paragraph: Any) -> dict[str, int]:
    offsets: dict[str, int] = {}
    raw_offset = 0
    for node in paragraph._p.iter():
        local_name = str(node.tag).rpartition("}")[2]
        if local_name == "t":
            raw_offset += len(str(node.text or ""))
            continue
        if local_name in {"tab", "br", "cr"}:
            raw_offset += 1
            continue
        if local_name != "inline":
            continue
        doc_pr_id = _inline_doc_pr_id(node)
        if doc_pr_id is None:
            continue
        offsets[doc_pr_id] = raw_offset
    return offsets


def _inline_doc_pr_id(inline: Any) -> str | None:
    for node in inline.iter():
        if str(node.tag).rpartition("}")[2] != "docPr":
            continue
        value = _doc_pr_attr(node, "id")
        if value is not None:
            return str(value)
    return None


def _extract_docx_figure_items(
    doc: Any,
    *,
    figure_offsets: dict[str, tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Extract lightweight figure metadata from DOCX inline shapes."""
    figures: List[Dict[str, Any]] = []
    for idx, shape in enumerate(getattr(doc, "inline_shapes", [])):
        figure_id = "fig:docx:%d" % (idx + 1)
        label = "DOCX Figure %d" % (idx + 1)
        description = ""
        paragraph_index: int | None = None
        paragraph_char_offset: int | None = None
        try:
            doc_pr = shape._inline.docPr
            doc_pr_id = _doc_pr_attr(doc_pr, "id")
            alt_text = (
                _doc_pr_attr(doc_pr, "descr") or _doc_pr_attr(doc_pr, "title") or ""
            )
            if alt_text:
                description = str(alt_text).strip()
            if doc_pr_id is not None:
                anchor = figure_offsets.get(str(doc_pr_id))
                if anchor is not None:
                    paragraph_index, paragraph_char_offset = anchor
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
                    "paragraph_index": paragraph_index,
                    "paragraph_char_offset": paragraph_char_offset,
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


def _attach_docx_figure_locators(
    figure_items: List[Dict[str, Any]],
    *,
    content: str,
    text_parts: List[_DocxTextPart],
) -> None:
    if not figure_items or not content.strip() or not text_parts:
        return

    offsets: dict[int, tuple[_DocxTextPart, int, int]] = {}
    cursor = 0
    for part in text_parts:
        start = cursor
        end = start + len(part.text)
        offsets[part.body_index] = (part, start, end)
        cursor = end + 2
    body_indices = sorted(offsets)

    for item in figure_items:
        metadata = dict(item.get("metadata") or {})
        paragraph_index = metadata.get("paragraph_index")
        if not isinstance(paragraph_index, int):
            continue
        exact = offsets.get(paragraph_index)
        if exact is not None:
            part, start, end = exact
            paragraph_char_offset = metadata.get("paragraph_char_offset")
            if isinstance(paragraph_char_offset, int):
                anchor = start + max(0, min(len(part.text) - 1, paragraph_char_offset))
            else:
                anchor = end - 1
        else:
            insertion = bisect_left(body_indices, paragraph_index)
            nearest_index = (
                body_indices[insertion - 1]
                if insertion > 0
                else body_indices[insertion]
            )
            part, start, end = offsets[nearest_index]
            anchor = end - 1 if nearest_index < paragraph_index else start
        if not part.text:
            continue
        metadata["text_anchor_start_char"] = anchor
        metadata["text_anchor_end_char"] = anchor
        metadata.pop("paragraph_char_offset", None)
        item["metadata"] = metadata


class DocxConverter(HybridConverter):
    """Converts DOCX files to markdown with heading style detection."""

    format_name = DOCX_CONVERTER_KEY

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        return await self._convert_office(
            file_bytes, filename, mime_type, parse_label="DOCX"
        )

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

            text_parts, figure_offsets = _extract_ordered_text_parts(doc)

            figure_items = _extract_docx_figure_items(
                doc,
                figure_offsets=figure_offsets,
            )
            parts = [part.text for part in text_parts]
            has_extracted_text = bool("\n\n".join(parts).strip())

            content = "\n\n".join(parts) if has_extracted_text else ""
            _attach_docx_figure_locators(
                figure_items,
                content=content,
                text_parts=text_parts,
            )
            quality = score_text_quality(content)

            metadata: Dict[str, Any] = {
                "parser": "local:python-docx",
                "needs_ocr": bool(figure_items and not has_extracted_text),
                "text_char_count": len("\n\n".join(parts).strip()),
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
