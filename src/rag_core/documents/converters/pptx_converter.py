"""PPTX converter with hybrid text extraction + OCR fallback.

Extracts text from slide shapes, tables, and speaker notes.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any, Dict, List

from .base import (
    ConversionResult,
    HybridConverter,
    render_markdown_table,
    score_text_quality,
)
from .converter_keys import PPTX_CONVERTER_KEY
from .quality import QualityVerdict, is_char_count_only_quality_failure

logger = logging.getLogger(__name__)


def _is_generic_figure_description(value: str) -> bool:
    normalized = " ".join(value.lower().split())
    if normalized.startswith("picture "):
        suffix = normalized.removeprefix("picture ").strip()
        return suffix.isdigit()
    if normalized.startswith("image "):
        suffix = normalized.removeprefix("image ").strip()
        return suffix.isdigit()
    return False


def _extract_shape_text(shape: Any) -> List[str]:
    """Extract text lines from a PPTX shape (text frames and tables)."""
    lines: List[str] = []

    if shape.has_text_frame:
        for paragraph in shape.text_frame.paragraphs:
            text = paragraph.text.strip()
            if text:
                lines.append(text)

    if shape.has_table:
        rows: List[List[str]] = []
        for row in shape.table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(cells)
        if rows:
            lines.append(render_markdown_table(rows))

    return lines


def _extract_slide_figure_items(slide: Any, slide_index: int) -> List[Dict[str, Any]]:
    """Extract figure metadata from image-like slide shapes."""
    figures: List[Dict[str, Any]] = []
    picture_shape_type = None
    try:
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        picture_shape_type = MSO_SHAPE_TYPE.PICTURE
    except Exception:
        # python-pptx may be unavailable in some environments; keep extraction going.
        pass

    figure_number = 0
    for shape in slide.shapes:
        is_picture = False
        try:
            if (
                picture_shape_type is not None
                and shape.shape_type == picture_shape_type
            ):
                is_picture = True
        except Exception:
            # Some shapes expose shape_type inconsistently; fall back to image detection.
            pass
        if not is_picture and getattr(shape, "image", None) is not None:
            is_picture = True
        if not is_picture:
            continue

        figure_number += 1
        figure_id = "fig:slide:%d:%d" % (slide_index + 1, figure_number)
        label = "Slide %d Figure %d" % (slide_index + 1, figure_number)
        description = ""
        try:
            alt_text = str(getattr(shape, "name", "") or "").strip()
            if alt_text and not _is_generic_figure_description(alt_text):
                description = alt_text
        except Exception:
            # Alt text is optional; missing metadata should not block figure extraction.
            pass

        figures.append(
            {
                "figure_id": figure_id,
                "page_index": slide_index,
                "label": label,
                "description": description,
                "metadata": {
                    "source": "pptx:picture_shape",
                    "slide_number": slide_index + 1,
                },
            }
        )

    return figures


class PptxConverter(HybridConverter):
    """Converts PPTX files to markdown with slide structure and speaker notes."""

    format_name = PPTX_CONVERTER_KEY

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
            raise ValueError("PPTX parse failed (%s)" % type(root).__name__) from exc
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
        """Extract text from PPTX using python-pptx."""
        from pptx import Presentation

        def _extract() -> ConversionResult:
            try:
                prs = Presentation(io.BytesIO(file_bytes))
            except Exception as exc:
                logger.warning(
                    "Failed to open Office document: format=%s error_type=%s",
                    self.format_name,
                    type(exc).__name__,
                )
                raise ValueError("PPTX parse failed (%s)" % type(exc).__name__) from exc

            slide_sections: List[str] = []
            figure_items: List[Dict[str, Any]] = []
            extracted_text_parts: List[str] = []

            for i, slide in enumerate(prs.slides):
                parts: List[str] = ["## Slide %d" % (i + 1)]
                try:
                    title_shape = slide.shapes.title
                except Exception:
                    # Some slide masters omit a readable title placeholder; keep extraction going without it.
                    title_shape = None
                slide_title = (
                    str(getattr(title_shape, "text", "") or "").strip()
                    if title_shape is not None
                    else ""
                )
                if slide_title:
                    parts.append("### %s" % slide_title)
                    extracted_text_parts.append(slide_title)

                for shape in slide.shapes:
                    shape_lines = _extract_shape_text(shape)
                    parts.extend(shape_lines)
                    extracted_text_parts.extend(shape_lines)

                slide_figures = _extract_slide_figure_items(slide, i)
                if slide_figures:
                    for item in slide_figures:
                        description = str(item.get("description", "") or "").strip()
                        if description:
                            parts.append(description)
                            extracted_text_parts.append(description)
                    figure_items.extend(slide_figures)

                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        parts.append("\n> **Notes:** %s" % notes)
                        extracted_text_parts.append(notes)

                slide_sections.append("\n\n".join(parts))

            text_content = "\n\n".join(extracted_text_parts).strip()
            content = "\n\n---\n\n".join(slide_sections) if text_content else ""
            quality = score_text_quality(content)

            metadata: Dict[str, Any] = {
                "parser": "local:python-pptx",
                "slide_count": len(prs.slides),
                "needs_ocr": bool(figure_items and not text_content),
                "text_char_count": len(text_content),
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
