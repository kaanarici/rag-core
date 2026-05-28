"""CSV converter with delimiter detection, header handling, and row limits."""

from __future__ import annotations

import asyncio
import csv
import io
import logging

from rag_core.config.env_access import get_env_int

from .base import (
    BaseConverter,
    ConversionResult,
    render_markdown_table,
    safe_decode,
    score_text_quality,
)
from .converter_keys import CSV_CONVERTER_KEY
from .csv_detection import detect_delimiter, detect_header_row

logger = logging.getLogger(__name__)


class CsvConverter(BaseConverter):
    """Converts CSV files to markdown tables with smart detection.

    - Auto-detects delimiter (tab, semicolon, pipe, comma)
    - Detects if first row is actually a header
    - Row limit for large CSVs
    """

    format_name = CSV_CONVERTER_KEY

    def __init__(self, *, max_rows: int = 0) -> None:
        self._max_rows = max(
            1, max_rows or get_env_int("LOCAL_PARSE_CSV_MAX_ROWS", 1000)
        )

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert CSV to markdown table."""

        def _convert() -> ConversionResult:
            try:
                text = safe_decode(file_bytes)
            except ValueError as exc:
                return ConversionResult(
                    metadata={"parser": "local:csv", "error": str(exc)},
                )

            if not text.strip():
                return ConversionResult(
                    metadata={"parser": "local:csv"},
                    quality=score_text_quality(""),
                )

            delimiter = detect_delimiter(text)

            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            rows: list[list[str]] = []
            truncated = False

            for row in reader:
                if len(rows) >= self._max_rows:
                    truncated = True
                    break
                rows.append(row)

            if not rows:
                return ConversionResult(
                    metadata={"parser": "local:csv"},
                    quality=score_text_quality(""),
                )

            has_header = detect_header_row(rows)

            if has_header and len(rows) > 1:
                content = render_markdown_table(rows)
            else:
                width = max(len(r) for r in rows)
                header = ["Col %d" % (i + 1) for i in range(width)]
                all_rows = [header] + rows
                content = render_markdown_table(all_rows)

            if truncated:
                content += "\n\n*[truncated after %d rows]*" % self._max_rows

            quality = score_text_quality(content)

            metadata: dict[str, str | int | bool] = {
                "parser": "local:csv",
                "delimiter": repr(delimiter),
                "has_header": has_header,
                "row_count": len(rows),
                "truncated": truncated,
                "needs_ocr": False,
            }

            return ConversionResult(
                content=content,
                metadata=metadata,
                quality=quality,
            )

        return await asyncio.to_thread(_convert)
