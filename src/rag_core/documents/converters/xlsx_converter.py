"""XLSX converter with computed values, optional formulas, and size limits."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from rag_core.config.env_access import get_env_int

from .base import BaseConverter, ConversionResult, score_text_quality
from .converter_keys import XLSX_CONVERTER_KEY
from .xlsx_rendering import build_xlsx_sheet_sections
from .xlsx_rows import extract_xlsx_sheet_rows
from .xlsx_workbooks import close_xlsx_workbooks, load_xlsx_workbooks

logger = logging.getLogger(__name__)


class XlsxConverter(BaseConverter):
    """Converts XLSX files to markdown tables.

    Uses computed values for indexing and can include formulas when configured.
    """

    format_name = XLSX_CONVERTER_KEY

    def __init__(
        self,
        *,
        max_rows: int = 0,
        max_cols: int = 0,
        include_formulas: bool = False,
    ) -> None:
        self._rows_per_chunk = max(
            1, max_rows or get_env_int("LOCAL_PARSE_XLSX_MAX_ROWS", 500)
        )
        self._max_total_rows = max(
            1, get_env_int("LOCAL_PARSE_XLSX_MAX_TOTAL_ROWS", 5000)
        )
        self._max_cols = max(
            1, max_cols or get_env_int("LOCAL_PARSE_XLSX_MAX_COLS", 50)
        )
        self._include_formulas = include_formulas

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert XLSX to markdown tables."""

        def _extract() -> ConversionResult:
            sections: List[str] = []
            sheet_count = 0
            figure_items: List[Dict[str, Any]] = []
            workbook_rows_consumed = 0
            workbook_truncated = False

            workbooks = load_xlsx_workbooks(
                file_bytes,
                format_name=self.format_name,
                logger=logger,
            )
            if workbooks.error:
                raise ValueError(workbooks.error)

            try:
                wb_data = workbooks.data
                if wb_data is None:
                    return ConversionResult(metadata={"parser": "local:openpyxl"})

                for sheet_index, sheet in enumerate(wb_data.worksheets):
                    sheet_count += 1
                    formula_sheet = None
                    if workbooks.formula is not None:
                        try:
                            formula_sheet = workbooks.formula[sheet.title]
                        except Exception:
                            formula_sheet = None

                    remaining_row_budget = max(
                        self._max_total_rows - workbook_rows_consumed,
                        0,
                    )
                    rows_data, rows_consumed, truncated = extract_xlsx_sheet_rows(
                        sheet,
                        formula_sheet=formula_sheet,
                        remaining_total_rows=remaining_row_budget,
                        max_total_rows=self._max_total_rows,
                        max_cols=self._max_cols,
                        include_formulas=self._include_formulas,
                    )
                    workbook_rows_consumed += rows_consumed
                    workbook_truncated = workbook_truncated or truncated
                    charts = workbooks.chart_map.get(sheet.title, [])
                    sheet_sections, sheet_figures = build_xlsx_sheet_sections(
                        sheet_title=sheet.title,
                        sheet_index=sheet_index,
                        rows_data=rows_data,
                        charts=charts,
                        truncated=truncated,
                        rows_per_chunk=self._rows_per_chunk,
                        max_total_rows=self._max_total_rows,
                    )
                    sections.extend(sheet_sections)
                    figure_items.extend(sheet_figures)
            finally:
                close_xlsx_workbooks(workbooks)

            content = "\n\n".join(sections)
            quality = score_text_quality(content)

            metadata: Dict[str, Any] = {
                "parser": "local:openpyxl",
                "sheet_count": sheet_count,
                "needs_ocr": False,
            }
            if workbook_truncated:
                metadata["row_truncated"] = True
                metadata["row_limit"] = self._max_total_rows
                metadata["row_limit_scope"] = "workbook"
                metadata["rows_emitted"] = workbook_rows_consumed
            if figure_items:
                metadata["figure_items"] = figure_items
                metadata["figure_count"] = len(figure_items)

            return ConversionResult(
                content=content,
                metadata=metadata,
                quality=quality,
            )

        return await asyncio.to_thread(_extract)
