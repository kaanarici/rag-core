"""Markdown rendering helpers for XLSX conversion."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .base import render_markdown_table
from .xlsx_rows import XlsxRow


def build_xlsx_sheet_sections(
    *,
    sheet_title: str,
    sheet_index: int,
    rows_data: List[XlsxRow],
    charts: List[Any],
    truncated: bool,
    rows_per_chunk: int,
    max_total_rows: int,
) -> tuple[List[str], List[Dict[str, Any]]]:
    sections: List[str] = []
    figure_items: List[Dict[str, Any]] = []

    if not rows_data and not charts:
        if truncated:
            sections.append(
                "\n\n".join(
                    [
                        "## Sheet: %s" % sheet_title,
                        "*[truncated after workbook limit of %d rows]*"
                        % max_total_rows,
                    ]
                )
            )
        return sections, figure_items

    row_groups = _build_row_groups(rows_data, rows_per_chunk=rows_per_chunk)
    for group_index, (start_row, end_row, group_rows) in enumerate(row_groups):
        if len(row_groups) > 1:
            heading = "## Sheet: %s (Rows %d-%d)" % (
                sheet_title,
                start_row,
                end_row,
            )
        else:
            heading = "## Sheet: %s" % sheet_title

        section_parts: List[str] = [heading]
        section_parts.append(render_markdown_table(_table_rows(rows_data, group_rows)))

        if truncated and group_index == len(row_groups) - 1:
            section_parts.append(
                "*[truncated after workbook limit of %d rows]*" % max_total_rows
            )

        sections.append("\n\n".join(section_parts))

    if charts:
        chart_sections, chart_figures = _build_chart_sections(
            sheet_title=sheet_title,
            sheet_index=sheet_index,
            charts=charts,
        )
        sections.extend(chart_sections)
        figure_items.extend(chart_figures)

    return sections, figure_items


def _build_row_groups(
    rows_data: List[XlsxRow],
    *,
    rows_per_chunk: int,
) -> List[Tuple[int, int, List[XlsxRow]]]:
    row_groups: List[Tuple[int, int, List[XlsxRow]]] = []
    chunk_size = max(1, rows_per_chunk)
    for start_idx in range(0, len(rows_data), chunk_size):
        chunk_rows = rows_data[start_idx : start_idx + chunk_size]
        if not chunk_rows:
            continue
        start_row = chunk_rows[0].row_number
        end_row = chunk_rows[-1].row_number
        row_groups.append((start_row, end_row, chunk_rows))
    return row_groups


def _table_rows(rows_data: List[XlsxRow], group_rows: List[XlsxRow]) -> List[List[str]]:
    header = rows_data[0]
    cells = [row.cells for row in group_rows]
    if group_rows and group_rows[0].row_number != header.row_number:
        return [header.cells, *cells]
    return cells


def _build_chart_sections(
    *,
    sheet_title: str,
    sheet_index: int,
    charts: List[Any],
) -> tuple[List[str], List[Dict[str, Any]]]:
    chart_sections: List[str] = []
    figure_items: List[Dict[str, Any]] = []
    for chart_index, chart in enumerate(charts):
        chart_title = _chart_title(chart) or (
            "Chart %d on sheet %s" % (chart_index + 1, sheet_title)
        )
        chart_label = "Sheet %s Chart %d" % (sheet_title, chart_index + 1)
        chart_sections.append(
            "\n\n".join(
                [
                    "## Sheet: %s" % sheet_title,
                    "### %s" % chart_label,
                    "- %s" % chart_title,
                ]
            )
        )
        figure_items.append(
            {
                "figure_id": "fig:sheet:%d:chart:%d"
                % (sheet_index + 1, chart_index + 1),
                "page_index": sheet_index,
                "label": chart_label,
                "description": chart_title,
                "metadata": {
                    "source": "xlsx:chart",
                    "sheet": sheet_title,
                    "sheet_name": sheet_title,
                },
            }
        )
    return chart_sections, figure_items


def _chart_title(chart: Any) -> str:
    title = getattr(chart, "title", None)
    if title is None:
        return ""
    text = _chart_title_text(title).strip()
    if text == "None":
        return ""
    return text


def _chart_title_text(title: Any) -> str:
    if isinstance(title, str):
        return title
    rich_text = getattr(getattr(title, "tx", None), "rich", None)
    paragraphs = getattr(rich_text, "p", None)
    if not isinstance(paragraphs, list):
        return str(title)
    runs: List[str] = []
    for paragraph in paragraphs:
        for run in getattr(paragraph, "r", []) or []:
            text = getattr(run, "t", None)
            if isinstance(text, str) and text.strip():
                runs.append(text.strip())
    return " ".join(runs) or str(title)
