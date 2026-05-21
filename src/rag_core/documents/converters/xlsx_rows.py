"""Row extraction helpers for XLSX conversion."""

from __future__ import annotations

from itertools import zip_longest
from typing import Any, List, NamedTuple


class XlsxRow(NamedTuple):
    row_number: int
    cells: List[str]


def extract_xlsx_sheet_rows(
    sheet: Any,
    *,
    formula_sheet: Any | None,
    remaining_total_rows: int | None,
    max_total_rows: int,
    max_cols: int,
    include_formulas: bool,
) -> tuple[List[XlsxRow], int, bool]:
    all_rows: List[XlsxRow] = []
    row_count = 0
    truncated = False
    max_rows_for_sheet = max_total_rows if remaining_total_rows is None else max(
        remaining_total_rows, 0
    )

    formula_iter = (
        formula_sheet.iter_rows(values_only=True) if formula_sheet is not None else []
    )
    for row, formula_row in zip_longest(
        sheet.iter_rows(values_only=True),
        formula_iter,
        fillvalue=(),
    ):
        row_count += 1
        if row_count > max_rows_for_sheet:
            truncated = True
            break

        cells = _render_xlsx_row(
            row=row,
            formula_row=formula_row,
            max_cols=max_cols,
            include_formulas=include_formulas,
        )
        if any(cell.strip() for cell in cells):
            all_rows.append(XlsxRow(row_number=row_count, cells=cells))

    return all_rows, min(row_count, max_rows_for_sheet), truncated


def _render_xlsx_row(
    *,
    row: Any,
    formula_row: Any,
    max_cols: int,
    include_formulas: bool,
) -> List[str]:
    cells: List[str] = []
    values = tuple(row) if isinstance(row, tuple) else tuple(row or ())
    formula_values = (
        tuple(formula_row) if isinstance(formula_row, tuple) else tuple(formula_row or ())
    )
    for column_index in range(max(len(values), len(formula_values))):
        if column_index >= max_cols:
            cells.append("[...]")
            break
        cells.append(
            _render_xlsx_cell(
                value=(values[column_index] if column_index < len(values) else None),
                formula_value=(
                    formula_values[column_index]
                    if column_index < len(formula_values)
                    else None
                ),
                include_formulas=include_formulas,
            )
        )
    return cells


def _render_xlsx_cell(
    *,
    value: Any,
    formula_value: Any,
    include_formulas: bool,
) -> str:
    formula = (
        formula_value
        if isinstance(formula_value, str) and formula_value.startswith("=")
        else None
    )
    if value is None and formula:
        return formula
    if include_formulas and formula:
        if value is None:
            return formula
        return f"{value} [formula: {formula}]"
    return "" if value is None else str(value)
