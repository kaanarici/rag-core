"""CSV delimiter and header detection helpers."""

from __future__ import annotations

import csv
import io
from collections import Counter
from itertools import islice


def detect_delimiter(sample: str) -> str:
    candidates = [",", "\t", ";", "|"]
    best_delimiter = ","
    best_score = (0.0, 0.0)
    for delimiter in candidates:
        try:
            rows = csv.reader(io.StringIO(sample, newline=""), delimiter=delimiter)
            column_counts = [len(row) for row in islice(rows, 20) if row]
        except csv.Error:
            continue
        if not column_counts:
            continue
        width, matching_rows = Counter(column_counts).most_common(1)[0]
        if width <= 1:
            continue
        score = (
            matching_rows / len(column_counts),
            sum(column_counts) / len(column_counts),
        )
        if score > best_score:
            best_score = score
            best_delimiter = delimiter
    return best_delimiter


def detect_header_row(rows: list[list[str]]) -> bool:
    if len(rows) < 2:
        return False

    first_row = rows[0]
    data_rows = rows[1 : min(6, len(rows))]
    first_all_text = all(not _is_numeric(cell) for cell in first_row if cell.strip())
    data_has_numbers = any(
        _is_numeric(cell) for row in data_rows for cell in row if cell.strip()
    )
    if first_all_text and data_has_numbers:
        return True

    stripped_first = [cell.strip().lower() for cell in first_row if cell.strip()]
    if (
        first_all_text
        and len(stripped_first) == len(set(stripped_first))
        and len(stripped_first) > 1
    ):
        return True
    return False


def _is_numeric(value: str) -> bool:
    cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        float(cleaned)
    except ValueError:
        return False
    return True
