"""CSV delimiter and header detection helpers."""

from __future__ import annotations

import csv


def detect_delimiter(sample: str) -> str:
    candidates = [",", "\t", ";", "|"]
    lines = sample.strip().split("\n")[:20]
    if len(lines) < 2:
        try:
            dialect = csv.Sniffer().sniff(sample[:4096])
        except csv.Error:
            return ","
        return str(dialect.delimiter)

    best_delimiter = ","
    best_score = -1.0
    for delimiter in candidates:
        column_counts = [len(line.split(delimiter)) for line in lines]
        average_columns = sum(column_counts) / len(column_counts)
        if average_columns <= 1:
            continue
        variance = sum(
            (count - average_columns) ** 2 for count in column_counts
        ) / len(column_counts)
        score = average_columns / (1.0 + variance)
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
        _is_numeric(cell)
        for row in data_rows
        for cell in row
        if cell.strip()
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
