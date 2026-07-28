"""Private converter-to-chunker metadata transport for XLSX sections."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass

_SHEET_MARKER_RE = re.compile(
    r"^<!-- rag-core-xlsx-sheet:([A-Za-z0-9_-]+)"
    r"(?::([1-9]\d*)-([1-9]\d*))? -->$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class XlsxSheetMarker:
    sheet_name: str
    start_row: int | None
    end_row: int | None


def render_xlsx_sheet_marker(
    sheet_name: str,
    *,
    start_row: int | None = None,
    end_row: int | None = None,
) -> str:
    encoded_name = (
        base64.urlsafe_b64encode(sheet_name.encode("utf-8")).decode("ascii").rstrip("=")
    )
    row_range = (
        f":{start_row}-{end_row}"
        if start_row is not None and end_row is not None
        else ""
    )
    return f"<!-- rag-core-xlsx-sheet:{encoded_name}{row_range} -->"


def extract_xlsx_sheet_marker(
    section: str,
) -> tuple[str, XlsxSheetMarker | None]:
    match = _SHEET_MARKER_RE.search(section)
    if match is None:
        return section, None
    encoded_name = match.group(1)
    padding = "=" * (-len(encoded_name) % 4)
    try:
        sheet_name = base64.urlsafe_b64decode(encoded_name + padding).decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return section, None
    cleaned = f"{section[: match.start()]}{section[match.end() :]}"
    return (
        cleaned,
        XlsxSheetMarker(
            sheet_name=sheet_name,
            start_row=_optional_int(match.group(2)),
            end_row=_optional_int(match.group(3)),
        ),
    )


def _optional_int(value: str | None) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "XlsxSheetMarker",
    "extract_xlsx_sheet_marker",
    "render_xlsx_sheet_marker",
]
