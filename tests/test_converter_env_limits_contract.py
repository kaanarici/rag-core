from __future__ import annotations

import asyncio

import pytest

from rag_core.documents.converters.csv_converter import CsvConverter
from rag_core.documents.converters.xlsx_converter import XlsxConverter


def test_csv_converter_falls_back_on_malformed_env_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_PARSE_CSV_MAX_ROWS", "not-an-int")

    converter = CsvConverter()

    assert converter._max_rows == 1000


def test_csv_converter_clamps_non_positive_env_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_PARSE_CSV_MAX_ROWS", "0")

    converter = CsvConverter()

    assert converter._max_rows == 1


def test_csv_converter_explicit_limit_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_PARSE_CSV_MAX_ROWS", "1000")

    converter = CsvConverter(max_rows=7)

    assert converter._max_rows == 7


def test_single_row_csv_metadata_matches_synthetic_headers() -> None:
    result = asyncio.run(
        CsvConverter().convert(
            b"100,200\n",
            filename="matrix.csv",
            mime_type="text/csv",
        )
    )

    assert "| Col 1 | Col 2 |" in result.content
    assert "| 100 | 200 |" in result.content
    assert result.metadata["has_header"] is False


def test_single_row_tsv_metadata_matches_synthetic_headers() -> None:
    result = asyncio.run(
        CsvConverter().convert(
            b"north\t42\n",
            filename="region.tsv",
            mime_type="text/tab-separated-values",
        )
    )

    assert "| Col 1 | Col 2 |" in result.content
    assert "| north | 42 |" in result.content
    assert result.metadata["has_header"] is False


def test_xlsx_converter_falls_back_on_malformed_env_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_PARSE_XLSX_MAX_ROWS", "not-an-int")
    monkeypatch.setenv("LOCAL_PARSE_XLSX_MAX_TOTAL_ROWS", "not-an-int")
    monkeypatch.setenv("LOCAL_PARSE_XLSX_MAX_COLS", "not-an-int")

    converter = XlsxConverter()

    assert converter._rows_per_chunk == 500
    assert converter._max_total_rows == 5000
    assert converter._max_cols == 50


def test_xlsx_converter_clamps_non_positive_env_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_PARSE_XLSX_MAX_ROWS", "0")
    monkeypatch.setenv("LOCAL_PARSE_XLSX_MAX_TOTAL_ROWS", "-2")
    monkeypatch.setenv("LOCAL_PARSE_XLSX_MAX_COLS", "0")

    converter = XlsxConverter()

    assert converter._rows_per_chunk == 1
    assert converter._max_total_rows == 1
    assert converter._max_cols == 1


def test_xlsx_converter_explicit_limits_override_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_PARSE_XLSX_MAX_ROWS", "500")
    monkeypatch.setenv("LOCAL_PARSE_XLSX_MAX_COLS", "50")

    converter = XlsxConverter(max_rows=7, max_cols=3)

    assert converter._rows_per_chunk == 7
    assert converter._max_cols == 3
