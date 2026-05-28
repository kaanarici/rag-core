from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from rag_core.documents.converters.base import BaseConverter
from rag_core.documents.converters.docx_converter import DocxConverter
from rag_core.documents.converters.pptx_converter import PptxConverter
from rag_core.documents.converters.xlsx_converter import XlsxConverter
from tests.support import assert_caplog_omits_private


class ProviderSecretError(RuntimeError):
    pass


@dataclass(frozen=True)
class OfficeFailureCase:
    name: str
    build_converter: Callable[[], BaseConverter]
    logger_name: str
    filename: str
    mime_type: str
    expected_message: str


CASES = (
    OfficeFailureCase(
        name="docx",
        build_converter=DocxConverter,
        logger_name="rag_core.documents.converters.docx_converter",
        filename="private-roadmap.docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        expected_message=r"DOCX parse failed \(BadZipFile\)",
    ),
    OfficeFailureCase(
        name="pptx",
        build_converter=PptxConverter,
        logger_name="rag_core.documents.converters.pptx_converter",
        filename="private-board-deck.pptx",
        mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        expected_message=r"PPTX parse failed \(BadZipFile\)",
    ),
    OfficeFailureCase(
        name="xlsx",
        build_converter=XlsxConverter,
        logger_name="rag_core.documents.converters.xlsx_converter",
        filename="private-financial-model.xlsx",
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        expected_message=r"XLSX parse failed \(BadZipFile\)",
    ),
)


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_office_open_failure_warning_is_sanitized(
    case: OfficeFailureCase,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=case.logger_name):
        with pytest.raises(ValueError, match=case.expected_message):
            asyncio.run(
                case.build_converter().convert(
                    b"not an office zip",
                    filename=case.filename,
                    mime_type=case.mime_type,
                )
            )

    assert case.name in caplog.text
    assert "BadZipFile" in caplog.text
    assert_caplog_omits_private(caplog, case.filename)


def test_xlsx_safe_failure_reason_does_not_expose_raw_library_text() -> None:
    with pytest.raises(ValueError, match=r"XLSX parse failed \([A-Za-z0-9_]+\)"):
        asyncio.run(
            XlsxConverter().convert(
                b"not an office zip",
                filename="private-financial-model.xlsx",
                mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        )


class _WorkbookWithoutSheets:
    worksheets: list[object] = []

    def close(self) -> None:
        pass


def test_xlsx_secondary_workbook_debug_logs_are_sanitized(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bool, bool]] = []

    def load_workbook(
        _stream: object,
        *,
        data_only: bool,
        read_only: bool,
    ) -> _WorkbookWithoutSheets:
        calls.append((data_only, read_only))
        if (data_only, read_only) == (True, True):
            return _WorkbookWithoutSheets()
        if (data_only, read_only) == (False, True):
            raise ProviderSecretError("raw formula detail with api key sk-test-secret")
        if (data_only, read_only) == (True, False):
            raise ProviderSecretError("raw chart detail with api key sk-test-secret")
        raise AssertionError((data_only, read_only))

    monkeypatch.setattr("openpyxl.load_workbook", load_workbook)

    with caplog.at_level(
        logging.DEBUG,
        logger="rag_core.documents.converters.xlsx_converter",
    ):
        result = asyncio.run(
            XlsxConverter().convert(
                b"fake xlsx bytes",
                filename="private-financial-model.xlsx",
                mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        )

    assert calls == [(True, True), (False, True), (True, False)]
    assert result.content == ""
    assert result.metadata["parser"] == "local:openpyxl"
    assert result.metadata["sheet_count"] == 0
    assert "formula workbook" in caplog.text
    assert "chart metadata" in caplog.text
    assert "xlsx" in caplog.text
    assert "ProviderSecretError" in caplog.text
    assert_caplog_omits_private(
        caplog,
        "private-financial-model.xlsx",
        "raw formula detail",
        "raw chart detail",
    )
