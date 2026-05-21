from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

import rag_core.documents.converters.pdf_converter as pdf_converter_module
from rag_core.documents.converters.pdf_converter import PdfConverter
from rag_core.documents.converters.pdf_converter_extraction import PageExtraction, PdfExtraction

LOGGER_NAME = "rag_core.documents.converters.pdf_converter"
PRIVATE_FILENAME = "private-roadmap.pdf"
RAW_ERROR = "raw private PDF detail with api key sk-test-secret"
RAW_DOCUMENT_TEXT = "private document body with api key sk-test-secret"
RAW_PDF_BYTES = b"%PDF private document bytes sk-test-secret"


class SecretPdfError(RuntimeError):
    pass


def _inspector_result(
    *,
    pdf_type: str = "text",
    page_count: int = 2,
    markdown: str = "",
    pages_needing_ocr: list[int] | None = None,
) -> Any:
    return SimpleNamespace(
        pdf_type=pdf_type,
        page_count=page_count,
        markdown=markdown,
        pages_needing_ocr=pages_needing_ocr or [],
        confidence=0.9,
        has_encoding_issues=False,
        processing_time_ms=1,
    )


def _assert_logs_are_sanitized(
    caplog: pytest.LogCaptureFixture,
    *required_fragments: str,
) -> None:
    for fragment in required_fragments:
        assert fragment in caplog.text
    assert PRIVATE_FILENAME not in caplog.text
    assert RAW_ERROR not in caplog.text
    assert RAW_DOCUMENT_TEXT not in caplog.text
    assert "sk-test-secret" not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_inspector_availability_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_enabled() -> bool:
        raise SecretPdfError(RAW_ERROR)

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", fail_enabled)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = asyncio.run(
            PdfConverter()._try_extract_with_inspector(RAW_PDF_BYTES, PRIVATE_FILENAME)
        )

    assert result is None
    _assert_logs_are_sanitized(
        caplog,
        "PDF Inspector availability check failed",
        "inspector_path=local:pdf_inspector",
        "SecretPdfError",
    )


def test_inspector_detection_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_detect(file_bytes: bytes) -> None:
        assert file_bytes == RAW_PDF_BYTES
        raise SecretPdfError(RAW_ERROR)

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(pdf_converter_module, "detect_pdf_with_inspector", fail_detect)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = asyncio.run(
            PdfConverter()._try_extract_with_inspector(RAW_PDF_BYTES, PRIVATE_FILENAME)
        )

    assert result is None
    _assert_logs_are_sanitized(
        caplog,
        "PDF Inspector detection failed",
        "inspector_path=local:pdf_inspector",
        "SecretPdfError",
    )


def test_inspector_extraction_failure_log_keeps_safe_context(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_extract(file_bytes: bytes) -> None:
        assert file_bytes == RAW_PDF_BYTES
        raise SecretPdfError(RAW_ERROR)

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "detect_pdf_with_inspector",
        lambda file_bytes: _inspector_result(pdf_type="text", page_count=7),
    )
    monkeypatch.setattr(pdf_converter_module, "extract_pdf_with_inspector", fail_extract)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = asyncio.run(
            PdfConverter()._try_extract_with_inspector(RAW_PDF_BYTES, PRIVATE_FILENAME)
        )

    assert result is None
    _assert_logs_are_sanitized(
        caplog,
        "PDF Inspector extraction failed",
        "inspector_path=local:pdf_inspector",
        "route=text",
        "page_count=7",
        "SecretPdfError",
    )


def test_inspector_classification_fallback_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "detect_pdf_with_inspector",
        lambda file_bytes: _inspector_result(pdf_type="form", page_count=4),
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = asyncio.run(
            PdfConverter()._try_extract_with_inspector(RAW_PDF_BYTES, PRIVATE_FILENAME)
        )

    assert result is None
    _assert_logs_are_sanitized(
        caplog,
        "PDF Inspector classified PDF",
        "route=form",
        "inspector_path=local:pdf_inspector",
        "page_count=4",
        "fallback_path=local:pymupdf",
    )


@pytest.mark.parametrize("route", ["text", "mixed"])
def test_inspector_no_markdown_fallback_log_is_sanitized(
    route: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "detect_pdf_with_inspector",
        lambda file_bytes: _inspector_result(pdf_type=route, page_count=3),
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "extract_pdf_with_inspector",
        lambda file_bytes: _inspector_result(pdf_type=route, page_count=3, markdown=""),
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = asyncio.run(
            PdfConverter()._try_extract_with_inspector(RAW_PDF_BYTES, PRIVATE_FILENAME)
        )

    assert result is None
    _assert_logs_are_sanitized(
        caplog,
        "PDF Inspector returned no markdown",
        f"route={route}",
        "inspector_path=local:pdf_inspector",
        "page_count=3",
        "fallback_path=local:pymupdf",
    )


def test_pymupdf_encrypted_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fake_extract_pdf(file_bytes: bytes) -> PdfExtraction:
        assert file_bytes == RAW_PDF_BYTES
        return PdfExtraction(page_count=5, is_encrypted=True)

    monkeypatch.setattr(pdf_converter_module, "extract_pdf", fake_extract_pdf)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = asyncio.run(
            PdfConverter()._try_extract_with_pymupdf(
                RAW_PDF_BYTES,
                PRIVATE_FILENAME,
                "application/pdf",
            )
        )

    assert result.needs_ocr is True
    assert result.metadata["parser"] == "local:pymupdf"
    assert result.metadata["page_count"] == 5
    assert result.metadata["is_encrypted"] is True
    assert result.metadata["needs_ocr"] is True
    assert result.metadata["ocr_processed_entire_document"] is True
    assert result.quality is not None
    assert result.quality.details == "encrypted PDF"
    _assert_logs_are_sanitized(
        caplog,
        "Encrypted PDF requires OCR",
        "path=local:pymupdf",
        "page_count=5",
    )


def test_pymupdf_partial_ocr_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fake_extract_pdf(file_bytes: bytes) -> PdfExtraction:
        assert file_bytes == RAW_PDF_BYTES
        return PdfExtraction(
            page_count=3,
            pages=[
                PageExtraction(
                    page_num=0,
                    text=RAW_DOCUMENT_TEXT,
                    needs_ocr=False,
                    char_count=len(RAW_DOCUMENT_TEXT),
                ),
                PageExtraction(page_num=1, needs_ocr=True),
                PageExtraction(page_num=2, needs_ocr=True),
            ],
        )

    monkeypatch.setattr(pdf_converter_module, "extract_pdf", fake_extract_pdf)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = asyncio.run(
            PdfConverter()._try_extract_with_pymupdf(
                RAW_PDF_BYTES,
                PRIVATE_FILENAME,
                "application/pdf",
            )
        )

    assert RAW_DOCUMENT_TEXT in result.content
    assert result.metadata["parser"] == "local:pymupdf"
    assert result.metadata["page_count"] == 3
    assert result.metadata["ocr_page_count"] == 2
    assert result.ocr_page_indices == [1, 2]
    assert result.needs_ocr is True
    _assert_logs_are_sanitized(
        caplog,
        "PDF pages need OCR",
        "path=local:pymupdf",
        "ocr_page_count=2",
        "page_count=3",
        "partial_ocr=true",
    )
