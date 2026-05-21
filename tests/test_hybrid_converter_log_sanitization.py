from __future__ import annotations

import asyncio
import logging

import pytest

from rag_core.documents.converters.base import (
    ConversionResult,
    HybridConverter,
    QualityScore,
    QualityVerdict,
)


class ProviderSecretError(RuntimeError):
    pass


class _HybridStub(HybridConverter):
    format_name = "hybrid-test"

    def __init__(
        self,
        result: ConversionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error

    async def _try_extract(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        assert file_bytes == b"private bytes"
        assert filename == "private-roadmap.pdf"
        assert mime_type == "application/pdf"
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def test_hybrid_converter_success_debug_log_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    converter = _HybridStub(
        result=ConversionResult(
            content="usable extracted text",
            metadata={"parser": "local:hybrid-test"},
            quality=QualityScore(verdict=QualityVerdict.GOOD),
        )
    )

    with caplog.at_level(logging.DEBUG, logger="rag_core.documents.converters.base"):
        result = asyncio.run(
            converter.convert(
                b"private bytes",
                filename="private-roadmap.pdf",
                mime_type="application/pdf",
            )
        )

    assert result.content == "usable extracted text"
    assert result.needs_ocr is False
    assert "hybrid-test" in caplog.text
    assert "extracted via text layer" in caplog.text
    assert "private-roadmap.pdf" not in caplog.text
    assert "private bytes" not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_hybrid_converter_quality_fallback_debug_log_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    converter = _HybridStub(
        result=ConversionResult(
            content="thin extraction",
            metadata={"parser": "local:hybrid-test"},
            quality=QualityScore(
                verdict=QualityVerdict.POOR,
                details="raw quality detail with api key sk-test-secret",
            ),
        )
    )

    with caplog.at_level(logging.DEBUG, logger="rag_core.documents.converters.base"):
        result = asyncio.run(
            converter.convert(
                b"private bytes",
                filename="private-roadmap.pdf",
                mime_type="application/pdf",
            )
        )

    assert result.needs_ocr is True
    assert result.quality is not None
    assert result.quality.details == "raw quality detail with api key sk-test-secret"
    assert "hybrid-test" in caplog.text
    assert "poor" in caplog.text
    assert "raw quality detail" not in caplog.text
    assert "sk-test-secret" not in caplog.text
    assert "private-roadmap.pdf" not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_hybrid_converter_failure_warning_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    converter = _HybridStub(
        error=ProviderSecretError("raw extraction detail with api key sk-test-secret")
    )

    with caplog.at_level(logging.WARNING, logger="rag_core.documents.converters.base"):
        result = asyncio.run(
            converter.convert(
                b"private bytes",
                filename="private-roadmap.pdf",
                mime_type="application/pdf",
            )
        )

    assert result.content == ""
    assert result.needs_ocr is True
    assert result.metadata["parser"] == "local:hybrid-test"
    assert result.metadata["error"] == "ProviderSecretError"
    assert "hybrid-test" in caplog.text
    assert "ProviderSecretError" in caplog.text
    assert "raw extraction detail" not in caplog.text
    assert "sk-test-secret" not in caplog.text
    assert "private-roadmap.pdf" not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
