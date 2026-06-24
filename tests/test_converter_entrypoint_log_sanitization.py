from __future__ import annotations

import asyncio
import logging

import pytest

import rag_core.documents.converters as converters
from rag_core.documents.converters.base import (
    BaseConverter,
    ConversionResult,
    QualityScore,
    QualityVerdict,
)
from tests.support import assert_caplog_omits_private


class _StubConverter(BaseConverter):
    format_name = "stub"

    def __init__(self, result: ConversionResult) -> None:
        self._result = result

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        assert file_bytes == b"private bytes"
        assert filename == "private-roadmap.md"
        assert mime_type == "text/markdown"
        return self._result


def test_convert_file_quality_summary_log_is_sanitized(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = ConversionResult(
        content="private result content with api key sk-test-secret",
        metadata={"parser": "local:stub"},
        quality=QualityScore(
            char_count=48,
            verdict=QualityVerdict.GOOD,
            details="raw quality detail with api key sk-test-secret",
        ),
    )
    monkeypatch.setattr(
        converters,
        "get_converter",
        lambda *, mime_type, filename: _StubConverter(result),
    )

    with caplog.at_level(logging.DEBUG, logger="rag_core.documents.converters"):
        converted = asyncio.run(
            converters.convert_file(
                b"private bytes",
                filename="private-roadmap.md",
                mime_type="text/markdown",
            )
        )

    assert converted is result
    assert "stub" in caplog.text
    assert "good" in caplog.text
    assert "48" in caplog.text
    assert_caplog_omits_private(
        caplog,
        "private-roadmap.md",
        "private bytes",
        "private result content",
        "raw quality detail",
    )


def test_convert_file_no_quality_summary_log_is_sanitized(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = ConversionResult(
        metadata={"parser": "local:stub", "error": "raw parse detail sk-test-secret"},
        needs_ocr=True,
    )
    monkeypatch.setattr(
        converters,
        "get_converter",
        lambda *, mime_type, filename: _StubConverter(result),
    )

    with caplog.at_level(logging.DEBUG, logger="rag_core.documents.converters"):
        converted = asyncio.run(
            converters.convert_file(
                b"private bytes",
                filename="private-roadmap.md",
                mime_type="text/markdown",
            )
        )

    assert converted is result
    assert "stub" in caplog.text
    assert "needs_ocr=True" in caplog.text
    assert result.metadata["error"] == "raw parse detail sk-test-secret"
    assert_caplog_omits_private(
        caplog,
        "private-roadmap.md",
        "private bytes",
        "raw parse detail",
    )
