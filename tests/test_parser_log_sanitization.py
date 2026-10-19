from __future__ import annotations

import asyncio
import logging

import pytest

from rag_core.documents import local_parse
from rag_core.documents.converters.base import ConversionResult
from rag_core.documents.converters.pdf_converter_extraction import _extract_page
from tests.support import TEST_API_SECRET, assert_caplog_omits_private


class ProviderSecretError(RuntimeError):
    pass


class _FailingPdfPage:
    def get_text(self, mode: str) -> str:
        assert mode == "text"
        raise ProviderSecretError("raw page detail with api key sk-test-secret")

    def get_images(self) -> list[object]:
        return []


def test_pdf_page_extraction_warning_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(
        logging.WARNING,
        logger="rag_core.documents.converters.pdf_converter_extraction",
    ):
        extraction = _extract_page(_FailingPdfPage(), 7)

    assert extraction.page_num == 7
    assert extraction.needs_ocr is True
    assert "ProviderSecretError" in caplog.text
    assert_caplog_omits_private(caplog, "raw page detail")






def test_local_parse_failure_log_is_sanitized(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_convert(
        file_bytes: bytes, filename: str, mime_type: str
    ) -> ConversionResult:
        assert file_bytes == b"private bytes"
        assert filename == "sensitive-roadmap.md"
        assert mime_type == "text/markdown"
        raise ProviderSecretError("raw converter detail with api key sk-test-secret")

    import rag_core.documents.converters as converters_module

    monkeypatch.setattr(converters_module, "convert_file", fail_convert)

    with caplog.at_level(logging.ERROR, logger="rag_core.documents.local_parse"):
        with pytest.raises(local_parse.LocalParseError) as exc_info:
            asyncio.run(
                local_parse.parse_file_bytes(
                    file_bytes=b"private bytes",
                    filename="sensitive-roadmap.md",
                    mime_type="text/markdown",
                )
            )

    error_message = str(exc_info.value)
    assert "sensitive-roadmap.md" in error_message
    assert "ProviderSecretError" in error_message
    assert "raw converter detail with api key sk-test-secret" not in error_message
    assert TEST_API_SECRET not in error_message
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "ProviderSecretError" in caplog.text
    assert_caplog_omits_private(
        caplog,
        "raw converter detail",
        "sensitive-roadmap.md",
    )


def test_selected_converter_initialization_error_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_core.documents.converters as converters
    imports: list[str] = []
    def fail_import(module_name: str, package: str | None = None) -> object:
        imports.append(module_name)
        raise ProviderSecretError("raw private path sk-test-secret")
    converters._converter.cache_clear()
    monkeypatch.setattr(converters, "import_module", fail_import)
    try:
        with pytest.raises(RuntimeError, match="error_type=ProviderSecretError") as error:
            converters.get_converter(filename="doc.pdf", mime_type="application/pdf")
        assert imports == [".pdf_converter"]
        assert "raw private" not in str(error.value)
        assert TEST_API_SECRET not in str(error.value)
        assert error.value.__cause__ is None
    finally:
        converters._converter.cache_clear()
