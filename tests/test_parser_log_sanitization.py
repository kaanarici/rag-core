from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from rag_core.documents import local_parse
from rag_core.documents.converters import registry_loader
from rag_core.documents.converters.base import ConversionResult
from rag_core.documents.converters.pdf_converter_extraction import _extract_page
from rag_core.documents.converters.registry_specs import ConverterSpec
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


def test_optional_converter_skip_warning_is_sanitized(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_loader, "_converters", None)
    monkeypatch.setattr(
        registry_loader,
        "CONVERTER_SPECS",
        (
            ConverterSpec("text", ".text_converter", "TextConverter", required=True),
            ConverterSpec("optional", ".missing", "MissingConverter"),
        ),
    )
    monkeypatch.setattr(registry_loader, "REQUIRED_CONVERTER_KEYS", ("text",))

    class TextConverter:
        async def convert(
            self, file_bytes: bytes, filename: str, mime_type: str
        ) -> ConversionResult:
            return ConversionResult(content=file_bytes.decode(), metadata={})

    def build_converter(spec: ConverterSpec) -> Any:
        if spec.key == "text":
            return TextConverter()
        try:
            raise ProviderSecretError(
                "raw optional converter detail with api key sk-test-secret"
            )
        except ProviderSecretError as exc:
            raise RuntimeError("wrapped private optional converter detail") from exc

    monkeypatch.setattr(registry_loader, "_build_converter", build_converter)

    with caplog.at_level(
        logging.WARNING, logger="rag_core.documents.converters.registry_loader"
    ):
        converters = registry_loader.get_registered_converters()

    assert set(converters) == {"text"}
    assert "optional" in caplog.text
    assert "ProviderSecretError" in caplog.text
    assert "raw optional converter detail" not in caplog.text
    assert_caplog_omits_private(
        caplog,
        "raw optional converter detail",
        "wrapped private optional converter detail",
    )


def test_required_converter_initialization_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(module_name: str, package: str | None = None) -> object:
        assert module_name == ".private"
        assert package == "rag_core.documents.converters"
        raise ProviderSecretError("raw required converter detail with api key sk-test-secret")

    monkeypatch.setattr(registry_loader.importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError) as exc_info:
        registry_loader._build_converter(
            ConverterSpec("required", ".private", "MissingConverter", required=True)
        )

    message = str(exc_info.value)
    assert message == "Failed to initialize required converter (error_type=ProviderSecretError)"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "raw required converter detail" not in message
    assert TEST_API_SECRET not in message
    assert "Traceback" not in message


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
