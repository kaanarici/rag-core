from __future__ import annotations

import asyncio
import logging
import subprocess
from types import SimpleNamespace

import pytest

import rag_core.documents.converters.pdf_converter as pdf_converter_module
import rag_core.documents.pdf_inspector as pdf_inspector_module
import rag_core.documents.pdf_inspector_runtime as pdf_inspector_runtime
from rag_core.documents.converters.pdf_converter import PdfConverter


def test_pdf_inspector_schema_error_log_uses_error_type(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: {"pdf_type": 7},
    )
    caplog.set_level(logging.WARNING, logger="rag_core.documents.pdf_inspector")

    result = pdf_inspector_module.detect_pdf_with_inspector(b"%PDF-1.7")

    assert result is None
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "detection payload was invalid" in message and "ValueError" in message
        for message in messages
    )
    assert not any("pdf_type must be a string" in message for message in messages)


def test_pdf_inspector_invalid_json_log_uses_error_type(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(pdf_inspector_runtime, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(pdf_inspector_runtime, "_resolve_binary_path", lambda _: "detect-pdf")
    monkeypatch.setattr(
        pdf_inspector_runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"secret":',
            stderr="",
        ),
    )
    caplog.set_level(logging.WARNING, logger="rag_core.documents.pdf_inspector")

    result = pdf_inspector_runtime.run_pdf_inspector(
        ["detect-pdf", "--analyze", "--json"],
        b"%PDF-1.7",
    )

    assert result is None
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "detect-pdf returned invalid JSON" in message and "JSONDecodeError" in message
        for message in messages
    )
    assert not any("secret" in message for message in messages)


def test_pdf_inspector_extraction_schema_error_log_uses_error_type(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        pdf_inspector_runtime,
        "run_pdf_inspector",
        lambda command, file_bytes: {
            "pdf_type": "text",
            "page_count": 1,
            "pages_needing_ocr": [],
        },
    )
    caplog.set_level(logging.WARNING, logger="rag_core.documents.pdf_inspector")

    result = pdf_inspector_module.extract_pdf_with_inspector(b"%PDF-1.7")

    assert result is None
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "extraction payload was invalid" in message and "ValueError" in message
        for message in messages
    )
    assert not any("markdown must be a string" in message for message in messages)


def test_pdf_inspector_startup_error_log_uses_error_type(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(pdf_inspector_runtime, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(pdf_inspector_runtime, "_resolve_binary_path", lambda _: "detect-pdf")

    def fail_run(*args: object, **kwargs: object) -> object:
        raise OSError("secret binary path /tmp/private/detect-pdf")

    monkeypatch.setattr(pdf_inspector_runtime.subprocess, "run", fail_run)
    caplog.set_level(logging.WARNING, logger="rag_core.documents.pdf_inspector")

    result = pdf_inspector_runtime.run_pdf_inspector(
        ["detect-pdf", "--analyze", "--json"],
        b"%PDF-1.7",
    )

    assert result is None
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "detect-pdf failed to start" in message and "OSError" in message
        for message in messages
    )
    assert not any("secret binary path" in message for message in messages)


def test_pdf_converter_availability_wrapper_log_uses_error_type(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_enabled() -> bool:
        raise RuntimeError("secret inspector token")

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", fail_enabled)
    caplog.set_level(logging.WARNING, logger="rag_core.documents.converters.pdf_converter")

    result = asyncio.run(
        PdfConverter()._try_extract_with_inspector(b"%PDF-1.7", "report.pdf")
    )

    assert result is None
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "availability check failed" in message and "error_type=RuntimeError" in message
        for message in messages
    )
    assert not any("report.pdf" in message for message in messages)
    assert not any("secret inspector token" in message for message in messages)


def test_pdf_converter_detection_wrapper_log_uses_error_type(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_detect(file_bytes: bytes) -> None:
        raise RuntimeError("secret detection payload")

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(pdf_converter_module, "detect_pdf_with_inspector", fail_detect)
    caplog.set_level(logging.WARNING, logger="rag_core.documents.converters.pdf_converter")

    result = asyncio.run(
        PdfConverter()._try_extract_with_inspector(b"%PDF-1.7", "report.pdf")
    )

    assert result is None
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "detection failed" in message and "error_type=RuntimeError" in message
        for message in messages
    )
    assert not any("report.pdf" in message for message in messages)
    assert not any("secret detection payload" in message for message in messages)


def test_pdf_converter_extraction_wrapper_log_uses_error_type(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    detection = pdf_inspector_module.PdfInspectorDetectionResult(
        pdf_type="text",
        page_count=1,
        pages_needing_ocr=[],
        confidence=0.9,
        has_encoding_issues=False,
        processing_time_ms=1,
    )

    def fail_extract(file_bytes: bytes) -> None:
        raise subprocess.SubprocessError("secret extraction payload")

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "detect_pdf_with_inspector",
        lambda file_bytes: detection,
    )
    monkeypatch.setattr(pdf_converter_module, "extract_pdf_with_inspector", fail_extract)
    caplog.set_level(logging.WARNING, logger="rag_core.documents.converters.pdf_converter")

    result = asyncio.run(
        PdfConverter()._try_extract_with_inspector(b"%PDF-1.7", "report.pdf")
    )

    assert result is None
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "extraction failed" in message and "error_type=SubprocessError" in message
        for message in messages
    )
    assert not any("report.pdf" in message for message in messages)
    assert not any("secret extraction payload" in message for message in messages)
