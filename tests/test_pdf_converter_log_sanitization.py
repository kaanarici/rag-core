import asyncio
import logging
import pytest
import rag_core.documents.converters.pdf_converter as pdf_module
from rag_core.documents.converters.pdf_converter import PdfConverter
from tests.support import assert_caplog_omits_private


def test_native_pdf_failure_is_sanitized(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    async def fail(data: bytes) -> object:
        raise RuntimeError("private document text sk-test-secret")
    monkeypatch.setattr(pdf_module, "extract_pdf", fail)
    with caplog.at_level(logging.WARNING, logger=pdf_module.__name__):
        with pytest.raises(ValueError, match=r"PDF parse failed \(RuntimeError\)"):
            asyncio.run(PdfConverter().convert(b"private bytes", "private-roadmap.pdf", "application/pdf"))
    assert "RuntimeError" in caplog.text
    assert_caplog_omits_private(caplog, "private document text", "private-roadmap.pdf")
