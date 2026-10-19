"""Readable PDF forms must not require OCR because of printed form symbols."""

from __future__ import annotations

import asyncio

import pytest

from rag_core._engine.core_prepare import prepare_document_bytes
from rag_core.config import ChunkingConfig
from rag_core.documents.local_parse import parse_file_bytes
from rag_core.documents.pdf_text_quality import looks_like_garbled_pdf_text


@pytest.mark.parametrize("symbols", ["□ □ □", "┌──────┐ │ │ └──────┘", "■ ▪ ▫ ● ○"])
def test_printed_form_symbols_are_valid_text(symbols: str) -> None:
    text = f"Vendor request: corporation {symbols}, partnership, or sole proprietor."
    assert not looks_like_garbled_pdf_text(text)


@pytest.mark.integration
def test_pdf_form_is_prepared_without_ocr_and_keeps_page_locations() -> None:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text(
            (40, 60),
            "Vendor request form. Select all that apply.\n"
            + "Corporation □ Partnership □ Service □\n" * 4,
            fontname="china-s",
            fontsize=10,
        )
        content = document.tobytes()
    finally:
        document.close()
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=content,
            filename="vendor-form.pdf",
            mime_type="application/pdf",
            path=None,
            ocr_provider=None,
            chunking_config=ChunkingConfig(),
        )
    )
    assert prepared.metadata["needs_ocr"] is False
    assert "□" in prepared.markdown
    assert prepared.chunks
    assert all(chunk.metadata["page_number"] == 1 for chunk in prepared.chunks)


@pytest.mark.integration
def test_pdf_of_only_symbols_still_requires_ocr() -> None:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text((40, 60), ("□" * 12 + "\n") * 8, fontname="china-s")
        content = document.tobytes()
    finally:
        document.close()
    _, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=content, filename="unreadable.pdf", mime_type="application/pdf"
        )
    )
    assert metadata["needs_ocr"] is True
    assert metadata["quality"]["details"].startswith("low meaningful ratio")
