from __future__ import annotations

import asyncio
import logging

import pytest

import rag_core.facade.prepare as core_prepare_facade
import rag_core.documents.converters.pdf_converter as pdf_converter_module
import rag_core.documents.pdf_inspector as pdf_inspector_module
import rag_core.documents.pdf_inspector_runtime as pdf_inspector_runtime
from rag_core import RAGCore
from rag_core._engine.core_builders import read_ocr_metadata
from rag_core.core_models import (
    OcrRoutingSignal,
    ParsedDocument,
    PreparedChunk,
    PreparedDocument,
)
from rag_core._engine.core_prepare import apply_ocr
from rag_core.documents.converters.pdf_converter import PdfConverter
from rag_core.documents.ocr import OcrRequest, OcrResult
from rag_core.documents.pdf_inspector import (
    PdfInspectorDetectionResult,
    PdfInspectorExtractionResult,
    pdf_inspector_enabled,
)
from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_test_config,
)


def test_pdf_inspector_enabled_defaults_to_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PDF_INSPECTOR_MODE", raising=False)
    assert pdf_inspector_enabled() is True


@pytest.mark.parametrize(
    "configured_path, expected_level",
    [
        (None, logging.INFO),
        ("/missing/pdf-inspector", logging.WARNING),
    ],
    ids=["default-info", "configured-warning"],
)
def test_missing_pdf_inspector_binary_logs_at_expected_level(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    configured_path: str | None,
    expected_level: int,
) -> None:
    if configured_path is None:
        monkeypatch.delenv("PDF_INSPECTOR_BINARY_PATH", raising=False)
    else:
        monkeypatch.setenv("PDF_INSPECTOR_BINARY_PATH", configured_path)
    monkeypatch.setattr(pdf_inspector_runtime, "_resolve_binary_path", lambda _: None)
    pdf_inspector_runtime._WARNED_BINARY_KEYS.clear()
    caplog.set_level(logging.INFO, logger="rag_core.documents.pdf_inspector")

    result = pdf_inspector_module.detect_pdf_with_inspector(b"%PDF-1.7")

    assert result is None
    assert any(
        record.levelno == expected_level and "detect-pdf" in record.getMessage()
        for record in caplog.records
    )
    if expected_level == logging.INFO:
        assert not any(record.levelno >= logging.WARNING for record in caplog.records)


def _stub_inspector(
    monkeypatch: pytest.MonkeyPatch,
    detection: PdfInspectorDetectionResult,
    extraction: PdfInspectorExtractionResult,
) -> None:
    async def fail_pymupdf(self, file_bytes: bytes, filename: str, mime_type: str):
        raise AssertionError(
            "PyMuPDF fallback should not run when inspector returns canonical text"
        )

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module, "detect_pdf_with_inspector", lambda file_bytes: detection
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "extract_pdf_with_inspector",
        lambda file_bytes: extraction,
    )
    monkeypatch.setattr(PdfConverter, "_try_extract_with_pymupdf", fail_pymupdf)


def test_pdf_converter_prefers_inspector_for_text_pdfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="text",
        page_count=2,
        pages_needing_ocr=[],
        confidence=0.99,
        has_encoding_issues=False,
        processing_time_ms=8,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="text",
        page_count=2,
        pages_needing_ocr=[],
        has_encoding_issues=False,
        processing_time_ms=12,
        markdown=("canonical inspector markdown " * 8).strip(),
    )
    _stub_inspector(monkeypatch, detection, extraction)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "report.pdf", "application/pdf")
    )

    assert result.content == extraction.markdown
    assert result.metadata["parser"] == "local:pdf_inspector"
    assert result.metadata["inspector_route"] == "text"
    assert result.metadata["needs_ocr"] is False
    assert result.metadata["ocr_page_count"] == 0
    assert result.needs_ocr is False


def test_pdf_converter_inspector_text_quality_keeps_one_character_as_poor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="text",
        page_count=1,
        pages_needing_ocr=[],
        confidence=0.91,
        has_encoding_issues=False,
        processing_time_ms=6,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="text",
        page_count=1,
        pages_needing_ocr=[],
        has_encoding_issues=False,
        processing_time_ms=10,
        markdown="A",
    )
    _stub_inspector(monkeypatch, detection, extraction)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "tiny.pdf", "application/pdf")
    )

    assert result.quality is not None
    assert result.quality.verdict.value == "poor"
    assert "minimum char count" in result.quality.details


def test_pdf_converter_emits_explicit_ocr_routing_metadata_for_mixed_pdfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="mixed",
        page_count=4,
        pages_needing_ocr=[],
        confidence=0.81,
        has_encoding_issues=False,
        processing_time_ms=7,
        is_complex=True,
        pages_with_tables=[2],
        pages_with_columns=[1, 2],
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="mixed",
        page_count=4,
        # Repeats and out-of-range indices are intentional; converter should
        # dedupe/clip before surfacing routing metadata.
        pages_needing_ocr=[2, True, 0, 2, False, -1, 9],
        has_encoding_issues=False,
        processing_time_ms=13,
        markdown=("mixed inspector markdown " * 12).strip(),
        is_complex=True,
        pages_with_tables=[2],
        pages_with_columns=[1],
    )
    _stub_inspector(monkeypatch, detection, extraction)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "mixed.pdf", "application/pdf")
    )

    assert result.metadata["parser"] == "local:pdf_inspector"
    assert result.metadata["inspector_route"] == "mixed"
    assert result.metadata["needs_ocr"] is True
    assert sorted(result.metadata["ocr_page_indices"]) == [0, 2]
    assert result.metadata["ocr_page_count"] == 2
    assert result.metadata["complex_ocr_page_indices"] == [2]
    assert result.metadata["extraction_ratio"] == 0.5
    assert result.needs_ocr is True
    assert sorted(result.ocr_page_indices or []) == [0, 2]


def test_pdf_converter_mixed_tiny_extraction_is_not_reported_as_good(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="mixed",
        page_count=8,
        pages_needing_ocr=[1, 2, 3, 4, 5, 6, 7, 8],
        confidence=0.75,
        has_encoding_issues=False,
        processing_time_ms=7,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="mixed",
        page_count=8,
        pages_needing_ocr=[1, 2, 3, 4, 5, 6, 7, 8],
        has_encoding_issues=False,
        processing_time_ms=11,
        markdown="ok",
    )
    _stub_inspector(monkeypatch, detection, extraction)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "mixed-tiny.pdf", "application/pdf")
    )

    assert result.quality is not None
    assert result.quality.verdict.value == "poor"
    assert result.quality.details.startswith("pdf inspector mixed extraction:")


def test_pdf_converter_ocr_only_route_keeps_full_page_indices_for_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="imageonly",
        page_count=450,
        pages_needing_ocr=[],
        confidence=0.95,
        has_encoding_issues=False,
        processing_time_ms=5,
    )
    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module, "detect_pdf_with_inspector", lambda file_bytes: detection
    )

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "scan.pdf", "application/pdf")
    )

    assert result.needs_ocr is True
    assert result.ocr_page_indices is not None
    assert len(result.ocr_page_indices) == 450
    assert result.ocr_page_indices[0] == 0
    assert result.ocr_page_indices[-1] == 449
    assert result.metadata["ocr_page_count"] == 450
    assert len(result.metadata["ocr_page_indices"]) == 450
    assert len(result.metadata["ocr_page_indices_telemetry"]) == 400


def test_prepare_bytes_surfaces_ocr_routing_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_prepare_document_bytes(
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None,
        ocr_provider,
        event_sink=None,
        contextualizer=None,
        chunk_context_cache=None,
        chunking_config=None,
    ):
        assert contextualizer is None
        assert chunk_context_cache is None
        return PreparedDocument(
            filename=filename,
            mime_type=mime_type,
            markdown="parsed markdown",
            chunks=[
                PreparedChunk(
                    chunk_index=0,
                    text="parsed markdown",
                    embedding_text="parsed markdown",
                    word_count=2,
                )
            ],
            metadata={
                "needs_ocr": True,
                "ocr_page_indices": [2, 0, 2],
                "parser": "local:pdf_inspector",
                "confidence": "0.62",
            },
            ocr=OcrRoutingSignal(
                needed=True,
                page_indices=[0, 2],
                confidence=0.62,
                parser="local:pdf_inspector",
            ),
        )

    monkeypatch.setattr(
        core_prepare_facade,
        "prepare_document_bytes",
        fake_prepare_document_bytes,
    )

    async def _run() -> None:
        core = RAGCore(
            make_test_config(embedding_dimensions=4),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(),
        )
        try:
            prepared = await core.prepare_bytes(
                file_bytes=b"%PDF-1.7",
                filename="report.pdf",
                mime_type="application/pdf",
            )
        finally:
            await core.close()

        assert [chunk.embedding_text for chunk in prepared.chunks] == [
            "parsed markdown"
        ]
        assert prepared.ocr.needed is True
        assert prepared.ocr.page_indices == [0, 2]
        assert prepared.ocr.confidence == 0.62
        assert prepared.ocr.parser == "local:pdf_inspector"

    asyncio.run(_run())


def test_apply_ocr_replaces_markdown_for_full_document_helpers() -> None:
    class FakeOcrProvider:
        provider_name = "gemini"
        model_name = "gemini-2.5-flash"
        supports_page_selection = False

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            return OcrResult(
                markdown="# OCR Full Document",
                merge_mode="replace",
                provider_name=self.provider_name,
                model_name=self.model_name,
                metadata={
                    "ocr_page_indices_ignored": True,
                    "ocr_processed_entire_document": True,
                },
            )

    async def _run() -> None:
        result = await apply_ocr(
            parsed=ParsedDocument(
                filename="scan.pdf",
                mime_type="application/pdf",
                markdown="# Local Extracted Text",
                metadata={"needs_ocr": True, "ocr_page_indices": [0], "page_count": 4},
            ),
            file_bytes=b"%PDF-1.7",
            provider=FakeOcrProvider(),
        )

        assert result.markdown == "# OCR Full Document"
        ocr_meta = read_ocr_metadata(result.metadata)
        assert ocr_meta.merge_mode == "replace"
        assert ocr_meta.pages_used == (0, 1, 2, 3)
        assert ocr_meta.page_count == 4
        assert ocr_meta.provider == "gemini"
        assert ocr_meta.model == "gemini-2.5-flash"
        assert "ocr_page_indices" not in result.metadata
        assert result.metadata["needs_ocr"] is False

    asyncio.run(_run())


def test_apply_ocr_replaces_matching_page_sections_for_partial_page_ocr() -> None:
    class FakeOcrProvider:
        provider_name = "mistral"
        model_name = "mistral-ocr-latest"
        supports_page_selection = True

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            assert request.page_indices == [2]
            return OcrResult(
                markdown="## Page 3\n\nOCR text for remediated page",
                merge_mode="append",
                provider_name=self.provider_name,
                model_name=self.model_name,
                pages_processed=[2],
            )

    async def _run() -> None:
        result = await apply_ocr(
            parsed=ParsedDocument(
                filename="scan.pdf",
                mime_type="application/pdf",
                markdown=(
                    "## Page 1\n\nReadable page one text\n\n"
                    "## Page 2\n\nReadable page two text\n\n"
                    "## Page 3\n\nUnreadable placeholder text"
                ),
                metadata={"needs_ocr": True, "ocr_page_indices": [2], "page_count": 4},
            ),
            file_bytes=b"%PDF-1.7",
            provider=FakeOcrProvider(),
        )

        assert result.markdown == (
            "## Page 1\n\nReadable page one text\n\n"
            "## Page 2\n\nReadable page two text\n\n"
            "## Page 3\n\nOCR text for remediated page"
        )
        ocr_meta = read_ocr_metadata(result.metadata)
        assert ocr_meta.merge_mode == "append"
        assert ocr_meta.pages_used == (2,)
        assert ocr_meta.provider == "mistral"
        assert result.metadata["ocr_page_indices"] == [2]
        assert result.metadata["quality"]["page_count"] == 4

    asyncio.run(_run())


def test_apply_ocr_partial_page_merge_orders_output_by_page_number() -> None:
    class FakeOcrProvider:
        provider_name = "mistral"
        model_name = "mistral-ocr-latest"
        supports_page_selection = True

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            assert request.page_indices == [0, 2]
            return OcrResult(
                markdown=(
                    "## Page 3\n\nOCR replacement for page three\n\n"
                    "## Page 1\n\nOCR replacement for page one"
                ),
                merge_mode="append",
                provider_name=self.provider_name,
                model_name=self.model_name,
                pages_processed=[2, 0],
            )

    async def _run() -> None:
        result = await apply_ocr(
            parsed=ParsedDocument(
                filename="scan.pdf",
                mime_type="application/pdf",
                markdown=(
                    "# Scan Title\n\nIntro text before page markers.\n\n"
                    "## Page 1\n\nUnreadable placeholder one\n\n"
                    "## Page 2\n\nReadable page two text\n\n"
                    "## Page 3\n\nUnreadable placeholder three"
                ),
                metadata={"needs_ocr": True, "ocr_page_indices": [2, 0], "page_count": 3},
            ),
            file_bytes=b"%PDF-1.7",
            provider=FakeOcrProvider(),
        )

        assert result.markdown == (
            "# Scan Title\n\nIntro text before page markers.\n\n"
            "## Page 1\n\nOCR replacement for page one\n\n"
            "## Page 2\n\nReadable page two text\n\n"
            "## Page 3\n\nOCR replacement for page three"
        )
        assert result.metadata["ocr_page_indices"] == [0, 2]

    asyncio.run(_run())


def test_apply_ocr_partial_page_merge_appends_ocr_only_page() -> None:
    class FakeOcrProvider:
        provider_name = "mistral"
        model_name = "mistral-ocr-latest"
        supports_page_selection = True

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            assert request.page_indices == [3]
            return OcrResult(
                markdown="## Page 4\n\nOCR text for page four",
                merge_mode="append",
                provider_name=self.provider_name,
                model_name=self.model_name,
                pages_processed=[3],
            )

    async def _run() -> None:
        result = await apply_ocr(
            parsed=ParsedDocument(
                filename="scan.pdf",
                mime_type="application/pdf",
                markdown=(
                    "## Page 1\n\nReadable page one text\n\n"
                    "## Page 2\n\nReadable page two text\n\n"
                    "## Page 3\n\nReadable page three text"
                ),
                metadata={"needs_ocr": True, "ocr_page_indices": [3], "page_count": 4},
            ),
            file_bytes=b"%PDF-1.7",
            provider=FakeOcrProvider(),
        )

        assert result.markdown == (
            "## Page 1\n\nReadable page one text\n\n"
            "## Page 2\n\nReadable page two text\n\n"
            "## Page 3\n\nReadable page three text\n\n"
            "## Page 4\n\nOCR text for page four"
        )
        assert result.metadata["ocr_page_indices"] == [3]

    asyncio.run(_run())


def test_apply_ocr_rejects_partial_provider_that_drops_requested_pages() -> None:
    class PartialOcrProvider:
        provider_name = "mistral"
        model_name = "mistral-ocr-latest"
        supports_page_selection = True

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            assert request.page_indices == [0, 2]
            return OcrResult(
                markdown="## OCR Page 1",
                merge_mode="append",
                provider_name=self.provider_name,
                model_name=self.model_name,
                pages_processed=[0],
            )

    async def _run() -> None:
        with pytest.raises(ValueError, match="did not return all requested pages"):
            await apply_ocr(
                parsed=ParsedDocument(
                    filename="scan.pdf",
                    mime_type="application/pdf",
                    markdown="# Local Extracted Text",
                    metadata={
                        "needs_ocr": True,
                        "ocr_page_indices": [0, 2],
                        "page_count": 4,
                    },
                ),
                file_bytes=b"%PDF-1.7",
                provider=PartialOcrProvider(),
            )

    asyncio.run(_run())


def test_apply_ocr_rejects_blank_partial_ocr_result() -> None:
    class BlankOcrProvider:
        provider_name = "mistral"
        model_name = "mistral-ocr-latest"
        supports_page_selection = True

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            assert request.page_indices == [2]
            return OcrResult(
                markdown="   ",
                merge_mode="append",
                provider_name=self.provider_name,
                model_name=self.model_name,
                pages_processed=[2],
            )

    async def _run() -> None:
        with pytest.raises(ValueError, match="OCR provider returned empty markdown"):
            await apply_ocr(
                parsed=ParsedDocument(
                    filename="scan.pdf",
                    mime_type="application/pdf",
                    markdown="# Local Extracted Text",
                    metadata={
                        "needs_ocr": True,
                        "ocr_page_indices": [2],
                        "page_count": 4,
                    },
                ),
                file_bytes=b"%PDF-1.7",
                provider=BlankOcrProvider(),
            )

    asyncio.run(_run())
