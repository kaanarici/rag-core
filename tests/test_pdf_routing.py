from __future__ import annotations

import asyncio
import logging
import sys
from types import SimpleNamespace
from typing import cast

import pytest

import rag_core.core as rag_core_engine
import rag_core.documents.converters.pdf_converter as pdf_converter_module
import rag_core.documents.converters.pdf_converter_inspector as pdf_converter_inspector_module
import rag_core.documents.converters.pdf_converter_extraction as pdf_extraction_module
import rag_core.documents.pdf_inspector as pdf_inspector_module
import rag_core.documents.pdf_inspector_runtime as pdf_inspector_runtime
from rag_core.core import Engine
from rag_core._engine.core_builders import read_ocr_metadata
from rag_core._engine.core_prepare import (
    apply_ocr,
    prepare_document_bytes,
    prepare_text_chunks,
)
from rag_core.config import ChunkingConfig
from rag_core.core_models import (
    OcrRoutingSignal,
    ParsedDocument,
    PreparedChunk,
    PreparedDocument,
)
from rag_core.documents.converters.pdf_converter import PdfConverter
from rag_core.documents.converters.base import ConversionResult, score_text_quality
from rag_core.documents.converters.pdf_converter_extraction import (
    PageExtraction,
    PdfExtraction,
    _extract_page,
    extract_pdf,
)
from rag_core.documents.converters.pdf_converter_pymupdf import (
    pymupdf_conversion_result,
)
from rag_core.documents.ocr import OcrRequest, OcrResult
from rag_core.documents.markdown_headings import parse_atx_heading
from rag_core.documents.pdf_inspector import (
    PdfInspectorDetectionResult,
    PdfInspectorExtractionResult,
    PdfInspectorProcessResult,
    pdf_inspector_enabled,
)
from rag_core.documents.pdf_inspector_payloads import (
    detection_result_from_payload,
    extraction_result_from_payload,
)
from rag_core.documents.pdf_limits import MAX_PDF_PAGE_COUNT
from rag_core.documents.pdf_page_locators import (
    canonicalize_pdf_page_markdown,
    normalize_pdf_page_body,
)
from rag_core.search import SearchResult
from rag_core.search.context_pack import build_context_pack
from rag_core.search.indexer import DocumentIndexer, IndexRequest
from rag_core.search.stored_payload import payload_to_result
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
        pdf_converter_module,
        "process_pdf_with_inspector_wheel",
        lambda file_bytes: None,
    )
    monkeypatch.setattr(
        pdf_converter_module, "detect_pdf_with_inspector", lambda file_bytes: detection
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "extract_pdf_with_inspector",
        lambda file_bytes: extraction,
    )
    monkeypatch.setattr(PdfConverter, "_try_extract_with_pymupdf", fail_pymupdf)


def _inspector_pages(*pages: str) -> str:
    return "\n\n".join(
        f"<!-- Page {page_number} -->\n\n{text}"
        for page_number, text in enumerate(pages, start=1)
    )


def _canonical_pages(*pages: str) -> str:
    return "\n\n".join(
        f"## Page {page_number}\n\n{text}"
        for page_number, text in enumerate(pages, start=1)
    )


async def _indexed_results(prepared: PreparedDocument) -> list[SearchResult]:
    store = RecordingVectorStore()
    indexer = DocumentIndexer(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(include_extra_channel=False),
        vector_store=store,
    )
    await indexer.index_document(
        IndexRequest(
            document_id="headings.pdf",
            collection="pdf-fixtures",
            namespace="fixture",
            text=prepared.markdown,
            filename="headings.pdf",
            mime_type="application/pdf",
            source_type="file",
            document_key="file:headings.pdf",
            content_sha256="sha256:headings.pdf",
            processing_version="test-pdf-context",
            document_metadata=prepared.metadata,
            pre_chunked_texts=[chunk.text for chunk in prepared.chunks],
            embedding_chunk_texts=[chunk.embedding_text for chunk in prepared.chunks],
            chunk_metadata=[dict(chunk.metadata) for chunk in prepared.chunks],
            prepared_chunks=list(prepared.chunks),
        )
    )
    return [
        payload_to_result(point_id=point.id, payload=point.payload, score=0.9)
        for points in store.upsert_calls
        for point in points
    ]


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
    page_one = ("canonical inspector page one " * 4).strip()
    page_two = ("canonical inspector page two " * 4).strip()
    extraction = PdfInspectorExtractionResult(
        pdf_type="text",
        page_count=2,
        pages_needing_ocr=[],
        has_encoding_issues=False,
        processing_time_ms=12,
        markdown=_inspector_pages(page_one, page_two),
    )
    _stub_inspector(monkeypatch, detection, extraction)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "report.pdf", "application/pdf")
    )

    assert result.content == _canonical_pages(page_one, page_two)
    assert result.metadata["parser"] == "local:pdf_inspector"
    assert result.metadata["inspector_route"] == "text"
    assert result.metadata["needs_ocr"] is False
    assert result.metadata["ocr_page_count"] == 0
    assert result.needs_ocr is False


def test_pdf_converter_text_route_preserves_extraction_ocr_pages(
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
        pages_needing_ocr=[1],
        has_encoding_issues=False,
        processing_time_ms=12,
        markdown=_inspector_pages(
            ("readable first page " * 8).strip(),
            ("partial second page " * 8).strip(),
        ),
    )
    _stub_inspector(monkeypatch, detection, extraction)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "report.pdf", "application/pdf")
    )

    assert result.needs_ocr is True
    assert result.ocr_page_indices == [1]
    assert result.metadata["needs_ocr"] is True
    assert result.metadata["ocr_page_indices"] == [1]
    assert result.metadata["ocr_page_count"] == 1
    assert result.metadata["extraction_ratio"] == 0.5


def test_pdf_converter_text_to_mixed_disagreement_unions_ocr_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="text",
        page_count=2,
        pages_needing_ocr=[0],
        confidence=0.99,
        has_encoding_issues=False,
        processing_time_ms=8,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="mixed",
        page_count=2,
        pages_needing_ocr=[1],
        has_encoding_issues=False,
        processing_time_ms=12,
        markdown=_inspector_pages(
            ("partial first page " * 8).strip(),
            ("partial second page " * 8).strip(),
        ),
    )
    _stub_inspector(monkeypatch, detection, extraction)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "report.pdf", "application/pdf")
    )

    assert result.needs_ocr is True
    assert result.ocr_page_indices == [0, 1]
    assert result.metadata["ocr_page_indices"] == [0, 1]
    assert result.metadata["ocr_page_count"] == 2
    assert result.metadata["extraction_ratio"] == 0.0


@pytest.mark.parametrize(
    "extraction_route",
    ["image_only", "imagebased", "scanned", "ocr-only", "ocronly"],
)
def test_pdf_converter_extraction_ocr_only_route_forces_all_pages(
    monkeypatch: pytest.MonkeyPatch,
    extraction_route: str,
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
        pdf_type=extraction_route,
        page_count=2,
        pages_needing_ocr=[],
        has_encoding_issues=False,
        processing_time_ms=12,
        markdown=_inspector_pages(
            ("partial first page " * 8).strip(),
            ("partial second page " * 8).strip(),
        ),
    )
    _stub_inspector(monkeypatch, detection, extraction)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "report.pdf", "application/pdf")
    )

    assert result.needs_ocr is True
    assert result.ocr_page_indices == [0, 1]
    assert result.metadata["ocr_page_indices"] == [0, 1]
    assert result.metadata["ocr_page_count"] == 2
    assert result.metadata["extraction_ratio"] == 0.0


def test_pdf_converter_mixed_to_image_only_forces_all_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="mixed",
        page_count=2,
        pages_needing_ocr=[],
        confidence=0.99,
        has_encoding_issues=False,
        processing_time_ms=8,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="image_only",
        page_count=2,
        pages_needing_ocr=[],
        has_encoding_issues=False,
        processing_time_ms=12,
        markdown=_inspector_pages(
            ("partial first page " * 8).strip(),
            ("partial second page " * 8).strip(),
        ),
    )
    _stub_inspector(monkeypatch, detection, extraction)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "report.pdf", "application/pdf")
    )

    assert result.needs_ocr is True
    assert result.ocr_page_indices == [0, 1]
    assert result.metadata["extraction_ratio"] == 0.0


def test_pdf_converter_invalid_only_ocr_pages_require_fallback() -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="mixed",
        page_count=2,
        pages_needing_ocr=[],
        confidence=0.99,
        has_encoding_issues=False,
        processing_time_ms=8,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="mixed",
        page_count=2,
        pages_needing_ocr=[-1, 99],
        has_encoding_issues=False,
        processing_time_ms=12,
        markdown=_inspector_pages(
            ("partial first page " * 8).strip(),
            ("partial second page " * 8).strip(),
        ),
    )

    result = PdfConverter()._inspector_conversion_from_results(
        detection=detection,
        extraction=extraction,
        page_count=2,
        adapter=None,
    )

    assert result is None


def test_pdf_converter_explicit_empty_mixed_ocr_pages_remain_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="mixed",
        page_count=2,
        pages_needing_ocr=[],
        confidence=0.99,
        has_encoding_issues=False,
        processing_time_ms=8,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="mixed",
        page_count=2,
        pages_needing_ocr=[],
        has_encoding_issues=False,
        processing_time_ms=12,
        markdown=_inspector_pages(
            ("complete first page " * 8).strip(),
            ("complete second page " * 8).strip(),
        ),
    )
    _stub_inspector(monkeypatch, detection, extraction)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "report.pdf", "application/pdf")
    )

    assert result.needs_ocr is False
    assert result.ocr_page_indices is None
    assert "ocr_page_indices" not in result.metadata
    assert result.metadata["ocr_page_count"] == 0
    assert result.metadata["extraction_ratio"] == 1.0


def test_pdf_converter_partially_valid_ocr_pages_preserve_valid_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="mixed",
        page_count=2,
        pages_needing_ocr=[],
        confidence=0.99,
        has_encoding_issues=False,
        processing_time_ms=8,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="mixed",
        page_count=2,
        pages_needing_ocr=[-1, 1, 99],
        has_encoding_issues=False,
        processing_time_ms=12,
        markdown=_inspector_pages(
            ("complete first page " * 8).strip(),
            ("partial second page " * 8).strip(),
        ),
    )
    _stub_inspector(monkeypatch, detection, extraction)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "report.pdf", "application/pdf")
    )

    assert result.needs_ocr is True
    assert result.ocr_page_indices == [1]
    assert result.metadata["ocr_page_indices"] == [1]
    assert result.metadata["extraction_ratio"] == 0.5


@pytest.mark.parametrize(
    "detection_route",
    ["scanned", "image_only", "imagebased", "ocr-only", "ocronly"],
)
def test_pdf_converter_detection_ocr_only_route_requests_every_page_through_apply_ocr(
    monkeypatch: pytest.MonkeyPatch,
    detection_route: str,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type=detection_route,
        page_count=2,
        pages_needing_ocr=[0],
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
        markdown="unused",
    )
    _stub_inspector(monkeypatch, detection, extraction)
    converted = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "scan.pdf", "application/pdf")
    )

    class _EveryPageOcr:
        provider_name = "fixture-ocr"
        model_name = "fixture-v1"
        supports_page_selection = True

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            assert request.page_indices == [0, 1]
            return OcrResult(
                markdown=_canonical_pages("OCR-PAGE-ONE", "OCR-PAGE-TWO"),
                provider_name="fixture-ocr",
                model_name="fixture-v1",
                pages_processed=[0, 1],
            )

    remediated = asyncio.run(
        apply_ocr(
            parsed=ParsedDocument(
                filename="scan.pdf",
                mime_type="application/pdf",
                markdown=converted.content,
                metadata=dict(converted.metadata),
            ),
            file_bytes=b"%PDF-1.7",
            provider=_EveryPageOcr(),
        )
    )

    assert converted.ocr_page_indices == [0, 1]
    assert remediated.metadata["ocr_page_indices"] == [0, 1]
    assert "OCR-PAGE-TWO" in remediated.markdown


def test_pdf_converter_missing_cli_ocr_fields_fall_back_for_mixed_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection = detection_result_from_payload(
        {"pdf_type": "mixed", "page_count": 2, "confidence": 0.9}
    )
    extraction = extraction_result_from_payload(
        {
            "pdf_type": "mixed",
            "page_count": 2,
            "markdown": _inspector_pages("PARTIAL-ONE", "PARTIAL-TWO"),
        }
    )
    fallback_calls: list[str] = []

    async def fallback(
        self: PdfConverter,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        fallback_calls.append(filename)
        content = _canonical_pages("PYMUPDF-ONE", "PYMUPDF-TWO")
        return ConversionResult(
            content=content,
            metadata={"parser": "local:pymupdf", "page_count": 2},
            quality=score_text_quality(content, page_count=2),
        )

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "process_pdf_with_inspector_wheel",
        lambda file_bytes: None,
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "detect_pdf_with_inspector",
        lambda file_bytes: detection,
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "extract_pdf_with_inspector",
        lambda file_bytes: extraction,
    )
    monkeypatch.setattr(PdfConverter, "_try_extract_with_pymupdf", fallback)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "missing-routing.pdf", "application/pdf")
    )

    assert fallback_calls == ["missing-routing.pdf"]
    assert result.metadata["parser"] == "local:pymupdf"


def test_pdf_converter_unknown_cli_extraction_route_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="text",
        page_count=1,
        pages_needing_ocr=[],
        confidence=0.9,
        has_encoding_issues=False,
        processing_time_ms=3,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="mystery",
        page_count=1,
        pages_needing_ocr=[],
        has_encoding_issues=False,
        processing_time_ms=4,
        markdown=("unknown extraction route " * 8).strip(),
    )
    fallback_calls: list[str] = []

    async def fallback(
        self: PdfConverter,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        fallback_calls.append(filename)
        content = _canonical_pages(("trusted fallback " * 8).strip())
        return ConversionResult(
            content=content,
            metadata={"parser": "local:pymupdf", "page_count": 1},
            quality=score_text_quality(content, page_count=1),
        )

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "process_pdf_with_inspector_wheel",
        lambda file_bytes: None,
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "detect_pdf_with_inspector",
        lambda file_bytes: detection,
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "extract_pdf_with_inspector",
        lambda file_bytes: extraction,
    )
    monkeypatch.setattr(PdfConverter, "_try_extract_with_pymupdf", fallback)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "unknown.pdf", "application/pdf")
    )

    assert fallback_calls == ["unknown.pdf"]
    assert result.metadata["parser"] == "local:pymupdf"


def test_pdf_converter_rejects_inconsistent_wheel_extraction_route_then_uses_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = PdfInspectorProcessResult(
        detection=PdfInspectorDetectionResult(
            pdf_type="text",
            page_count=1,
            pages_needing_ocr=[],
            confidence=0.9,
            has_encoding_issues=False,
            processing_time_ms=3,
        ),
        extraction=PdfInspectorExtractionResult(
            pdf_type="mystery",
            page_count=1,
            pages_needing_ocr=[],
            has_encoding_issues=False,
            processing_time_ms=3,
            markdown=("untrusted wheel text " * 8).strip(),
        ),
    )
    cli_detection = PdfInspectorDetectionResult(
        pdf_type="text",
        page_count=1,
        pages_needing_ocr=[],
        confidence=0.9,
        has_encoding_issues=False,
        processing_time_ms=4,
    )
    cli_extraction = PdfInspectorExtractionResult(
        pdf_type="text",
        page_count=1,
        pages_needing_ocr=[],
        has_encoding_issues=False,
        processing_time_ms=5,
        markdown=("trusted cli text " * 8).strip(),
    )
    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "process_pdf_with_inspector_wheel",
        lambda file_bytes: wheel,
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "detect_pdf_with_inspector",
        lambda file_bytes: cli_detection,
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "extract_pdf_with_inspector",
        lambda file_bytes: cli_extraction,
    )

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "wheel.pdf", "application/pdf")
    )

    assert result.metadata["parser"] == "local:pdf_inspector"
    assert "trusted cli text" in result.content
    assert "untrusted wheel text" not in result.content
    assert "inspector_adapter" not in result.metadata


def test_pdf_converter_wheel_ocr_only_without_aliases_requests_all_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = pdf_inspector_module._process_wheel_result(
        {
            "pdf_type": "scanned",
            "page_count": 2,
            "confidence": 0.9,
            "markdown": "",
        }
    )

    def fail_cli(file_bytes: bytes) -> None:
        raise AssertionError("OCR-only wheel result should not fall through to CLI")

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "process_pdf_with_inspector_wheel",
        lambda file_bytes: processed,
    )
    monkeypatch.setattr(pdf_converter_module, "detect_pdf_with_inspector", fail_cli)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "scan.pdf", "application/pdf")
    )

    assert result.needs_ocr is True
    assert result.ocr_page_indices == [0, 1]
    assert result.metadata["inspector_adapter"] == "wheel"


def test_pdf_reconciliation_rejects_amplified_page_count_before_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="scanned",
        page_count=1_000_000,
        pages_needing_ocr=[],
        confidence=0.9,
        has_encoding_issues=False,
        processing_time_ms=3,
    )

    def fail_range(*args: int) -> range:
        raise AssertionError(
            "out-of-range page count must fail before range allocation"
        )

    monkeypatch.setattr(
        pdf_converter_inspector_module, "range", fail_range, raising=False
    )

    routing = pdf_converter_inspector_module._reconciled_inspector_ocr_routing(
        detection=detection,
        extraction=None,
        page_count=1_000_000,
    )

    assert routing is None


def test_pdf_reconciliation_accepts_exact_page_count_bound() -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="scanned",
        page_count=MAX_PDF_PAGE_COUNT,
        pages_needing_ocr=[0],
        confidence=0.9,
        has_encoding_issues=False,
        processing_time_ms=3,
    )

    routing = pdf_converter_inspector_module._reconciled_inspector_ocr_routing(
        detection=detection,
        extraction=None,
        page_count=MAX_PDF_PAGE_COUNT,
    )

    assert routing is not None
    assert len(routing.page_indices) == MAX_PDF_PAGE_COUNT
    assert routing.page_indices[0] == 0
    assert routing.page_indices[-1] == MAX_PDF_PAGE_COUNT - 1


@pytest.mark.parametrize(
    ("page_count", "accepted"),
    [
        (0, True),
        (MAX_PDF_PAGE_COUNT, True),
        (MAX_PDF_PAGE_COUNT + 1, False),
    ],
)
def test_pymupdf_enforces_page_limit_before_page_iteration(
    monkeypatch: pytest.MonkeyPatch,
    page_count: int,
    accepted: bool,
) -> None:
    page_reads = 0

    class FakeDocument:
        needs_pass = False

        def __len__(self) -> int:
            return page_count

        def __getitem__(self, page_index: int) -> object:
            nonlocal page_reads
            page_reads += 1
            return object()

        def close(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "fitz",
        SimpleNamespace(open=lambda **kwargs: FakeDocument()),
    )
    monkeypatch.setattr(
        pdf_extraction_module,
        "_extract_page",
        lambda page, page_num: PageExtraction(
            page_num=page_num,
            text="readable",
            char_count=8,
        ),
    )

    if accepted:
        result = asyncio.run(extract_pdf(b"%PDF"))
        assert result.page_count == page_count
        assert page_reads == page_count
    else:
        with pytest.raises(ValueError, match="page_count must be between"):
            asyncio.run(extract_pdf(b"%PDF"))
        assert page_reads == 0


def test_pymupdf_preserves_structural_source_before_page_normalization() -> None:
    import fitz

    source = "\n".join(
        [
            "# Existing heading",
            "## C#",
            "    ## Page 99",
            "PYMUPDF-STRUCTURE-EVIDENCE " + ("retrievable body " * 12),
        ]
    )
    document = fitz.open()
    page = document.new_page(width=612, height=900)
    page.insert_textbox(
        fitz.Rect(72, 72, 540, 820),
        source,
        fontname="courier",
        fontsize=10,
    )
    try:
        pdf_bytes = cast(bytes, document.tobytes())
    finally:
        document.close()

    extraction = asyncio.run(extract_pdf(pdf_bytes))
    result = pymupdf_conversion_result(
        extraction,
        logger=logging.getLogger(__name__),
    )
    chunks = prepare_text_chunks(
        result.content,
        filename="pymupdf-structure.pdf",
        mime_type="application/pdf",
        chunking_config=ChunkingConfig(max_chars=220, overlap=0),
    )
    prepared = PreparedDocument(
        filename="pymupdf-structure.pdf",
        mime_type="application/pdf",
        markdown=result.content,
        chunks=chunks,
        metadata=result.metadata,
    )

    assert "# Existing heading" in extraction.pages[0].text
    assert "## C#" in extraction.pages[0].text
    assert "    ## Page 99" in extraction.pages[0].text
    assert "## Page 1\n\n# Existing heading" in prepared.markdown
    assert "## C#" in prepared.markdown
    assert "    ## Page 99" in prepared.markdown
    evidence = next(
        chunk for chunk in chunks if "PYMUPDF-STRUCTURE-EVIDENCE" in chunk.text
    )
    assert evidence.metadata["page_number"] == 1
    assert evidence.metadata.get("section_title") != "Page 99"
    assert "Page 99" not in str(evidence.metadata.get("section_path", "")).split(" > ")

    indexed = asyncio.run(_indexed_results(prepared))
    reconstructed = next(
        item for item in indexed if "PYMUPDF-STRUCTURE-EVIDENCE" in item.text
    )
    [snippet] = build_context_pack(
        [reconstructed],
        query="structural source evidence",
    ).snippets
    assert reconstructed.metadata["page_number"] == 1
    assert "Page 1 > Page 99" not in snippet.header
    assert "Page 1 > Page 99" not in snippet.prompt_header


def test_pymupdf_whitespace_padding_does_not_satisfy_text_threshold() -> None:
    class WhitespacePage:
        def get_text(self, mode: str, *, sort: bool = False) -> object:
            if mode == "blocks":
                return [(0, 0, 100, 20, " " * 80, 0, 0)]
            assert mode == "text"
            return " " * 80

        def get_images(self) -> list[object]:
            return []

    extraction = _extract_page(WhitespacePage(), 0)

    assert extraction.text == " " * 80
    assert extraction.char_count == 0
    assert extraction.needs_ocr is True


@pytest.mark.parametrize(
    "source_heading",
    [
        "# page 099",
        " ## Page\t099 ##",
        "  ### PAGE\N{NO-BREAK SPACE}099 ###   ",
        "   #### Page   099",
        "##### page 099 #####",
        "###### PAGE\t099",
    ],
)
def test_pymupdf_page_assembly_escapes_semantic_page_title_grammar(
    source_heading: str,
) -> None:
    result = pymupdf_conversion_result(
        PdfExtraction(
            pages=[
                PageExtraction(
                    page_num=0,
                    text=f"{source_heading}\nvisible evidence",
                    char_count=len(source_heading) + 17,
                )
            ],
            page_count=1,
        ),
        logger=logging.getLogger(__name__),
    )

    parsed_heading = parse_atx_heading(source_heading)
    assert parsed_heading is not None
    marker_start = len(parsed_heading.indent)
    escaped_heading = (
        source_heading[:marker_start]
        + r"\#" * parsed_heading.level
        + source_heading[marker_start + parsed_heading.level :]
    )
    assert escaped_heading in result.content
    assert result.content.splitlines().count("## Page 1") == 1


@pytest.mark.parametrize(
    (
        "detection_route",
        "detection_explicit",
        "extraction_route",
        "extraction_explicit",
    ),
    [
        ("text", True, "mixed", False),
        ("mixed", False, "text", True),
    ],
)
def test_pdf_reconciliation_does_not_borrow_explicit_empty_from_sibling_result(
    detection_route: str,
    detection_explicit: bool,
    extraction_route: str,
    extraction_explicit: bool,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type=detection_route,
        page_count=2,
        pages_needing_ocr=[],
        confidence=0.9,
        has_encoding_issues=False,
        processing_time_ms=3,
        has_explicit_ocr_page_info=detection_explicit,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type=extraction_route,
        page_count=2,
        pages_needing_ocr=[],
        has_encoding_issues=False,
        processing_time_ms=4,
        markdown=_canonical_pages("one", "two"),
        has_explicit_ocr_page_info=extraction_explicit,
    )

    assert (
        pdf_converter_inspector_module._reconciled_inspector_ocr_routing(
            detection=detection,
            extraction=extraction,
            page_count=2,
        )
        is None
    )


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
        markdown=_inspector_pages(
            *(("mixed inspector markdown " * 3).strip() for _ in range(4))
        ),
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
        page_count=1,
        pages_needing_ocr=[0],
        confidence=0.75,
        has_encoding_issues=False,
        processing_time_ms=7,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="mixed",
        page_count=1,
        pages_needing_ocr=[0],
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
        pdf_converter_module,
        "process_pdf_with_inspector_wheel",
        lambda file_bytes: None,
    )
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


def test_pdf_converter_prefers_in_process_inspector_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markdown = ("wheel inspector markdown " * 8).strip()
    processed = PdfInspectorProcessResult(
        detection=PdfInspectorDetectionResult(
            pdf_type="text_based",
            page_count=1,
            pages_needing_ocr=[],
            confidence=0.97,
            has_encoding_issues=False,
            processing_time_ms=3,
        ),
        extraction=PdfInspectorExtractionResult(
            pdf_type="text_based",
            page_count=1,
            pages_needing_ocr=[],
            has_encoding_issues=False,
            processing_time_ms=3,
            markdown=markdown,
        ),
    )

    def fail_cli(file_bytes: bytes) -> None:
        raise AssertionError("CLI fallback should not run when the wheel succeeds")

    async def fail_pymupdf(self, file_bytes: bytes, filename: str, mime_type: str):
        raise AssertionError("PyMuPDF fallback should not run when the wheel succeeds")

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "process_pdf_with_inspector_wheel",
        lambda file_bytes: processed,
    )
    monkeypatch.setattr(pdf_converter_module, "detect_pdf_with_inspector", fail_cli)
    monkeypatch.setattr(PdfConverter, "_try_extract_with_pymupdf", fail_pymupdf)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "report.pdf", "application/pdf")
    )

    assert result.content == _canonical_pages(markdown)
    assert result.metadata["parser"] == "local:pdf_inspector"
    assert result.metadata["inspector_adapter"] == "wheel"
    assert result.metadata["inspector_route"] == "text_based"
    assert result.metadata["confidence"] == 0.97
    assert result.needs_ocr is False


def test_pdf_converter_wheel_text_route_preserves_explicit_ocr_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = PdfInspectorProcessResult(
        detection=PdfInspectorDetectionResult(
            pdf_type="text",
            page_count=1,
            pages_needing_ocr=[0],
            confidence=0.97,
            has_encoding_issues=False,
            processing_time_ms=3,
        ),
        extraction=PdfInspectorExtractionResult(
            pdf_type="text",
            page_count=1,
            pages_needing_ocr=[0],
            has_encoding_issues=False,
            processing_time_ms=3,
            markdown=("partial wheel text " * 8).strip(),
        ),
    )

    def fail_cli(file_bytes: bytes) -> None:
        raise AssertionError("CLI fallback should not run for one trusted wheel page")

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "process_pdf_with_inspector_wheel",
        lambda file_bytes: processed,
    )
    monkeypatch.setattr(pdf_converter_module, "detect_pdf_with_inspector", fail_cli)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "report.pdf", "application/pdf")
    )

    assert result.metadata["inspector_adapter"] == "wheel"
    assert result.needs_ocr is True
    assert result.ocr_page_indices == [0]
    assert result.metadata["ocr_page_indices"] == [0]


def test_pdf_converter_falls_back_to_cli_when_wheel_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="text",
        page_count=1,
        pages_needing_ocr=[],
        confidence=0.91,
        has_encoding_issues=False,
        processing_time_ms=4,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="text",
        page_count=1,
        pages_needing_ocr=[],
        has_encoding_issues=False,
        processing_time_ms=5,
        markdown=("cli inspector markdown " * 8).strip(),
    )
    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "process_pdf_with_inspector_wheel",
        lambda file_bytes: None,
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "detect_pdf_with_inspector",
        lambda file_bytes: detection,
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "extract_pdf_with_inspector",
        lambda file_bytes: extraction,
    )

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "report.pdf", "application/pdf")
    )

    assert result.content == _canonical_pages(extraction.markdown)
    assert result.metadata["parser"] == "local:pdf_inspector"
    assert "inspector_adapter" not in result.metadata


def test_pdf_converter_falls_back_when_wheel_mixed_lacks_page_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel_markdown = ("wheel mixed markdown " * 8).strip()
    processed = PdfInspectorProcessResult(
        detection=PdfInspectorDetectionResult(
            pdf_type="mixed",
            page_count=2,
            pages_needing_ocr=[],
            confidence=0.8,
            has_encoding_issues=False,
            processing_time_ms=3,
            has_explicit_ocr_page_info=False,
        ),
        extraction=PdfInspectorExtractionResult(
            pdf_type="mixed",
            page_count=2,
            pages_needing_ocr=[],
            has_encoding_issues=False,
            processing_time_ms=3,
            markdown=wheel_markdown,
            has_explicit_ocr_page_info=False,
        ),
    )
    cli_detection = PdfInspectorDetectionResult(
        pdf_type="mixed",
        page_count=2,
        pages_needing_ocr=[],
        confidence=0.75,
        has_encoding_issues=False,
        processing_time_ms=4,
    )
    cli_extraction = PdfInspectorExtractionResult(
        pdf_type="mixed",
        page_count=2,
        pages_needing_ocr=[1],
        has_encoding_issues=False,
        processing_time_ms=5,
        markdown=_inspector_pages(
            ("cli mixed page one " * 4).strip(),
            ("cli mixed page two " * 4).strip(),
        ),
    )
    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "process_pdf_with_inspector_wheel",
        lambda file_bytes: processed,
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "detect_pdf_with_inspector",
        lambda file_bytes: cli_detection,
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "extract_pdf_with_inspector",
        lambda file_bytes: cli_extraction,
    )

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "mixed.pdf", "application/pdf")
    )

    assert result.content == _canonical_pages(
        ("cli mixed page one " * 4).strip(),
        ("cli mixed page two " * 4).strip(),
    )
    assert result.metadata["parser"] == "local:pdf_inspector"
    assert "inspector_adapter" not in result.metadata
    assert result.metadata["ocr_page_indices"] == [1]


def test_pdf_converter_does_not_trust_wheel_marker_lookalikes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = PdfInspectorProcessResult(
        detection=PdfInspectorDetectionResult(
            pdf_type="text",
            page_count=2,
            pages_needing_ocr=[],
            confidence=0.9,
            has_encoding_issues=False,
            processing_time_ms=3,
        ),
        extraction=PdfInspectorExtractionResult(
            pdf_type="text",
            page_count=2,
            pages_needing_ocr=[],
            has_encoding_issues=False,
            processing_time_ms=3,
            markdown=_inspector_pages("UNTRUSTED-WHEEL-ONE", "UNTRUSTED-WHEEL-TWO"),
        ),
    )
    cli_detection = PdfInspectorDetectionResult(
        pdf_type="text",
        page_count=2,
        pages_needing_ocr=[],
        confidence=0.9,
        has_encoding_issues=False,
        processing_time_ms=4,
    )
    cli_extraction = PdfInspectorExtractionResult(
        pdf_type="text",
        page_count=2,
        pages_needing_ocr=[],
        has_encoding_issues=False,
        processing_time_ms=5,
        markdown=_inspector_pages("TRUSTED-CLI-ONE", "TRUSTED-CLI-TWO"),
    )

    async def fail_pymupdf(
        self: PdfConverter,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        raise AssertionError("trusted CLI page markers should prevent PyMuPDF fallback")

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "process_pdf_with_inspector_wheel",
        lambda file_bytes: wheel,
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "detect_pdf_with_inspector",
        lambda file_bytes: cli_detection,
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "extract_pdf_with_inspector",
        lambda file_bytes: cli_extraction,
    )
    monkeypatch.setattr(PdfConverter, "_try_extract_with_pymupdf", fail_pymupdf)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "report.pdf", "application/pdf")
    )

    assert result.content == _canonical_pages("TRUSTED-CLI-ONE", "TRUSTED-CLI-TWO")
    assert "UNTRUSTED-WHEEL" not in result.content
    assert "inspector_adapter" not in result.metadata


@pytest.mark.parametrize(
    ("source_heading", "escaped_heading"),
    [
        ("## Page 99", r"\#\# Page 99"),
        ("## Page 2", r"\#\# Page 2"),
        (" ## Page 99", r" \#\# Page 99"),
        ("  ## Page 99", r"  \#\# Page 99"),
        ("   ## Page 99", r"   \#\# Page 99"),
        ("##  Page 99", r"\#\#  Page 99"),
        ("##\tPage 99", "\\#\\#\tPage 99"),
        ("##\N{NO-BREAK SPACE}Page 99", "\\#\\#\N{NO-BREAK SPACE}Page 99"),
        ("## Page 99 ##", r"\#\# Page 99 ##"),
        ("## Page 99   ", r"\#\# Page 99   "),
        ("# Page 99", r"\# Page 99"),
        ("### Page 99", r"\#\#\# Page 99"),
        ("#### Page 99", r"\#\#\#\# Page 99"),
        ("##### Page 99", r"\#\#\#\#\# Page 99"),
        ("###### Page 99", r"\#\#\#\#\#\# Page 99"),
        ("## Page   99", r"\#\# Page   99"),
        ("## Page\t99", "\\#\\# Page\t99"),
        ("## Page\N{NO-BREAK SPACE}99", "\\#\\# Page\N{NO-BREAK SPACE}99"),
        ("## page 99", r"\#\# page 99"),
        ("## PAGE 99", r"\#\# PAGE 99"),
        ("## Page 099", r"\#\# Page 099"),
    ],
)
def test_pdf_source_page_heading_grammar_is_escaped(
    source_heading: str,
    escaped_heading: str,
) -> None:
    assert normalize_pdf_page_body(source_heading) == escaped_heading


def test_pdf_inspector_cli_source_page_heading_does_not_create_locators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_heading = "## PAGE\N{NO-BREAK SPACE}099 ##"
    escaped_heading = "\\#\\# PAGE\N{NO-BREAK SPACE}099 ##"
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
        markdown=_inspector_pages(
            (
                "PAGE-ONE-BEFORE stable source evidence. "
                "This remains on the first page.\n\n"
                "## Methods\n\n"
                "METHODS-EVIDENCE normal headings remain unchanged.\n\n"
                f"{source_heading}\n\n"
                "PAGE-ONE-AFTER collision-adjacent evidence remains on page one."
            ),
            (
                "PAGE-TWO-EVIDENCE belongs only to the second page. "
                "The owned boundary remains authoritative."
            ),
        ),
    )
    _stub_inspector(monkeypatch, detection, extraction)

    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=b"%PDF-1.7",
            filename="headings.pdf",
            mime_type="application/pdf",
            path=None,
            ocr_provider=None,
            chunking_config=ChunkingConfig(max_chars=180, overlap=0),
        )
    )

    page_one_after = next(
        chunk for chunk in prepared.chunks if "PAGE-ONE-AFTER" in chunk.text
    )
    page_two = next(
        chunk for chunk in prepared.chunks if "PAGE-TWO-EVIDENCE" in chunk.text
    )
    assert page_one_after.metadata["page_number"] == 1
    assert page_one_after.metadata["page_index"] == 0
    assert page_two.metadata["page_number"] == 2
    assert page_two.metadata["page_index"] == 1
    parsed_source_heading = parse_atx_heading(source_heading)
    assert parsed_source_heading is not None
    collision_title = parsed_source_heading.title
    assert page_one_after.metadata.get("section_title") != collision_title
    assert collision_title not in str(
        page_one_after.metadata.get("section_path", "")
    ).split(" > ")
    assert escaped_heading in prepared.markdown
    assert "## Methods" in prepared.markdown
    assert "METHODS-EVIDENCE" in prepared.markdown
    assert "<!-- Page " not in prepared.markdown

    reconstructed = next(
        result
        for result in asyncio.run(_indexed_results(prepared))
        if "PAGE-ONE-AFTER" in result.text
    )
    assert reconstructed.metadata["page_number"] == 1
    assert reconstructed.metadata["page_index"] == 0
    assert reconstructed.section_title != collision_title
    assert collision_title not in str(reconstructed.section_path).split(" > ")
    [snippet] = build_context_pack(
        [reconstructed],
        query="collision-adjacent evidence",
    ).snippets
    for header in (snippet.header, snippet.prompt_header):
        assert collision_title not in header.split(" > ")[1:]


def test_pdf_inspector_preserves_page_start_indented_code() -> None:
    cli = pdf_converter_inspector_module._canonicalize_inspector_markdown(
        "<!-- Page 1 -->\n    ## Methods\n    body",
        page_count=1,
        trusted_page_markers=True,
    )
    wheel = pdf_converter_inspector_module._canonicalize_inspector_markdown(
        "    ## Page 99\n    body",
        page_count=1,
        trusted_page_markers=False,
    )

    assert cli == "## Page 1\n\n    ## Methods\n    body"
    assert wheel == "## Page 1\n\n    ## Page 99\n    body"
    cli_chunk = next(
        chunk
        for chunk in prepare_text_chunks(
            cli,
            filename="cli-code.pdf",
            mime_type="application/pdf",
        )
        if "## Methods" in chunk.text
    )
    wheel_chunk = next(
        chunk
        for chunk in prepare_text_chunks(
            wheel,
            filename="wheel-code.pdf",
            mime_type="application/pdf",
        )
        if "## Page 99" in chunk.text
    )
    assert cli_chunk.metadata.get("section_title") == "Page 1"
    assert wheel_chunk.metadata.get("section_title") == "Page 1"


@pytest.mark.parametrize(
    ("markdown", "trusted_page_markers"),
    [
        ("## Page 1\nsource body", False),
        ("<!-- Page 1 -->\n## Page 1\nsource body", True),
    ],
    ids=["single-page-wheel", "trusted-cli-marker"],
)
def test_pdf_inspector_preserves_matching_source_page_heading(
    markdown: str,
    trusted_page_markers: bool,
) -> None:
    canonical = pdf_converter_inspector_module._canonicalize_inspector_markdown(
        markdown,
        page_count=1,
        trusted_page_markers=trusted_page_markers,
    )

    assert canonical == "## Page 1\n\n\\#\\# Page 1\nsource body"


def test_pdf_inspector_single_page_wheel_source_heading_does_not_create_locator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_heading = "  ###\tPage 99 ###"
    escaped_heading = "  \\#\\#\\#\tPage 99 ###"
    processed = PdfInspectorProcessResult(
        detection=PdfInspectorDetectionResult(
            pdf_type="text",
            page_count=1,
            pages_needing_ocr=[],
            confidence=0.97,
            has_encoding_issues=False,
            processing_time_ms=3,
        ),
        extraction=PdfInspectorExtractionResult(
            pdf_type="text",
            page_count=1,
            pages_needing_ocr=[],
            has_encoding_issues=False,
            processing_time_ms=3,
            markdown=(
                "WHEEL-BEFORE stable source evidence.\n\n"
                f"{source_heading}\n\n"
                "WHEEL-AFTER collision-adjacent evidence.\n\n"
                "## Methods\n\n"
                "NORMAL-HEADING-EVIDENCE remains unchanged."
            ),
        ),
    )

    def fail_cli(file_bytes: bytes) -> None:
        raise AssertionError("CLI fallback should not run for one trusted wheel page")

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "process_pdf_with_inspector_wheel",
        lambda file_bytes: processed,
    )
    monkeypatch.setattr(pdf_converter_module, "detect_pdf_with_inspector", fail_cli)

    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=b"%PDF-1.7",
            filename="wheel-heading.pdf",
            mime_type="application/pdf",
            path=None,
            ocr_provider=None,
            chunking_config=ChunkingConfig(max_chars=160, overlap=0),
        )
    )

    collision_chunk = next(
        chunk for chunk in prepared.chunks if "WHEEL-AFTER" in chunk.text
    )
    parsed_source_heading = parse_atx_heading(source_heading)
    assert parsed_source_heading is not None
    collision_title = parsed_source_heading.title
    assert collision_chunk.metadata["page_number"] == 1
    assert collision_chunk.metadata["page_index"] == 0
    assert collision_chunk.metadata.get("section_title") != collision_title
    assert collision_title not in str(
        collision_chunk.metadata.get("section_path", "")
    ).split(" > ")
    assert {chunk.metadata.get("page_number") for chunk in prepared.chunks} <= {1}
    assert escaped_heading in prepared.markdown
    assert "## Methods" in prepared.markdown
    assert "NORMAL-HEADING-EVIDENCE" in prepared.markdown

    reconstructed = next(
        result
        for result in asyncio.run(_indexed_results(prepared))
        if "WHEEL-AFTER" in result.text
    )
    [snippet] = build_context_pack(
        [reconstructed],
        query="wheel collision evidence",
    ).snippets
    assert reconstructed.metadata["page_number"] == 1
    assert collision_title not in str(reconstructed.section_path).split(" > ")
    assert collision_title not in snippet.header.split(" > ")[1:]
    assert collision_title not in snippet.prompt_header.split(" > ")[1:]


def test_pdf_inspector_fenced_source_page_heading_preserves_code_and_locators(
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
        markdown=_inspector_pages(
            (
                "PAGE-ONE-FENCE-BEFORE remains on the first page.\n\n"
                "```markdown\n"
                "## Page 99\n"
                "FENCED-PAGE-HEADING-CODE\n"
                "```\n\n"
                "    ## Page 99\n"
                "    INDENTED-PAGE-HEADING-CODE\n\n"
                "PAGE-ONE-FENCE-AFTER remains on the first page."
            ),
            "PAGE-TWO-FENCE-EVIDENCE belongs to the true second page.",
        ),
    )
    _stub_inspector(monkeypatch, detection, extraction)

    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=b"%PDF-1.7",
            filename="fenced-heading.pdf",
            mime_type="application/pdf",
            path=None,
            ocr_provider=None,
            chunking_config=ChunkingConfig(max_chars=180, overlap=0),
        )
    )

    fenced_code = next(
        chunk for chunk in prepared.chunks if "FENCED-PAGE-HEADING-CODE" in chunk.text
    )
    page_one_after = next(
        chunk for chunk in prepared.chunks if "PAGE-ONE-FENCE-AFTER" in chunk.text
    )
    page_two = next(
        chunk for chunk in prepared.chunks if "PAGE-TWO-FENCE-EVIDENCE" in chunk.text
    )
    assert "```markdown\n## Page 99\nFENCED-PAGE-HEADING-CODE\n```" in (
        prepared.markdown
    )
    assert "    ## Page 99\n    INDENTED-PAGE-HEADING-CODE" in prepared.markdown
    assert fenced_code.metadata["page_number"] == 1
    assert page_one_after.metadata["page_number"] == 1
    assert page_two.metadata["page_number"] == 2
    assert {chunk.metadata.get("page_number") for chunk in prepared.chunks} <= {1, 2}
    assert all(
        chunk.metadata.get("section_title") != "Page 99"
        and "Page 99" not in str(chunk.metadata.get("section_path", "")).split(" > ")
        for chunk in prepared.chunks
    )


def test_pdf_inspector_closes_unclosed_source_fence_before_owned_page_boundary(
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
        markdown=_inspector_pages(
            ("```markdown\n## Page 99\nUNCLOSED-FENCE-PAGE-ONE-EVIDENCE"),
            "UNCLOSED-FENCE-PAGE-TWO-EVIDENCE belongs to the second page.",
        ),
    )
    _stub_inspector(monkeypatch, detection, extraction)

    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=b"%PDF-1.7",
            filename="unclosed-fence.pdf",
            mime_type="application/pdf",
            path=None,
            ocr_provider=None,
            chunking_config=ChunkingConfig(max_chars=180, overlap=0),
        )
    )

    page_one = next(
        chunk
        for chunk in prepared.chunks
        if "UNCLOSED-FENCE-PAGE-ONE-EVIDENCE" in chunk.text
    )
    page_two = next(
        chunk
        for chunk in prepared.chunks
        if "UNCLOSED-FENCE-PAGE-TWO-EVIDENCE" in chunk.text
    )
    assert "UNCLOSED-FENCE-PAGE-ONE-EVIDENCE\n```\n\n## Page 2" in prepared.markdown
    assert page_one.metadata["page_number"] == 1
    assert page_two.metadata["page_number"] == 2
    assert all(
        chunk.metadata.get("section_title") != "Page 99"
        and "Page 99" not in str(chunk.metadata.get("section_path", "")).split(" > ")
        for chunk in prepared.chunks
    )


@pytest.mark.parametrize(
    ("page_count", "extraction_page_count", "markdown"),
    [
        (2, 2, "markerless page one and page two"),
        (2, 2, _inspector_pages("page one only")),
        (
            2,
            2,
            "<!-- Page 1 -->\n\none\n\n<!-- Page 1 -->\n\nduplicate",
        ),
        (
            3,
            3,
            "<!-- Page 1 -->\n\none\n\n<!-- Page 3 -->\n\nthree",
        ),
        (
            2,
            2,
            "<!-- Page 1 -->\n\none\n\n<!-- Page 3 -->\n\nout of range",
        ),
        (
            2,
            2,
            "<!-- Page one -->\n\none\n\n<!-- Page 2 -->\n\ntwo",
        ),
        (2, 1, "concatenated page one and page two"),
    ],
    ids=[
        "markerless",
        "missing",
        "duplicate",
        "skipped",
        "out-of-range",
        "malformed",
        "page-count-mismatch",
    ],
)
def test_pdf_converter_rejects_untrusted_cli_page_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    page_count: int,
    extraction_page_count: int,
    markdown: str,
) -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="text",
        page_count=page_count,
        pages_needing_ocr=[],
        confidence=0.9,
        has_encoding_issues=False,
        processing_time_ms=3,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="text",
        page_count=extraction_page_count,
        pages_needing_ocr=[],
        has_encoding_issues=False,
        processing_time_ms=4,
        markdown=markdown,
    )
    fallback_calls: list[str] = []

    async def fallback(
        self: PdfConverter,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        fallback_calls.append(filename)
        content = _canonical_pages("PYMUPDF-ONE", "PYMUPDF-TWO")
        return ConversionResult(
            content=content,
            metadata={"parser": "local:pymupdf", "page_count": 2},
            quality=score_text_quality(content, page_count=2),
        )

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(
        pdf_converter_module,
        "process_pdf_with_inspector_wheel",
        lambda file_bytes: None,
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "detect_pdf_with_inspector",
        lambda file_bytes: detection,
    )
    monkeypatch.setattr(
        pdf_converter_module,
        "extract_pdf_with_inspector",
        lambda file_bytes: extraction,
    )
    monkeypatch.setattr(PdfConverter, "_try_extract_with_pymupdf", fallback)

    result = asyncio.run(
        PdfConverter().convert(b"%PDF-1.7", "fallback.pdf", "application/pdf")
    )

    assert fallback_calls == ["fallback.pdf"]
    assert result.metadata["parser"] == "local:pymupdf"
    assert "PYMUPDF-TWO" in result.content


def test_prepare_bytes_surfaces_ocr_routing_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_prepare_document_bytes(
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None,
        namespace: str,
        collection: str,
        document_id: str,
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
        rag_core_engine,
        "prepare_document_bytes",
        fake_prepare_document_bytes,
    )

    async def _run() -> None:
        core = Engine(
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
                markdown=_canonical_pages(
                    "OCR full page one",
                    "OCR full page two",
                    "OCR full page three",
                    "OCR full page four",
                ),
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

        assert result.markdown == _canonical_pages(
            "OCR full page one",
            "OCR full page two",
            "OCR full page three",
            "OCR full page four",
        )
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
                    "## Page 1\n\nOCR replacement for page one\n\n"
                    "## Page 3\n\nOCR replacement for page three"
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
                metadata={
                    "needs_ocr": True,
                    "ocr_page_indices": [2, 0],
                    "page_count": 3,
                },
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


@pytest.mark.parametrize(
    "ocr_markdown",
    [
        "markerless OCR page body",
        "## Page 2\n\nwrong physical page",
        "## Page 1\n\nrequested page\n\n## Page 2\n\nextra page",
        "## Page 1\n\nfirst copy\n\n## Page 1\n\nduplicate copy",
        "```markdown\n## Page 1\nfenced fake boundary\n```",
    ],
    ids=[
        "markerless",
        "wrong-page",
        "extra-page",
        "duplicate-page",
        "fenced-fake",
    ],
)
def test_apply_ocr_rejects_unowned_partial_pdf_page_boundaries(
    ocr_markdown: str,
) -> None:
    class InvalidBoundaryOcrProvider:
        provider_name = "local-test"
        model_name = "local-test"
        supports_page_selection = True

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            assert request.page_indices == [0]
            return OcrResult(
                markdown=ocr_markdown,
                merge_mode="append",
                provider_name=self.provider_name,
                model_name=self.model_name,
                pages_processed=[0],
            )

    async def _run() -> None:
        with pytest.raises(ValueError, match="OCR page boundaries"):
            await apply_ocr(
                parsed=ParsedDocument(
                    filename="scan.pdf",
                    mime_type="application/pdf",
                    markdown=_canonical_pages(
                        "unreadable page one",
                        "readable page two",
                    ),
                    metadata={
                        "needs_ocr": True,
                        "ocr_page_indices": [0],
                        "page_count": 2,
                    },
                ),
                file_bytes=b"%PDF-1.7",
                provider=InvalidBoundaryOcrProvider(),
            )

    asyncio.run(_run())


def test_apply_ocr_rejects_real_second_page_hidden_by_unclosed_source_fence() -> None:
    class FencedBoundaryOcrProvider:
        provider_name = "local-test"
        model_name = "local-test"
        supports_page_selection = True

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            assert request.page_indices == [0, 1]
            return OcrResult(
                markdown=(
                    "## Page 1\n\n```\n## Page 99\nsource code\n"
                    "## Page 2\n\nsecond page OCR"
                ),
                merge_mode="append",
                provider_name=self.provider_name,
                model_name=self.model_name,
                pages_processed=[0, 1],
            )

    async def _run() -> None:
        with pytest.raises(ValueError, match="OCR page boundaries"):
            await apply_ocr(
                parsed=ParsedDocument(
                    filename="scan.pdf",
                    mime_type="application/pdf",
                    markdown=_canonical_pages(
                        "unreadable page one",
                        "unreadable page two",
                    ),
                    metadata={
                        "needs_ocr": True,
                        "ocr_page_indices": [0, 1],
                        "page_count": 2,
                    },
                ),
                file_bytes=b"%PDF-1.7",
                provider=FencedBoundaryOcrProvider(),
            )

    asyncio.run(_run())


@pytest.mark.parametrize(
    "markdown",
    [
        "## Page 2\n\ntwo\n\n## Page 1\n\none",
        "non-newline prefix\n## Page 1\n\none\n\n## Page 2\n\ntwo",
        "   ## Page 1\n\none\n\n## Page 2\n\ntwo",
    ],
    ids=["out-of-order", "text-prefix", "space-prefix"],
)
def test_pdf_page_canonicalizer_rejects_order_and_prefix_contradictions(
    markdown: str,
) -> None:
    assert canonicalize_pdf_page_markdown(markdown, [1, 2]) is None


def test_pdf_page_canonicalizer_preserves_fenced_source_heading_in_owned_page() -> None:
    markdown = canonicalize_pdf_page_markdown(
        "## Page 1\n\n```markdown\n## Page 99\nsource code\n```",
        [1],
    )

    assert markdown == "## Page 1\n\n```markdown\n## Page 99\nsource code\n```"


def test_apply_ocr_fenced_source_heading_cannot_reorder_real_page() -> None:
    class FencedSourceOcrProvider:
        provider_name = "local-test"
        model_name = "local-test"
        supports_page_selection = True

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            return OcrResult(
                markdown=("## Page 1\n\n```markdown\n## Page 99\nOCR source code\n```"),
                merge_mode="append",
                provider_name=self.provider_name,
                model_name=self.model_name,
                pages_processed=[0],
            )

    result = asyncio.run(
        apply_ocr(
            parsed=ParsedDocument(
                filename="scan.pdf",
                mime_type="application/pdf",
                markdown=_canonical_pages(
                    "unreadable page one",
                    "readable page two",
                ),
                metadata={
                    "needs_ocr": True,
                    "ocr_page_indices": [0],
                    "page_count": 2,
                },
            ),
            file_bytes=b"%PDF-1.7",
            provider=FencedSourceOcrProvider(),
        )
    )

    assert result.markdown == (
        "## Page 1\n\n```markdown\n## Page 99\nOCR source code\n```\n\n"
        "## Page 2\n\nreadable page two"
    )
    chunks = prepare_text_chunks(
        result.markdown,
        filename="scan.pdf",
        mime_type="application/pdf",
    )
    fenced = next(chunk for chunk in chunks if "OCR source code" in chunk.text)
    page_two = next(chunk for chunk in chunks if "readable page two" in chunk.text)
    assert fenced.metadata["page_number"] == 1
    assert page_two.metadata["page_number"] == 2
    assert all(chunk.metadata.get("section_title") != "Page 99" for chunk in chunks)
