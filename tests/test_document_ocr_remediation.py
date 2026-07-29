from __future__ import annotations

import asyncio
import logging
from urllib import request

from rag_core.core_models import ParsedDocument
from rag_core._engine.core_ocr_metadata import read_ocr_metadata
from rag_core._engine.core_prepare import apply_ocr
from rag_core.documents.converters.pdf_converter_extraction import (
    PageExtraction,
    PdfExtraction,
)
from rag_core.documents.converters.pdf_converter_pymupdf import (
    pymupdf_conversion_result,
)
from rag_core.documents.converters.pdf_converter_inspector_results import (
    inspector_mixed_result,
)
from rag_core.documents.ocr_commands import gemini as gemini_command
from rag_core.documents.ocr_commands import mistral as mistral_command
from rag_core.documents.ocr import OcrRequest, OcrResult
from rag_core.documents.ocr_provider_names import COMMAND_OCR_PROVIDER
from rag_core.documents.pdf_inspector import PdfInspectorDetectionResult
from rag_core.documents.pdf_inspector import PdfInspectorExtractionResult


def test_mistral_partial_ocr_reports_only_response_pages() -> None:
    raw_pages = [
        {"index": 1, "markdown": "page one"},
        {"index": 3, "markdown": "page three"},
    ]

    assert mistral_command._processed_page_indices(raw_pages, [0, 1, 2]) == [0, 2]
    assert mistral_command._collect_markdown(raw_pages, [0, 1, 2]) == (
        "## Page 1\n\npage one\n\n## Page 3\n\npage three"
    )


def test_mistral_partial_ocr_reports_only_pages_with_markdown() -> None:
    raw_pages = [
        {"index": 1, "markdown": "page one"},
        {"index": 3, "markdown": "   "},
    ]

    assert mistral_command._processed_page_indices(raw_pages, [0, 2]) == [0]
    assert mistral_command._collect_markdown(raw_pages, [0, 2]) == (
        "## Page 1\n\npage one"
    )


def test_mistral_partial_ocr_does_not_duplicate_existing_page_heading() -> None:
    raw_pages = [{"index": 3, "markdown": "## Page 3\n\npage three"}]

    assert (
        mistral_command._collect_markdown(raw_pages, [2]) == "## Page 3\n\npage three"
    )


def test_mistral_full_document_ocr_uses_provider_page_indices() -> None:
    raw_pages = [
        {"index": 2, "markdown": "page two"},
        {"index": 5, "markdown": "page five"},
    ]

    assert mistral_command._processed_page_indices(raw_pages, []) == [1, 4]


def test_mistral_full_document_ocr_preserves_page_headings() -> None:
    raw_pages = [
        {"index": 1, "markdown": "page one"},
        {"index": 3, "markdown": "## Page 3\n\npage three"},
    ]

    assert mistral_command._collect_markdown(raw_pages, []) == (
        "## Page 1\n\npage one\n\n## Page 3\n\npage three"
    )


def test_mistral_full_document_ocr_ignores_blank_pages_in_processed_indices() -> None:
    raw_pages = [
        {"index": 1, "markdown": "page one"},
        {"index": 2, "markdown": "   "},
        {"index": 4, "markdown": "page four"},
    ]

    assert mistral_command._processed_page_indices(raw_pages, []) == [0, 3]


def test_gemini_image_ocr_reports_one_page() -> None:
    assert gemini_command._whole_document_page_count("scan.png", "image/png") == 1


def test_gemini_pdf_ocr_uses_existing_page_count_when_known() -> None:
    assert (
        gemini_command._whole_document_page_count(
            "report.pdf",
            "application/pdf",
            {"page_count": 7},
        )
        == 7
    )


def test_gemini_pdf_ocr_does_not_report_unknown_zero_page_count() -> None:
    assert (
        gemini_command._whole_document_page_count(
            "report.pdf",
            "application/pdf",
            {"page_count": 0},
        )
        == 0
    )


def test_read_ocr_metadata_ignores_bool_page_values() -> None:
    metadata: dict[str, object] = {
        "ocr": {
            "pages_used": [0, True, 2, False],
            "page_count": True,
        }
    }

    ocr = read_ocr_metadata(metadata)

    assert ocr.pages_used == (0, 2)
    assert ocr.page_count == 0


def test_gemini_ocr_request_keeps_api_key_out_of_url() -> None:
    req = request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        data=b"{}",
        method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": "secret-key"},
    )

    assert "secret-key" not in req.full_url
    assert req.get_header("X-goog-api-key") == "secret-key"


def test_full_document_ocr_unknown_page_count_is_explicit() -> None:
    class _UnknownPageCountOcrProvider:
        provider_name = COMMAND_OCR_PROVIDER
        model_name = "ocr-test"
        supports_page_selection = False

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            assert request.page_indices == []
            return OcrResult(
                markdown="# OCR text\n\nBody",
                merge_mode="replace",
                provider_name=self.provider_name,
                model_name=self.model_name,
                pages_processed=[],
                metadata={"ocr_processed_entire_document": True},
            )

    parsed = ParsedDocument(
        filename="scan.pdf",
        mime_type="application/pdf",
        markdown="",
        metadata={"needs_ocr": True},
    )

    result = asyncio.run(
        apply_ocr(
            parsed=parsed,
            file_bytes=b"%PDF",
            provider=_UnknownPageCountOcrProvider(),
        )
    )

    assert result.metadata["needs_ocr"] is False
    assert result.metadata["ocr_page_count_unknown"] is True
    assert result.metadata["ocr"]["pages_used"] == ()
    assert result.metadata["ocr"]["page_count"] == 0


def test_pymupdf_mixed_pdf_needs_ocr_when_any_page_needs_ocr() -> None:
    extraction = PdfExtraction(
        page_count=4,
        pages=[
            PageExtraction(
                page_num=0, text="Enough page text for quality.", needs_ocr=False
            ),
            PageExtraction(
                page_num=1, text="More text from a readable page.", needs_ocr=False
            ),
            PageExtraction(page_num=2, text="Another readable page.", needs_ocr=False),
            PageExtraction(page_num=3, needs_ocr=True),
        ],
    )

    result = pymupdf_conversion_result(extraction, logger=logging.getLogger(__name__))

    assert result.metadata["needs_ocr"] is True
    assert result.needs_ocr is True
    assert result.ocr_page_indices == [3]


def test_pdf_inspector_confidence_reaches_conversion_metadata() -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="mixed",
        page_count=2,
        pages_needing_ocr=[1],
        confidence=0.82,
        has_encoding_issues=False,
        processing_time_ms=4,
    )

    result = inspector_mixed_result(
        markdown="readable page text",
        detection=detection,
        extraction=None,
        page_count=2,
    )

    assert result.metadata["confidence"] == 0.82


def test_pdf_inspector_mixed_result_falls_back_to_detection_ocr_pages() -> None:
    detection = PdfInspectorDetectionResult(
        pdf_type="mixed",
        page_count=2,
        pages_needing_ocr=[1],
        confidence=0.82,
        has_encoding_issues=False,
        processing_time_ms=4,
    )
    extraction = PdfInspectorExtractionResult(
        pdf_type="mixed",
        page_count=2,
        pages_needing_ocr=[],
        has_encoding_issues=False,
        processing_time_ms=5,
        markdown="readable page text",
    )

    result = inspector_mixed_result(
        markdown="readable page text",
        detection=detection,
        extraction=extraction,
        page_count=2,
    )

    assert result.needs_ocr is True
    assert result.metadata["needs_ocr"] is True
    assert result.metadata["ocr_page_indices"] == [1]
