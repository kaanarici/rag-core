from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import pytest
import rag_core.documents.converters.pdf_converter as pdf_converter_module
from rag_core.documents.converters import convert_file
from rag_core.documents.converters.pdf_converter_extraction import (
    PageExtraction,
    PdfExtraction,
)
from rag_core.documents.ocr import OcrRequest, OcrResult
from rag_core.documents.converters.xlsx_converter import XlsxConverter
from rag_core.documents.local_parse import LocalParseError, parse_file_bytes
from rag_core.core_prepare import prepare_document_bytes, prepare_text_chunks
from rag_core.core_prepare_figure_locators import with_figure_locators
from rag_core.core_models import PreparedChunk
from rag_core.local_corpus import ManifestPreviewRequest, preview_manifest
from rag_core.manifest_entries import sanitize_manifest_metadata


BytesFactory = Callable[[], bytes]


@dataclass(frozen=True)
class ParserCase:
    name: str
    filename: str
    mime_type: str
    payload: BytesFactory
    expected: tuple[str, ...]
    metadata: dict[str, object]


def _text_bytes() -> bytes:
    return (
        "RAG Core parser regression text.\n\n"
        "The converter should preserve plain text for indexing and quality diagnostics."
    ).encode()


def _code_bytes() -> bytes:
    return (
        "def answer(query: str) -> str:\n    return f'Retrieval answer for {query}'\n"
    ).encode()


def _html_bytes() -> bytes:
    return (
        "<html><body><nav>Skip navigation</nav><main>"
        "<h1>Parser Regression</h1>"
        "<p>Main content should survive HTML extraction.</p>"
        "</main></body></html>"
    ).encode()


def _csv_bytes() -> bytes:
    return b"team,score\nretrieval,98\nparsing,95\n"


def _tsv_bytes() -> bytes:
    return b"team\tscore\nretrieval\t98\nparsing\t95\n"


def _json_bytes() -> bytes:
    return b'{"team": "retrieval", "score": 98, "status": "ready"}'


def _jsonl_bytes() -> bytes:
    return b'{"team": "retrieval"}\n{"team": "parsing"}\n'


def _xml_bytes() -> bytes:
    return b"<root><team>retrieval</team><score>98</score></root>"


def _docx_bytes() -> bytes:
    docx = importlib.import_module("docx")
    document = docx.Document()
    document.add_heading("Retrieval Runbook", level=1)
    document.add_paragraph(
        "This DOCX fixture covers parser regression behavior for headings, "
        "paragraphs, and tables in document ingestion workflows."
    )
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Signal"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "retrieval_quality"
    table.cell(1, 1).text = "high"
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _interleaved_docx_bytes() -> bytes:
    docx = importlib.import_module("docx")
    document = docx.Document()
    document.add_paragraph("Before the table.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Signal"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "latency_budget"
    table.cell(1, 1).text = "tracked"
    document.add_paragraph("After the table.")
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _short_docx_bytes() -> bytes:
    docx = importlib.import_module("docx")
    document = docx.Document()
    document.add_paragraph("OK.")
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _docx_with_two_image_bytes() -> bytes:
    docx = importlib.import_module("docx")
    document = docx.Document()
    document.add_heading("Retrieval Runbook", level=1)
    document.add_paragraph("Figure section with two embedded images.")
    document.add_picture(BytesIO(_png_bytes()))
    document.add_picture(BytesIO(_png_bytes()))
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _docx_with_captioned_image_bytes() -> bytes:
    docx = importlib.import_module("docx")
    document = docx.Document()
    document.add_heading("Retrieval Runbook", level=1)
    document.add_paragraph("Body text anchors the embedded architecture image.")
    shape = document.add_picture(BytesIO(_png_bytes()))
    shape._inline.docPr.attrib["descr"] = "Architecture diagram"
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _image_only_docx_bytes() -> bytes:
    docx = importlib.import_module("docx")
    document = docx.Document()
    document.add_picture(BytesIO(_png_bytes()))
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _pptx_bytes() -> bytes:
    pptx = importlib.import_module("pptx")
    util = importlib.import_module("pptx.util")
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    if slide.shapes.title is not None:
        slide.shapes.title.text = "Retrieval Review"
    text_box = slide.shapes.add_textbox(
        util.Inches(1),
        util.Inches(1.5),
        util.Inches(8),
        util.Inches(2),
    )
    text_box.text_frame.text = (
        "This PPTX fixture verifies slide text extraction, speaker-facing "
        "content, and parser regression behavior for presentation files."
    )
    buffer = BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


def _short_pptx_bytes() -> bytes:
    pptx = importlib.import_module("pptx")
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    if slide.shapes.title is not None:
        slide.shapes.title.text = "Hi"
    buffer = BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


def _xlsx_bytes() -> bytes:
    openpyxl = importlib.import_module("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Signals"
    sheet.append(["Signal", "Value"])
    sheet.append(["retrieval_quality", "high"])
    sheet.append(["parser_regression", "covered"])
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _xlsx_windowed_bytes() -> bytes:
    openpyxl = importlib.import_module("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Signals"
    sheet.append(["Signal", "Value"])
    sheet.append(["retrieval_quality", "high"])
    sheet.append(["parser_regression", "covered"])
    sheet.append(["latency_budget", "tracked"])
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _xlsx_with_skipped_row_bytes() -> bytes:
    openpyxl = importlib.import_module("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Signals"
    sheet.append(["Signal", "Value"])
    sheet.append(["retrieval_quality", "high"])
    sheet.append([None, None])
    sheet.append(["latency_budget", "tracked"])
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _xlsx_single_row_bytes() -> bytes:
    openpyxl = importlib.import_module("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Signals"
    sheet.append(["latency_budget", "tracked"])
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _xlsx_with_two_chart_bytes() -> bytes:
    openpyxl = importlib.import_module("openpyxl")
    chart_module = importlib.import_module("openpyxl.chart")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Signals"
    sheet.append(["Signal", "Value"])
    sheet.append(["quality", 98])
    sheet.append(["latency", 42])

    data = chart_module.Reference(sheet, min_col=2, min_row=1, max_row=3)
    categories = chart_module.Reference(sheet, min_col=1, min_row=2, max_row=3)
    for cell, title in (("D2", "Quality Chart"), ("D18", "Latency Chart")):
        chart = chart_module.BarChart()
        chart.title = title
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(categories)
        sheet.add_chart(chart, cell)

    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01"
        b"\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _pptx_with_image_bytes() -> bytes:
    pptx = importlib.import_module("pptx")
    util = importlib.import_module("pptx.util")
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    if slide.shapes.title is not None:
        slide.shapes.title.text = "Retrieval Review"
    text_box = slide.shapes.add_textbox(
        util.Inches(1),
        util.Inches(1.5),
        util.Inches(8),
        util.Inches(1),
    )
    text_box.text_frame.text = "Slide content with a diagram reference."
    picture = slide.shapes.add_picture(
        BytesIO(_png_bytes()),
        util.Inches(1),
        util.Inches(3),
        width=util.Inches(1),
    )
    picture.name = "Architecture diagram"
    buffer = BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


def _pptx_with_two_image_bytes() -> bytes:
    pptx = importlib.import_module("pptx")
    util = importlib.import_module("pptx.util")
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    if slide.shapes.title is not None:
        slide.shapes.title.text = "Retrieval Review"
    first = slide.shapes.add_picture(
        BytesIO(_png_bytes()),
        util.Inches(1),
        util.Inches(2),
        width=util.Inches(1),
    )
    first.name = "Architecture diagram"
    second = slide.shapes.add_picture(
        BytesIO(_png_bytes()),
        util.Inches(3),
        util.Inches(2),
        width=util.Inches(1),
    )
    second.name = "Pipeline diagram"
    buffer = BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


def _image_only_pptx_bytes() -> bytes:
    pptx = importlib.import_module("pptx")
    util = importlib.import_module("pptx.util")
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.shapes.add_picture(
        BytesIO(_png_bytes()),
        util.Inches(1),
        util.Inches(1),
        width=util.Inches(1),
    )
    buffer = BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


PARSER_CASES = (
    ParserCase(
        name="text",
        filename="notes.md",
        mime_type="text/markdown",
        payload=_text_bytes,
        expected=("RAG Core parser regression text",),
        metadata={"parser": "local:text", "needs_ocr": False},
    ),
    ParserCase(
        name="code",
        filename="answer.py",
        mime_type="text/x-python",
        payload=_code_bytes,
        expected=("def answer", "Retrieval answer"),
        metadata={"parser": "local:code", "language": "python", "needs_ocr": False},
    ),
    ParserCase(
        name="html",
        filename="page.html",
        mime_type="text/html",
        payload=_html_bytes,
        expected=("Parser Regression", "Main content should survive"),
        metadata={"needs_ocr": False},
    ),
    ParserCase(
        name="csv",
        filename="scores.csv",
        mime_type="text/csv",
        payload=_csv_bytes,
        expected=("| team | score |", "| retrieval | 98 |"),
        metadata={"parser": "local:csv", "needs_ocr": False},
    ),
    ParserCase(
        name="tsv",
        filename="scores.tsv",
        mime_type="text/tab-separated-values",
        payload=_tsv_bytes,
        expected=("| team | score |", "| retrieval | 98 |"),
        metadata={"parser": "local:csv", "needs_ocr": False},
    ),
    ParserCase(
        name="json",
        filename="score.json",
        mime_type="application/json",
        payload=_json_bytes,
        expected=("```json", '"team": "retrieval"'),
        metadata={"parser": "local:json", "needs_ocr": False},
    ),
    ParserCase(
        name="jsonl",
        filename="scores.jsonl",
        mime_type="application/x-ndjson",
        payload=_jsonl_bytes,
        expected=("```json", '{"team": "retrieval"}'),
        metadata={"parser": "local:json"},
    ),
    ParserCase(
        name="xml",
        filename="score.xml",
        mime_type="application/xml",
        payload=_xml_bytes,
        expected=("```xml", "<team>retrieval</team>"),
        metadata={"parser": "local:xml", "needs_ocr": False},
    ),
    ParserCase(
        name="docx",
        filename="runbook.docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        payload=_docx_bytes,
        expected=("# Retrieval Runbook", "retrieval_quality"),
        metadata={"parser": "local:python-docx"},
    ),
    ParserCase(
        name="pptx",
        filename="review.pptx",
        mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        payload=_pptx_bytes,
        expected=("## Slide 1", "Retrieval Review", "parser regression behavior"),
        metadata={"parser": "local:python-pptx", "slide_count": 1},
    ),
    ParserCase(
        name="xlsx",
        filename="signals.xlsx",
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        payload=_xlsx_bytes,
        expected=("## Sheet: Signals", "| retrieval_quality | high |"),
        metadata={"parser": "local:openpyxl", "sheet_count": 1, "needs_ocr": False},
    ),
)


@pytest.mark.parametrize("case", PARSER_CASES, ids=[case.name for case in PARSER_CASES])
def test_representative_supported_formats_parse_with_quality_metadata(
    case: ParserCase,
) -> None:
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=case.payload(),
            filename=case.filename,
            mime_type=case.mime_type,
        )
    )

    assert markdown.strip()
    for expected in case.expected:
        assert expected in markdown
    for key, expected_value in case.metadata.items():
        assert metadata.get(key) == expected_value
    assert_quality_metadata(metadata)


def test_image_converter_requires_ocr_without_extracting_text() -> None:
    result = asyncio.run(
        convert_file(
            _png_bytes(),
            filename="scan.png",
            mime_type="image/png",
        )
    )

    assert result.content == ""
    assert result.needs_ocr is True
    assert result.metadata["parser"] == "ocr_required"
    assert result.metadata["needs_ocr"] is True
    assert result.quality is not None
    assert result.quality.verdict.value == "empty"
    assert result.quality.details == "image file requires OCR"


def test_prepare_document_bytes_rejects_ocr_required_document_without_provider() -> None:
    with pytest.raises(ValueError, match="requires OCR"):
        asyncio.run(
            prepare_document_bytes(
                file_bytes=_png_bytes(),
                filename="scan.png",
                mime_type="image/png",
                path=None,
                ocr_provider=None,
            )
        )


def test_prepare_document_bytes_routes_image_bytes_to_injected_ocr_provider() -> None:
    class FakeImageOcrProvider:
        provider_name = "fake"
        model_name = "fake-model"
        supports_page_selection = False

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            assert request.filename == "scan.png"
            assert request.mime_type == "image/png"
            assert request.page_indices == []
            return OcrResult(
                markdown="# OCR Image Text",
                merge_mode="replace",
                provider_name=self.provider_name,
                model_name=self.model_name,
            )

    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_png_bytes(),
            filename="scan.png",
            mime_type="image/png",
            path=None,
            ocr_provider=FakeImageOcrProvider(),
        )
    )

    assert prepared.markdown == "# OCR Image Text"
    assert prepared.metadata["needs_ocr"] is False
    assert any(chunk.embedding_text == "# OCR Image Text" for chunk in prepared.chunks)


def test_prepare_document_bytes_rejects_blank_ocr_result() -> None:
    class BlankImageOcrProvider:
        provider_name = "fake"
        model_name = "fake-model"
        supports_page_selection = False

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            return OcrResult(
                markdown="   ",
                merge_mode="replace",
                provider_name=self.provider_name,
                model_name=self.model_name,
            )

    with pytest.raises(ValueError, match="OCR provider returned empty markdown"):
        asyncio.run(
            prepare_document_bytes(
                file_bytes=_png_bytes(),
                filename="scan.png",
                mime_type="image/png",
                path=None,
                ocr_provider=BlankImageOcrProvider(),
            )
        )


def test_prepare_document_bytes_recomputes_quality_after_ocr() -> None:
    class FakeImageOcrProvider:
        provider_name = "fake"
        model_name = "fake-model"
        supports_page_selection = False

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            return OcrResult(
                markdown="# OCR Image Text\n\nThis image text is now extracted for indexing.",
                merge_mode="replace",
                provider_name=self.provider_name,
                model_name=self.model_name,
            )

    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_png_bytes(),
            filename="scan.png",
            mime_type="image/png",
            path=None,
            ocr_provider=FakeImageOcrProvider(),
        )
    )

    quality = prepared.metadata["quality"]
    assert isinstance(quality, dict)
    assert quality["verdict"] != "empty"
    assert quality["char_count"] == len(prepared.markdown.strip())


def test_prepare_document_bytes_normalizes_ocr_processed_pages() -> None:
    class MixedPageOcrProvider:
        provider_name = "fake"
        model_name = "fake-model"
        supports_page_selection = False

        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            return OcrResult(
                markdown="# OCR Image Text",
                merge_mode="replace",
                provider_name=self.provider_name,
                model_name=self.model_name,
                pages_processed=[True, 2, False, 0, 2],
            )

    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_png_bytes(),
            filename="scan.png",
            mime_type="image/png",
            path=None,
            ocr_provider=MixedPageOcrProvider(),
        )
    )

    assert prepared.metadata["ocr_page_indices"] == [0, 2]


def test_tsv_mime_without_extension_routes_to_csv_converter() -> None:
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=_tsv_bytes(),
            filename="scores",
            mime_type="text/tab-separated-values",
        )
    )

    assert "| team | score |" in markdown
    assert metadata["parser"] == "local:csv"
    assert metadata["needs_ocr"] is False


def test_ndjson_extension_routes_to_jsonl_without_ndjson_mime() -> None:
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=_jsonl_bytes(),
            filename="events.ndjson",
            mime_type="application/octet-stream",
        )
    )

    assert markdown == '```jsonl\n{"team": "retrieval"}\n{"team": "parsing"}\n```'
    assert metadata["parser"] == "local:json"
    assert metadata["format"] == "jsonl"
    assert metadata["record_count"] == 2
    assert "parse_error" not in metadata


@pytest.mark.parametrize(
    "mime_type",
    ["application/jsonlines", "application/ldjson", "application/x-ldjson"],
)
def test_jsonl_alias_mime_types_route_to_jsonl_converter(mime_type: str) -> None:
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=_jsonl_bytes(),
            filename="events",
            mime_type=mime_type,
        )
    )

    assert markdown == '```jsonl\n{"team": "retrieval"}\n{"team": "parsing"}\n```'
    assert metadata["parser"] == "local:json"
    assert metadata["format"] == "jsonl"


def test_headerless_csv_does_not_promote_first_data_row_to_header() -> None:
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=b"100,200\n300,400\n",
            filename="matrix.csv",
            mime_type="text/csv",
        )
    )

    assert "| Col 1 | Col 2 |" in markdown
    assert "| 100 | 200 |" in markdown
    assert metadata["has_header"] is False


@pytest.mark.parametrize(
    ("filename", "mime_type", "message"),
    [
        (
            "corrupt.pdf",
            "application/pdf",
            "PDF parse failed",
        ),
        (
            "corrupt.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "DOCX parse failed",
        ),
        (
            "corrupt.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "PPTX parse failed",
        ),
        (
            "corrupt.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "XLSX parse failed",
        ),
    ],
)
def test_corrupt_office_files_fail_as_parse_errors(
    filename: str,
    mime_type: str,
    message: str,
) -> None:
    with pytest.raises(LocalParseError, match=message):
        asyncio.run(
            parse_file_bytes(
                file_bytes=b"not an ooxml zip",
                filename=filename,
                mime_type=mime_type,
            )
        )


def test_legacy_office_format_rejection_keeps_specific_reason() -> None:
    with pytest.raises(LocalParseError) as exc_info:
        asyncio.run(
            parse_file_bytes(
                file_bytes=b"legacy office bytes",
                filename="legacy.doc",
                mime_type="application/msword",
            )
        )

    message = str(exc_info.value)
    assert "Unsupported format" in message
    assert "extension '.doc'" in message


def test_pptx_figure_metadata_becomes_prepared_chunk_locator() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_pptx_with_image_bytes(),
            filename="review.pptx",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            path=None,
            ocr_provider=None,
        )
    )

    figure_chunks = [
        chunk
        for chunk in prepared.chunks
        if chunk.metadata.get("figure_id") == "fig:slide:1:1"
    ]

    assert len(figure_chunks) == 1
    assert figure_chunks[0].metadata["slide_number"] == 1
    assert figure_chunks[0].metadata["page_index"] == 0
    assert figure_chunks[0].metadata["figure_caption"] == "Architecture diagram"
    assert "Slide 1 Figure 1" not in prepared.markdown


def test_pptx_multiple_figures_become_distinct_prepared_chunk_locators() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_pptx_with_two_image_bytes(),
            filename="review.pptx",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            path=None,
            ocr_provider=None,
        )
    )

    figure_items = prepared.metadata.get("figure_items")
    assert isinstance(figure_items, list)
    assert {item["figure_id"] for item in figure_items} == {
        "fig:slide:1:1",
        "fig:slide:1:2",
    }
    assert "Slide 1 Figure 1" not in prepared.markdown


def test_docx_multiple_figures_become_distinct_prepared_chunk_locators() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_docx_with_two_image_bytes(),
            filename="runbook.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            path=None,
            ocr_provider=None,
        )
    )

    figure_chunks = [
        chunk for chunk in prepared.chunks if chunk.metadata.get("figure_id")
    ]

    assert figure_chunks == []
    assert prepared.metadata["figure_count"] == 2
    figure_items = prepared.metadata["figure_items"]
    assert isinstance(figure_items, list)
    assert {item["figure_id"] for item in figure_items} == {"fig:docx:1", "fig:docx:2"}
    assert {item["description"] for item in figure_items} == {""}


def test_docx_captioned_figure_metadata_becomes_prepared_chunk_locator() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_docx_with_captioned_image_bytes(),
            filename="runbook.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            path=None,
            ocr_provider=None,
        )
    )

    figure_chunks = [
        chunk
        for chunk in prepared.chunks
        if chunk.metadata.get("figure_id") == "fig:docx:1"
    ]

    assert len(figure_chunks) == 1
    assert figure_chunks[0].metadata["figure_caption"] == "Architecture diagram"
    assert "Body text anchors the embedded architecture image." in figure_chunks[0].text
    assert "DOCX Figure 1" not in prepared.markdown
    assert "Architecture diagram" not in prepared.markdown


def test_docx_and_pptx_avoid_synthetic_figure_placeholder_markdown() -> None:
    docx_markdown, _ = asyncio.run(
        parse_file_bytes(
            file_bytes=_docx_with_two_image_bytes(),
            filename="runbook.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    )
    pptx_markdown, _ = asyncio.run(
        parse_file_bytes(
            file_bytes=_pptx_with_two_image_bytes(),
            filename="review.pptx",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    )

    assert "DOCX Figure " not in docx_markdown
    assert "Embedded image extracted from" not in docx_markdown
    assert "Slide 1 Figure " not in pptx_markdown
    assert "Embedded image extracted from slide" not in pptx_markdown


def test_docx_preserves_body_order_across_paragraphs_and_tables() -> None:
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=_interleaved_docx_bytes(),
            filename="interleaved.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    )

    before_index = markdown.index("Before the table.")
    table_index = markdown.index("| Signal | Value |")
    after_index = markdown.index("After the table.")
    assert before_index < table_index < after_index
    assert metadata["parser"] == "local:python-docx"
    assert metadata["needs_ocr"] is False


def test_figure_locator_matching_does_not_use_raw_substrings() -> None:
    chunks = [
        PreparedChunk(
            chunk_index=0,
            text="This config:slide:1:1 mention is not a figure id.",
            embedding_text="This config:slide:1:1 mention is not a figure id.",
            word_count=7,
        ),
        PreparedChunk(
            chunk_index=1,
            text="The exact fig:slide:1:1 locator appears here.",
            embedding_text="The exact fig:slide:1:1 locator appears here.",
            word_count=6,
        ),
    ]

    annotated = with_figure_locators(
        chunks=chunks,
        metadata={
            "figure_items": [
                {
                    "figure_id": "fig:slide:1:1",
                    "page_index": 3,
                    "label": "Fig 1",
                    "description": "Architecture diagram",
                }
            ]
        },
    )

    assert annotated[0].metadata.get("figure_id") is None
    assert annotated[1].metadata["figure_id"] == "fig:slide:1:1"
    assert annotated[1].metadata["page_index"] == 3


def test_xlsx_row_windows_become_prepared_chunk_locators() -> None:
    result = asyncio.run(
        XlsxConverter(max_rows=2).convert(
            _xlsx_windowed_bytes(),
            filename="signals.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    )

    chunks = prepare_text_chunks(
        result.content,
        filename="signals.xlsx",
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    assert "## Sheet: Signals (Rows 1-2)" in result.content
    assert "## Sheet: Signals (Rows 3-4)" in result.content
    assert [chunk.metadata.get("sheet_name") for chunk in chunks] == [
        "Signals",
        "Signals",
    ]
    assert [chunk.metadata.get("row_range") for chunk in chunks] == [
        "1-2",
        "3-4",
    ]


def test_xlsx_row_windows_use_original_rows_and_preserve_tail_table() -> None:
    result = asyncio.run(
        XlsxConverter(max_rows=2).convert(
            _xlsx_with_skipped_row_bytes(),
            filename="signals.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    )

    chunks = prepare_text_chunks(
        result.content,
        filename="signals.xlsx",
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    tail_section = result.content.split("## Sheet: Signals (Rows 4-4)", 1)[1]

    assert "## Sheet: Signals (Rows 1-2)" in result.content
    assert "## Sheet: Signals (Rows 4-4)" in result.content
    assert "## Sheet: Signals (Rows 3-3)" not in result.content
    assert "| Signal | Value |" in tail_section
    assert "| latency_budget | tracked |" in tail_section
    assert "- latency_budget" not in tail_section
    assert [chunk.metadata.get("row_range") for chunk in chunks] == [
        "1-2",
        "4-4",
    ]


def test_xlsx_single_row_renders_as_table() -> None:
    result = asyncio.run(
        XlsxConverter().convert(
            _xlsx_single_row_bytes(),
            filename="signals.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    )

    assert "| latency_budget | tracked |" in result.content
    assert "- latency_budget" not in result.content


def test_xlsx_multiple_charts_become_distinct_prepared_chunk_locators() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_xlsx_with_two_chart_bytes(),
            filename="signals.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            path=None,
            ocr_provider=None,
        )
    )

    figure_chunks = [
        chunk for chunk in prepared.chunks if chunk.metadata.get("figure_id")
    ]

    assert {chunk.metadata["figure_id"] for chunk in figure_chunks} == {
        "fig:sheet:1:chart:1",
        "fig:sheet:1:chart:2",
    }
    assert {chunk.metadata["sheet_name"] for chunk in figure_chunks} == {"Signals"}
    assert {chunk.metadata["figure_caption"] for chunk in figure_chunks} == {
        "Quality Chart",
        "Latency Chart",
    }


def test_pdf_parse_records_page_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = (
        "Retrieval PDF parser contract fixture with enough text for "
        "quality scoring and page-level extraction coverage."
    )

    async def fake_extract_pdf(file_bytes: bytes) -> PdfExtraction:
        assert file_bytes == b"%PDF-1.4"
        return PdfExtraction(
            pages=[
                PageExtraction(
                    page_num=0,
                    text=text,
                    needs_ocr=False,
                    char_count=len(text),
                )
            ],
            page_count=1,
        )

    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: False)
    monkeypatch.setattr(pdf_converter_module, "extract_pdf", fake_extract_pdf)

    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=b"%PDF-1.4",
            filename="fixture.pdf",
            mime_type="application/pdf",
        )
    )

    assert "## Page 1" in markdown
    assert "Retrieval PDF parser contract fixture" in markdown
    assert metadata["parser"] == "local:pymupdf"
    assert metadata["page_count"] == 1
    assert metadata["ocr_page_count"] == 0
    assert metadata["extraction_ratio"] == 1.0
    assert metadata["needs_ocr"] is False
    assert_quality_metadata(metadata)


def test_local_manifest_preview_uses_same_converter_metadata_as_direct_bytes(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "scores.csv"
    payload = _csv_bytes()
    file_path.write_bytes(payload)

    _, direct_metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=payload,
            filename=file_path.name,
            mime_type="text/csv",
        )
    )
    preview = asyncio.run(
        preview_manifest(
            ManifestPreviewRequest(
                path=file_path,
                namespace="acme",
                corpus_id="docs",
            )
        )
    )

    assert preview.manifest_entry.parser == direct_metadata["parser"]
    assert preview.manifest_entry.needs_ocr == direct_metadata["needs_ocr"]
    assert (
        preview.manifest_entry.metadata["quality"]
        == sanitize_manifest_metadata(direct_metadata)["quality"]
    )


def test_short_docx_keeps_extracted_text_without_forcing_ocr() -> None:
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=_short_docx_bytes(),
            filename="short.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    )

    assert markdown.strip() == "OK."
    assert metadata["needs_ocr"] is False
    assert metadata["quality"]["char_count"] == 3
    assert metadata["quality_warning"] == "short_extracted_text"


def test_short_pptx_keeps_extracted_text_without_forcing_ocr() -> None:
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=_short_pptx_bytes(),
            filename="short.pptx",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    )

    assert "## Slide 1" in markdown
    assert "Hi" in markdown
    assert metadata["needs_ocr"] is False
    assert metadata["quality"]["char_count"] < 50
    assert metadata["quality_warning"] == "short_extracted_text"


def test_image_only_docx_requires_ocr_without_indexing_placeholder_text() -> None:
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=_image_only_docx_bytes(),
            filename="scan.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    )

    assert markdown == ""
    assert metadata["needs_ocr"] is True
    assert metadata["figure_count"] == 1
    assert metadata["text_char_count"] == 0
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_image_only_docx_bytes(),
            filename="scan.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            path=None,
            ocr_provider=None,
            allow_needs_ocr=True,
        )
    )
    assert prepared.chunks == []
    assert prepared.ocr.needed is True


def test_image_only_pptx_requires_ocr_without_indexing_placeholder_text() -> None:
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=_image_only_pptx_bytes(),
            filename="scan.pptx",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    )

    assert markdown == ""
    assert metadata["needs_ocr"] is True
    assert metadata["figure_count"] == 1
    assert metadata["text_char_count"] == 0
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_image_only_pptx_bytes(),
            filename="scan.pptx",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            path=None,
            ocr_provider=None,
            allow_needs_ocr=True,
        )
    )
    assert prepared.chunks == []
    assert prepared.ocr.needed is True


def test_text_binary_payload_is_rejected_instead_of_indexed() -> None:
    with pytest.raises(LocalParseError):
        asyncio.run(
            parse_file_bytes(
                file_bytes=(b"\x00\x01\x02\x03" * 256),
                filename="blob.txt",
                mime_type="text/plain",
            )
        )


def test_local_manifest_preview_keeps_ocr_required_metadata_without_provider(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "scan.png"
    file_path.write_bytes(_png_bytes())

    preview = asyncio.run(
        preview_manifest(
            ManifestPreviewRequest(
                path=file_path,
                namespace="acme",
                corpus_id="docs",
            )
        )
    )

    assert preview.document.ocr.needed is True
    assert preview.document.chunk_count == 0
    assert preview.manifest_entry.needs_ocr is True
    assert preview.manifest_entry.metadata["parser"] == "ocr_required"


def assert_quality_metadata(metadata: dict[str, Any]) -> None:
    quality = metadata.get("quality")
    assert isinstance(quality, dict)
    assert quality["verdict"] in {"good", "poor", "empty"}
    assert isinstance(quality["details"], str)
    assert isinstance(quality["char_count"], int)
    assert quality["char_count"] > 0
    assert isinstance(quality["meaningful_ratio"], float)
    assert isinstance(quality["mojibake_ratio"], float)
    assert isinstance(quality["text_to_page_ratio"], float)
    assert isinstance(quality["page_count"], int)
