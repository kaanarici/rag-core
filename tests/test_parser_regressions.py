from __future__ import annotations

import asyncio
import importlib
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import pytest
import rag_core.documents.converters.pdf_converter as pdf_converter_module
import rag_core.documents.chunking.budget as chunk_budget_module
import rag_core.documents.chunking.xlsx_fragments as xlsx_fragments_module
from rag_core.core import Engine
from rag_core.config import ChunkingConfig
from rag_core.documents.converters import convert_file
from rag_core.documents.converters.pdf_converter_extraction import (
    PageExtraction,
    PdfExtraction,
    _extract_page_layout_text,
    _order_text_blocks,
)
from rag_core.documents.chunking.budget import token_budget_for_char_limit
from rag_core.documents.chunking.protocol import ChunkConfig
from rag_core.documents.chunking.xlsx import XlsxChunker
from rag_core.documents.ocr import OcrRequest, OcrResult
from rag_core.documents.converters.xlsx_converter import XlsxConverter
from rag_core.documents.local_parse import LocalParseError, parse_file_bytes
from rag_core._engine.core_prepare import prepare_document_bytes, prepare_text_chunks
from rag_core._engine.core_prepare_figure_locators import with_figure_locators
from rag_core.core_models import (
    IngestedDocument,
    PreparedChunk,
    PreparedDocument,
    estimate_token_count,
)
from rag_core.ingest.local import ManifestPreviewRequest, preview_manifest
from rag_core.manifest.entries import sanitize_manifest_metadata
from rag_core.search.context_pack import Context, build_context_pack
from rag_core.search.indexer import DocumentIndexer, IndexRequest
from rag_core.search.stored_payload import payload_to_result
from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_test_config,
)


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


def _minimal_pdf_bytes(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


def _docx_bytes() -> bytes:
    docx = importlib.import_module("docx")
    document = docx.Document()
    document.add_heading("Retrieval Runbook", level=1)
    document.add_paragraph(
        "This DOCX fixture covers parser regression behavior for headings, "
        "paragraphs, and tables in document ingestion flows."
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


def _docx_with_long_text_and_image_bytes() -> bytes:
    docx = importlib.import_module("docx")
    document = docx.Document()
    document.add_paragraph("界" * 600)
    shape = document.add_picture(BytesIO(_png_bytes()))
    shape._inline.docPr.attrib["descr"] = "Long document diagram"
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _docx_with_true_inline_image_bytes() -> bytes:
    docx = importlib.import_module("docx")
    document = docx.Document()
    paragraph = document.add_paragraph()
    paragraph.add_run("INLINE-BEFORE ")
    shape = paragraph.add_run().add_picture(BytesIO(_png_bytes()))
    shape._inline.docPr.attrib["descr"] = "Inline precision figure"
    paragraph.add_run(" INLINE-AFTER " + ("tailword " * 700))
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _docx_with_table_inline_image_bytes() -> bytes:
    docx = importlib.import_module("docx")
    document = docx.Document()
    table = document.add_table(rows=1, cols=1)
    paragraph = table.cell(0, 0).paragraphs[0]
    paragraph.add_run(("table-prefix " * 80) + "TABLE-BEFORE ")
    shape = paragraph.add_run().add_picture(BytesIO(_png_bytes()))
    shape._inline.docPr.attrib["descr"] = "Table precision figure"
    paragraph.add_run(" TABLE-AFTER " + ("table-suffix " * 80))
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _docx_with_many_inline_images_bytes(count: int) -> bytes:
    docx = importlib.import_module("docx")
    document = docx.Document()
    paragraph = document.add_paragraph("before ")
    paragraph.add_run().add_picture(BytesIO(_png_bytes()))
    paragraph.add_run(" after")
    template = deepcopy(paragraph._p)
    body = document.element.body
    section_properties = body.sectPr
    for figure_number in range(2, count + 1):
        clone = deepcopy(template)
        for node in clone.iter():
            if str(node.tag).rpartition("}")[2] == "docPr":
                node.set("id", str(figure_number))
        body.insert(body.index(section_properties), clone)
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


def _xlsx_large_window_bytes() -> bytes:
    openpyxl = importlib.import_module("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Supplier Inventory"
    sheet.append(["record", "state", "inspection"])
    for index in range(1, 301):
        inspection = (
            "supplier lot KQ-733 is quarantined pending chromatic inspection"
            if index == 299
            else f"routine receiving check {index:03d}"
        )
        sheet.append([f"lot-{index:03d}", "received", inspection])
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _xlsx_oversized_row_bytes() -> bytes:
    openpyxl = importlib.import_module("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Oversized"
    sheet.append(["record", "detail"])
    sheet.append(
        [
            "OVERSIZED-ROW-NEEDLE",
            "dense-cell-" + ("abcdefghij" * 600),
        ]
    )
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _xlsx_oversized_header_bytes(*, include_row: bool) -> bytes:
    openpyxl = importlib.import_module("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Wide Header"
    sheet.append([f"HEADER-{index}-" + ("界" * 100) for index in range(24)])
    if include_row:
        sheet.append(["HEADER-ROW-PAYLOAD", *(["value"] * 23)])
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _xlsx_multiline_cells_bytes() -> bytes:
    openpyxl = importlib.import_module("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Multiline"
    sheet.append(["Head A\nHead B", "Status"])
    sheet.append(["ROW-2\nCONTINUED", "ready"])
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _xlsx_literal_row_suffix_sheet_bytes() -> bytes:
    openpyxl = importlib.import_module("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Data (Rows 1-2)"
    sheet.append(["record", "value"])
    for index in range(1, 6):
        sheet.append([f"literal-sheet-row-{index}", f"value-{index}"])
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _normalized_xlsx_rows(row_count: int) -> str:
    lines = [
        "## Sheet: Scaling",
        "",
        "| record | value | <!-- rag-core-xlsx-row:1 -->",
        "| --- | --- |",
    ]
    lines.extend(
        f"| row-{row_number} | bounded value {row_number} | "
        f"<!-- rag-core-xlsx-row:{row_number} -->"
        for row_number in range(2, row_count + 2)
    )
    return "\n".join(lines)


def _normalized_wide_xlsx_header(column_count: int) -> str:
    headers = [f"COLUMN-{index:05d}" for index in range(column_count)]
    return (
        "## Sheet: Wide\n\n"
        f"| {' | '.join(headers)} | <!-- rag-core-xlsx-row:1 -->\n"
        f"| {' | '.join('---' for _ in headers)} |"
    )


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


def _xlsx_with_chart_only_sheet_bytes(
    *,
    chart_sheet_title: str = "Dashboard",
) -> bytes:
    openpyxl = importlib.import_module("openpyxl")
    chart_module = importlib.import_module("openpyxl.chart")
    workbook = openpyxl.Workbook()
    data_sheet = workbook.active
    data_sheet.title = "Data"
    data_sheet.append(["Month", "Quality"])
    data_sheet.append(["January", 98])
    data_sheet.append(["February", 99])
    chart_sheet = workbook.create_sheet(chart_sheet_title)
    chart = chart_module.BarChart()
    chart.title = "Quality Trend"
    chart.add_data(
        chart_module.Reference(data_sheet, min_col=2, min_row=1, max_row=3),
        titles_from_data=True,
    )
    chart.set_categories(
        chart_module.Reference(data_sheet, min_col=1, min_row=2, max_row=3)
    )
    chart_sheet.add_chart(chart, "A1")
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


def _two_page_pdf_bytes() -> bytes:
    fitz = importlib.import_module("fitz")
    document = fitz.open()
    for page_number, evidence in (
        (1, "PAGE-ONE-EVIDENCE archive intake"),
        (2, "PAGE-TWO-EVIDENCE cold storage"),
    ):
        page = document.new_page()
        page.insert_text((72, 72), evidence)
        page.insert_text(
            (72, 96),
            f"Page {page_number} carries enough stable text for extraction quality.",
        )
        page.insert_text(
            (72, 120),
            "Locator boundaries must survive document preparation and chunking.",
        )
    payload = bytes(document.tobytes())
    document.close()
    return payload


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


def _pptx_with_long_text_and_image_bytes() -> bytes:
    pptx = importlib.import_module("pptx")
    util = importlib.import_module("pptx.util")
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    text_box = slide.shapes.add_textbox(
        util.Inches(1),
        util.Inches(1),
        util.Inches(8),
        util.Inches(2),
    )
    text_box.text_frame.text = "界" * 600
    slide.shapes.add_picture(
        BytesIO(_png_bytes()),
        util.Inches(1),
        util.Inches(4),
        width=util.Inches(1),
    )
    buffer = BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


def _pptx_with_picture_near_second_text_box_bytes() -> bytes:
    pptx = importlib.import_module("pptx")
    util = importlib.import_module("pptx.util")
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    if slide.shapes.title is not None:
        slide.shapes.title.text = "Geometry Anchor Review"
    first = slide.shapes.add_textbox(
        util.Inches(0.5),
        util.Inches(1.2),
        util.Inches(4),
        util.Inches(2),
    )
    first.text_frame.text = "first-box " * 350
    second = slide.shapes.add_textbox(
        util.Inches(0.5),
        util.Inches(4.2),
        util.Inches(6),
        util.Inches(2),
    )
    second.text_frame.text = (
        ("second-prefix " * 140) + "IMAGE-NEAR-SECOND " + ("second-suffix " * 140)
    )
    slide.shapes.add_picture(
        BytesIO(_png_bytes()),
        util.Inches(7),
        util.Inches(4.4),
        width=util.Inches(1),
    )
    buffer = BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


def _pptx_with_picture_near_multi_paragraph_text_box_bytes() -> bytes:
    pptx = importlib.import_module("pptx")
    util = importlib.import_module("pptx.util")
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    text_box = slide.shapes.add_textbox(
        util.Inches(0.5),
        util.Inches(1.5),
        util.Inches(6),
        util.Inches(3),
    )
    text_box.text_frame.paragraphs[0].text = (
        ("paragraph-before " * 56)
        + "IMAGE-NEAR-MULTI-PARAGRAPH "
        + ("paragraph-before " * 4)
    ).strip()
    second = text_box.text_frame.add_paragraph()
    second.text = ("paragraph-after " * 60).strip()
    slide.shapes.add_picture(
        BytesIO(_png_bytes()),
        util.Inches(7),
        util.Inches(2.3),
        width=util.Inches(1),
    )
    buffer = BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


def _pptx_with_picture_tied_between_text_boxes_bytes() -> bytes:
    pptx = importlib.import_module("pptx")
    util = importlib.import_module("pptx.util")
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    for left, marker in ((1, "LEFT-TIE"), (5, "RIGHT-TIE")):
        text_box = slide.shapes.add_textbox(
            util.Inches(left),
            util.Inches(1),
            util.Inches(2),
            util.Inches(2),
        )
        text_box.text_frame.text = marker
    slide.shapes.add_picture(
        BytesIO(_png_bytes()),
        util.Inches(3.5),
        util.Inches(1.5),
        width=util.Inches(1),
    )
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


def test_code_line_locators_survive_index_payload_and_context_pack() -> None:
    async def go() -> tuple[PreparedDocument, Context]:
        prepared = await prepare_document_bytes(
            file_bytes=_code_bytes(),
            filename="answer.py",
            mime_type="text/x-python",
            path=None,
            ocr_provider=None,
        )
        store = RecordingVectorStore()
        indexer = DocumentIndexer(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(include_extra_channel=False),
            vector_store=store,
        )
        await indexer.index_document(
            IndexRequest(
                document_id="answer.py",
                collection="code-fixtures",
                namespace="fixture",
                text=prepared.markdown,
                filename="answer.py",
                mime_type="text/x-python",
                source_type="file",
                document_key="file:answer.py",
                content_sha256="sha256:answer.py",
                processing_version="test-code-context",
                document_metadata=prepared.metadata,
                pre_chunked_texts=[chunk.text for chunk in prepared.chunks],
                embedding_chunk_texts=[
                    chunk.embedding_text for chunk in prepared.chunks
                ],
                chunk_metadata=[dict(chunk.metadata) for chunk in prepared.chunks],
                prepared_chunks=list(prepared.chunks),
            )
        )
        [points] = store.upsert_calls
        results = [
            payload_to_result(point_id=point.id, payload=point.payload, score=0.9)
            for point in points
        ]
        return prepared, build_context_pack(results, query="answer")

    prepared, pack = asyncio.run(go())

    assert prepared.metadata["parser"] == "local:code"
    assert prepared.metadata["language"] == "python"
    [chunk] = prepared.chunks
    assert chunk.metadata["line_start"] == 1
    assert chunk.metadata["line_end"] == 2
    [snippet] = pack.snippets
    assert snippet.locator.line_start == 1
    assert snippet.locator.line_end == 2
    assert "lines 1-2" in snippet.header
    assert "lines 1-2" in snippet.prompt_header
    assert pack.source_previews[0].locator_label == "lines 1-2, chunk 0"
    assert pack.prompt_source_previews[0].locator_label == "lines 1-2, chunk 0"


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


def test_prepare_document_bytes_rejects_ocr_required_document_without_provider() -> (
    None
):
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
    assert all(chunk.metadata.get("figure_id") is None for chunk in prepared.chunks)
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


def test_true_inline_docx_figure_anchors_to_surrounding_text_chunk() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_docx_with_true_inline_image_bytes(),
            filename="inline.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            path=None,
            ocr_provider=None,
        )
    )

    [figure_chunk] = [
        chunk
        for chunk in prepared.chunks
        if chunk.metadata.get("figure_id") == "fig:docx:1"
    ]

    assert figure_chunk.chunk_index == 0
    assert "INLINE-BEFORE" in figure_chunk.text
    assert "INLINE-AFTER" in figure_chunk.text


def test_docx_table_inline_figure_anchors_to_neighboring_table_text() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_docx_with_table_inline_image_bytes(),
            filename="table-inline.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            path=None,
            ocr_provider=None,
            chunking_config=ChunkingConfig(max_chars=400, overlap=200),
        )
    )

    [figure_chunk] = [
        chunk
        for chunk in prepared.chunks
        if chunk.metadata.get("figure_id") == "fig:docx:1"
    ]

    assert "TABLE-BEFORE" in figure_chunk.text
    assert "TABLE-AFTER" in figure_chunk.text


def test_pptx_unlabelled_picture_anchors_to_unique_nearby_text_box() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_pptx_with_picture_near_second_text_box_bytes(),
            filename="geometry.pptx",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            path=None,
            ocr_provider=None,
            chunking_config=ChunkingConfig(max_chars=400, overlap=200),
        )
    )

    [figure_chunk] = [
        chunk
        for chunk in prepared.chunks
        if chunk.metadata.get("figure_id") == "fig:slide:1:1"
    ]

    assert "IMAGE-NEAR-SECOND" in figure_chunk.text
    assert "Geometry Anchor Review" not in figure_chunk.text


def test_pptx_unlabelled_picture_uses_one_anchor_per_multi_paragraph_shape() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_pptx_with_picture_near_multi_paragraph_text_box_bytes(),
            filename="multi-paragraph-anchor.pptx",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            path=None,
            ocr_provider=None,
            chunking_config=ChunkingConfig(max_chars=400, overlap=200),
        )
    )

    [figure_chunk] = [
        chunk
        for chunk in prepared.chunks
        if chunk.metadata.get("figure_id") == "fig:slide:1:1"
    ]

    assert "IMAGE-NEAR-MULTI-PARAGRAPH" in figure_chunk.text


def test_pptx_unlabelled_picture_omits_locator_for_distinct_shape_tie() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_pptx_with_picture_tied_between_text_boxes_bytes(),
            filename="ambiguous-anchor.pptx",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            path=None,
            ocr_provider=None,
        )
    )

    assert all(chunk.metadata.get("figure_id") is None for chunk in prepared.chunks)


def test_docx_many_inline_figure_anchors_remain_near_linear() -> None:
    payload = _docx_with_many_inline_images_bytes(1_600)

    started = perf_counter()
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=payload,
            filename="many-inline.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    )
    elapsed = perf_counter() - started

    assert metadata["figure_count"] == 1_600
    assert markdown.count("before  after") == 1_600
    assert elapsed < 2.5


@pytest.mark.parametrize("overlap", [200, 0], ids=["default-overlap", "zero-overlap"])
@pytest.mark.parametrize(
    ("payload", "filename", "mime_type", "figure_id"),
    [
        (
            _docx_with_long_text_and_image_bytes,
            "long.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "fig:docx:1",
        ),
        (
            _pptx_with_long_text_and_image_bytes,
            "long.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "fig:slide:1:1",
        ),
    ],
    ids=["docx", "pptx"],
)
def test_long_office_text_assigns_one_figure_to_one_chunk(
    overlap: int,
    payload: BytesFactory,
    filename: str,
    mime_type: str,
    figure_id: str,
) -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=payload(),
            filename=filename,
            mime_type=mime_type,
            path=None,
            ocr_provider=None,
            chunking_config=ChunkingConfig(max_chars=400, overlap=overlap),
        )
    )

    assert len(prepared.chunks) >= 6
    assert (
        sum(chunk.metadata.get("figure_id") == figure_id for chunk in prepared.chunks)
        == 1
    )


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


def test_large_xlsx_uses_bounded_row_windows_with_provenance() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_xlsx_large_window_bytes(),
            filename="inventory.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            path=None,
            ocr_provider=None,
        )
    )

    assert 2 <= len(prepared.chunks) <= 12
    assert all(
        chunk.text.startswith("## Sheet: Supplier Inventory")
        for chunk in prepared.chunks
    )
    assert all(
        "| record | state | inspection |" in chunk.text for chunk in prepared.chunks
    )
    assert all(
        chunk.token_count <= token_budget_for_char_limit(2_000)
        for chunk in prepared.chunks
    )
    [gold_chunk] = [
        chunk for chunk in prepared.chunks if "supplier lot KQ-733" in chunk.text
    ]
    start_row, end_row = (
        int(value) for value in str(gold_chunk.metadata["row_range"]).split("-")
    )
    assert start_row <= 300 <= end_row
    assert gold_chunk.metadata["sheet_name"] == "Supplier Inventory"
    total_tokens = sum(chunk.token_count for chunk in prepared.chunks)
    repeated_context_tokens = sum(
        value
        for chunk in prepared.chunks[1:]
        if isinstance(
            value := chunk.metadata.get("xlsx_context_token_count"),
            int,
        )
        and not isinstance(value, bool)
    )
    assert repeated_context_tokens / total_tokens <= 0.10


def test_oversized_xlsx_row_uses_labelled_budgeted_fragments() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_xlsx_oversized_row_bytes(),
            filename="oversized.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            path=None,
            ocr_provider=None,
            chunking_config=ChunkingConfig(max_chars=400, overlap=200),
        )
    )

    assert len(prepared.chunks) > 2
    assert all(chunk.token_count <= 100 for chunk in prepared.chunks)
    assert all(
        chunk.text.startswith("## Sheet: Oversized")
        and "| record | detail |" in chunk.text
        and chunk.metadata.get("row_fragment")
        for chunk in prepared.chunks
    )
    assert all(
        (lambda bounds: bounds[0] <= 2 <= bounds[1])(
            tuple(int(value) for value in str(chunk.metadata["row_range"]).split("-"))
        )
        for chunk in prepared.chunks
    )
    assert sum("OVERSIZED-ROW-NEEDLE" in chunk.text for chunk in prepared.chunks) == 1


@pytest.mark.parametrize("include_row", [False, True], ids=["header-only", "with-row"])
def test_oversized_xlsx_header_keeps_every_chunk_within_budget(
    include_row: bool,
) -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_xlsx_oversized_header_bytes(include_row=include_row),
            filename="wide-header.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            path=None,
            ocr_provider=None,
            chunking_config=ChunkingConfig(max_chars=400, overlap=200),
        )
    )

    assert prepared.chunks
    assert all(len(chunk.text) <= 400 for chunk in prepared.chunks)
    assert all(chunk.token_count <= 100 for chunk in prepared.chunks)
    assert all(
        chunk.metadata.get("sheet_name") == "Wide Header" for chunk in prepared.chunks
    )
    assert all(chunk.metadata.get("row_range") for chunk in prepared.chunks)
    header_chunks = [
        chunk for chunk in prepared.chunks if chunk.metadata.get("header_fragment")
    ]
    assert header_chunks
    assert all(chunk.metadata["row_range"] == "1-1" for chunk in header_chunks)
    header_payloads = [
        chunk.text.split("Header fragment ", 1)[1].split(": ", 1)[1]
        for chunk in header_chunks
    ]
    header_stream = "".join(header_payloads)
    expected_headers = [f"HEADER-{index}-" + ("界" * 100) for index in range(24)]
    assert header_stream == " | ".join(expected_headers)
    for index in range(24):
        marker = f"HEADER-{index}-"
        assert any(marker in payload for payload in header_payloads)
        assert header_stream.count(f"HEADER-{index}-") == 1
    assert header_stream.count("界") == 2_400
    if include_row:
        [row_chunk] = [
            chunk for chunk in prepared.chunks if "HEADER-ROW-PAYLOAD" in chunk.text
        ]
        assert row_chunk.metadata["row_range"] == "1-2"
        assert max(chunk.chunk_index for chunk in header_chunks) < row_chunk.chunk_index
    else:
        assert all(chunk.metadata.get("header_fragment") for chunk in prepared.chunks)


@pytest.mark.parametrize(
    ("max_rows", "expected_ranges"),
    [
        (500, ["1-6"]),
        (2, ["1-2", "3-4", "5-6"]),
    ],
    ids=["single-window", "multiple-windows"],
)
def test_xlsx_literal_row_suffix_sheet_name_survives_windows(
    max_rows: int,
    expected_ranges: list[str],
) -> None:
    result = asyncio.run(
        XlsxConverter(max_rows=max_rows).convert(
            _xlsx_literal_row_suffix_sheet_bytes(),
            filename="literal-sheet-name.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    )

    chunks = prepare_text_chunks(
        result.content,
        filename="literal-sheet-name.xlsx",
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    assert "## Sheet: Data (Rows 1-2)" in result.content
    assert all(
        chunk.metadata.get("sheet_name") == "Data (Rows 1-2)" for chunk in chunks
    )
    assert all(
        chunk.metadata.get("section_path") == "Sheet: Data (Rows 1-2)"
        for chunk in chunks
    )
    assert [chunk.metadata.get("row_range") for chunk in chunks] == expected_ranges
    assert all(
        "rag-core-xlsx-sheet" not in chunk.text
        and "rag-core-xlsx-sheet" not in chunk.embedding_text
        for chunk in chunks
    )


def test_multiline_xlsx_cells_keep_row_provenance_without_marker_leakage() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_xlsx_multiline_cells_bytes(),
            filename="multiline.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            path=None,
            ocr_provider=None,
        )
    )

    assert "Head A Head B" in prepared.markdown
    assert "ROW-2 CONTINUED" in prepared.markdown
    assert all("rag-core-xlsx-row" not in chunk.text for chunk in prepared.chunks)
    [row_chunk] = [
        chunk for chunk in prepared.chunks if "ROW-2 CONTINUED" in chunk.text
    ]
    assert row_chunk.metadata["row_range"] == "1-2"
    assert row_chunk.metadata["sheet_name"] == "Multiline"


def test_xlsx_fragment_labels_stay_budgeted_after_ten_thousand_fragments() -> None:
    normalized = (
        "## Sheet: Dense\n\n"
        "| record | detail | <!-- rag-core-xlsx-row:1 -->\n"
        "| --- | --- |\n"
        f"| key | {'界' * 800_000} | <!-- rag-core-xlsx-row:2 -->"
    )

    chunks = XlsxChunker().chunk(
        normalized,
        ChunkConfig(max_chars=400, overlap=200),
    )

    assert len(chunks) > 10_000
    assert max(len(chunk.text) for chunk in chunks) <= 400
    assert max(chunk.token_count for chunk in chunks) <= 100
    assert chunks[-1].metadata["row_fragment"] == f"{len(chunks)}/{len(chunks)}"


def test_xlsx_semantic_header_accumulation_tokenizes_linear_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "## Sheet: Wide (Rows 1-1)\nHeader fragment 99999/99999: "
    payload = " | ".join(f"COLUMN-{index:05d}" for index in range(2_000))
    units = xlsx_fragments_module.header_units(payload)
    examined_chars = 0

    def counted_estimate(text: str) -> int:
        nonlocal examined_chars
        examined_chars += len(text)
        return estimate_token_count(text)

    monkeypatch.setattr(
        xlsx_fragments_module,
        "estimate_token_count",
        counted_estimate,
        raising=False,
    )
    monkeypatch.setattr(
        chunk_budget_module,
        "estimate_token_count",
        counted_estimate,
    )

    fragments = xlsx_fragments_module.split_semantic_payload(
        units,
        prefix=prefix,
        max_chars=1_000_000,
    )

    assert fragments == [payload]
    assert examined_chars <= 3 * (len(prefix) + len(payload))


def test_xlsx_wide_header_scales_near_linearly() -> None:
    chunker = XlsxChunker()
    config = ChunkConfig(max_chars=1_000_000, overlap=200)
    small = _normalized_wide_xlsx_header(1_000)
    large = _normalized_wide_xlsx_header(5_000)
    chunker.chunk(_normalized_wide_xlsx_header(10), config)

    started = perf_counter()
    small_chunks = chunker.chunk(small, config)
    small_elapsed = perf_counter() - started
    started = perf_counter()
    large_chunks = chunker.chunk(large, config)
    large_elapsed = perf_counter() - started

    assert len(small_chunks) == len(large_chunks) == 1
    [large_chunk] = large_chunks
    payload = large_chunk.text.split("Header fragment ", 1)[1].split(": ", 1)[1]
    assert payload == " | ".join(f"COLUMN-{index:05d}" for index in range(5_000))
    assert len(large_chunk.text) <= config.max_chars
    assert large_chunk.token_count <= token_budget_for_char_limit(config.max_chars)
    assert large_elapsed < 1.0
    assert large_elapsed / max(small_elapsed, 0.001) < 8.0


def test_xlsx_large_configured_window_scales_near_linearly() -> None:
    chunker = XlsxChunker()
    config = ChunkConfig(max_chars=1_000_000, overlap=200)
    small = _normalized_xlsx_rows(1_000)
    large = _normalized_xlsx_rows(5_000)
    chunker.chunk(_normalized_xlsx_rows(10), config)

    started = perf_counter()
    small_chunks = chunker.chunk(small, config)
    small_elapsed = perf_counter() - started
    started = perf_counter()
    large_chunks = chunker.chunk(large, config)
    large_elapsed = perf_counter() - started

    assert small_chunks and large_chunks
    assert large_elapsed < 2.0
    assert large_elapsed / max(small_elapsed, 0.001) < 7.5


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


def test_default_pdf_inspector_route_preserves_two_page_locators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pdf_inspector", reason="requires the optional pdf extra")
    monkeypatch.delenv("PDF_INSPECTOR_MODE", raising=False)

    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_two_page_pdf_bytes(),
            filename="two-page.pdf",
            mime_type="application/pdf",
            path=None,
            ocr_provider=None,
            chunking_config=ChunkingConfig(max_chars=220, overlap=0),
        )
    )

    assert prepared.metadata["parser"] == "local:pdf_inspector"
    assert prepared.metadata["page_count"] == 2
    assert prepared.markdown.count("## Page 1") == 1
    assert prepared.markdown.count("## Page 2") == 1
    page_one = next(
        chunk for chunk in prepared.chunks if "PAGE-ONE-EVIDENCE" in chunk.text
    )
    page_two = next(
        chunk for chunk in prepared.chunks if "PAGE-TWO-EVIDENCE" in chunk.text
    )
    assert (page_one.metadata["page_number"], page_one.metadata["page_index"]) == (
        1,
        0,
    )
    assert (page_two.metadata["page_number"], page_two.metadata["page_index"]) == (
        2,
        1,
    )


@pytest.mark.parametrize(
    "sheet_name",
    [
        "Dashboard",
        "Dashboard (Rows 1-2)",
        "仪表板 (Rows 1-2)",
    ],
)
def test_xlsx_chart_only_sheet_preserves_scope_and_figure_locator(
    sheet_name: str,
) -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_xlsx_with_chart_only_sheet_bytes(
                chart_sheet_title=sheet_name,
            ),
            filename="dashboard.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            path=None,
            ocr_provider=None,
        )
    )

    chart_chunk = next(
        chunk
        for chunk in prepared.chunks
        if chunk.metadata.get("figure_id") == "fig:sheet:2:chart:1"
    )

    assert f"## Sheet: {sheet_name}" in prepared.markdown
    assert "Quality Trend" in chart_chunk.text
    assert chart_chunk.metadata["sheet_name"] == sheet_name
    assert chart_chunk.metadata["section_path"] == f"Sheet: {sheet_name}"
    assert chart_chunk.metadata["section_title"] == f"Sheet: {sheet_name}"
    assert chart_chunk.metadata.get("row_range") is None
    assert chart_chunk.start_char is None
    assert chart_chunk.end_char is None
    assert chart_chunk.metadata["offset_reconstruction"] == "unreliable"
    assert "rag-core-xlsx-sheet" not in chart_chunk.text
    assert "rag-core-xlsx-sheet" not in chart_chunk.embedding_text


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


def test_pymupdf_block_heuristic_reads_clean_two_columns_column_first() -> None:
    class TwoColumnPage:
        def get_text(self, mode: str, *, sort: bool = False) -> object:
            assert mode == "blocks"
            assert sort is False
            return [
                (320, 80, 520, 110, "R1", 1, 0),
                (60, 80, 260, 110, "L1", 0, 0),
                (320, 130, 520, 160, "R2", 3, 0),
                (60, 130, 260, 160, "L2", 2, 0),
            ]

        def get_images(self) -> list[object]:
            return []

    assert _extract_page_layout_text(TwoColumnPage()) == "L1\n\nL2\n\nR1\n\nR2"


def test_pymupdf_block_heuristic_keeps_title_before_column_first_body() -> None:
    class TitledTwoColumnPage:
        def get_text(self, mode: str, *, sort: bool = False) -> object:
            assert mode == "blocks"
            assert sort is False
            return [
                (60, 20, 520, 50, "TITLE", 0, 0),
                (320, 80, 520, 110, "R1", 2, 0),
                (60, 80, 260, 110, "L1", 1, 0),
                (320, 130, 520, 160, "R2", 4, 0),
                (60, 130, 260, 160, "L2", 3, 0),
            ]

        def get_images(self) -> list[object]:
            return []

    assert (
        _extract_page_layout_text(TitledTwoColumnPage())
        == "TITLE\n\nL1\n\nL2\n\nR1\n\nR2"
    )


def test_pymupdf_real_blocks_keep_full_width_title_before_columns() -> None:
    fitz = importlib.import_module("fitz")
    document = fitz.open()
    page = document.new_page(width=580, height=300)
    page.insert_text(
        (60, 40),
        "FULL WIDTH TITLE " + ("WIDE " * 12),
        fontsize=8,
    )
    page.insert_text((60, 90), "L1")
    page.insert_text((60, 140), "L2")
    page.insert_text((320, 90), "R1")
    page.insert_text((320, 140), "R2")

    extracted = _extract_page_layout_text(page)
    document.close()

    assert extracted.index("FULL WIDTH TITLE") < extracted.index("L1")
    assert extracted.index("L1") < extracted.index("L2")
    assert extracted.index("L2") < extracted.index("R1")
    assert extracted.index("R1") < extracted.index("R2")


def test_pymupdf_block_ordering_scales_to_large_layouts() -> None:
    blocks: list[tuple[float, float, float, float, str, int]] = []
    for index in range(16_000):
        y = 80.0 + index * 12.0
        blocks.append((60.0, y, 260.0, y + 10.0, f"L{index}", index * 2))
        blocks.append((320.0, y, 520.0, y + 10.0, f"R{index}", index * 2 + 1))

    started = perf_counter()
    ordered = _order_text_blocks(blocks)
    elapsed = perf_counter() - started

    assert ordered[0][4] == "L0"
    assert ordered[15_999][4] == "L15999"
    assert ordered[16_000][4] == "R0"
    assert elapsed < 2.0


@pytest.mark.integration
def test_real_pdf_parse_metadata_survives_prepare_and_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = (
        "Retrieval PDF quality parser fixture proves page metadata survives "
        "prepare and ingest with enough text for quality scoring."
    )
    file_bytes = _minimal_pdf_bytes(text)
    monkeypatch.setattr(pdf_converter_module, "pdf_inspector_enabled", lambda: False)

    async def go() -> tuple[PreparedDocument, IngestedDocument, RecordingVectorStore]:
        prepared = await prepare_document_bytes(
            file_bytes=file_bytes,
            filename="quality-proof.pdf",
            mime_type="application/pdf",
            path="/fixtures/quality-proof.pdf",
            ocr_provider=None,
        )
        store = RecordingVectorStore()
        core = Engine(
            make_test_config(embedding_dimensions=4),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        try:
            ingested = await core.add_bytes(
                file_bytes=file_bytes,
                filename="quality-proof.pdf",
                mime_type="application/pdf",
                namespace="parsers",
                collection="fixtures",
                document_id="real-pdf",
                document_key="quality-proof.pdf",
            )
        finally:
            await core.close()
        return prepared, ingested, store

    prepared, ingested, store = asyncio.run(go())

    assert prepared.metadata["parser"] == "local:pymupdf"
    assert prepared.metadata["needs_ocr"] is False
    assert prepared.metadata["page_count"] == 1
    assert_quality_metadata(prepared.metadata)

    assert ingested.metadata["parser"] == prepared.metadata["parser"]
    assert ingested.metadata["needs_ocr"] is False
    assert ingested.metadata["page_count"] == prepared.metadata["page_count"]
    assert ingested.metadata["quality"] == prepared.metadata["quality"]
    assert ingested.chunk_count >= 1

    indexed_payloads = [point.payload for call in store.upsert_calls for point in call]
    assert indexed_payloads
    assert any(payload.get("page_index") == 0 for payload in indexed_payloads)
    assert any(
        payload.get("quality_verdict") == prepared.metadata["quality"]["verdict"]
        for payload in indexed_payloads
    )
    assert any(
        payload.get("quality_char_count") == prepared.metadata["quality"]["char_count"]
        for payload in indexed_payloads
    )


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
                collection="docs",
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
                collection="docs",
            )
        )
    )

    assert preview.document.ocr.needed is True
    assert preview.document.chunk_count == 0
    assert preview.manifest_entry.needs_ocr is True
    assert preview.manifest_entry.metadata["parser"] == "ocr_required"


def test_single_private_use_glyph_routes_pdf_page_to_ocr() -> None:
    import logging

    from rag_core.documents.converters.pdf_converter_extraction import (
        PdfExtraction,
        _extract_page,
    )
    from rag_core.documents.converters.pdf_converter_pymupdf import (
        pymupdf_conversion_result,
    )

    pua = "\ue000"
    raw = ("Retrieval quality baseline text for parser regression. " * 2) + pua

    class _Page:
        def get_text(self, mode: str) -> str:
            return raw

        def get_images(self) -> list[object]:
            return []

    page = _extract_page(_Page(), 0)
    assert page.needs_ocr is True
    assert page.has_garbled_text is True
    result = pymupdf_conversion_result(
        PdfExtraction(pages=[page], page_count=1),
        logger=logging.getLogger("test"),
    )
    assert result.needs_ocr is True


def test_pymupdf_page_assembly_escapes_source_page_heading_collision() -> None:
    import logging

    from rag_core.documents.converters.pdf_converter_pymupdf import (
        pymupdf_conversion_result,
    )

    result = pymupdf_conversion_result(
        PdfExtraction(
            pages=[
                PageExtraction(
                    page_num=0,
                    text=(
                        "Page 2\n"
                        "PYMUPDF-PAGE-ONE collision-adjacent evidence remains first."
                    ),
                    char_count=68,
                ),
                PageExtraction(
                    page_num=1,
                    text="PYMUPDF-PAGE-TWO belongs to the true second page.",
                    char_count=50,
                ),
            ],
            page_count=2,
        ),
        logger=logging.getLogger("test"),
    )
    chunks = prepare_text_chunks(
        result.content,
        filename="pymupdf-headings.pdf",
        mime_type="application/pdf",
        chunking_config=ChunkingConfig(max_chars=160, overlap=0),
    )

    page_one = next(chunk for chunk in chunks if "PYMUPDF-PAGE-ONE" in chunk.text)
    page_two = next(chunk for chunk in chunks if "PYMUPDF-PAGE-TWO" in chunk.text)
    assert page_one.metadata["page_number"] == 1
    assert page_one.metadata["page_index"] == 0
    assert page_one.metadata.get("section_title") != "Page 2"
    assert "Page 2" not in str(page_one.metadata.get("section_path", "")).split(" > ")
    assert page_two.metadata["page_number"] == 2
    assert page_two.metadata["page_index"] == 1
    assert r"\#\# Page 2" in result.content
    assert result.content.splitlines().count("## Page 2") == 1


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
