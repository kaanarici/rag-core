"""Generated Office/CSV fixtures and real OCR process cleanup regressions."""

from __future__ import annotations

import asyncio
import csv
import importlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic
from typing import Any

import pytest

from rag_core._engine.core_prepare import prepare_document_bytes, prepare_text_chunks
from rag_core.config import ChunkingConfig
from rag_core.documents.local_parse import parse_file_bytes
from rag_core.documents.ocr import CommandOcrProvider, OcrRequest, OcrResult

pytestmark = pytest.mark.integration

_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01"
    b"\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _save(document: Any) -> bytes:
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _prepare(payload: bytes, extension: str) -> Any:
    return asyncio.run(
        prepare_document_bytes(
            file_bytes=payload,
            filename=f"review.{extension}",
            mime_type=_DOCX if extension == "docx" else _PPTX,
            path=None,
            ocr_provider=None,
            chunking_config=ChunkingConfig(max_chars=240, overlap=20),
        )
    )


def test_docx_nested_tables_preserve_order_and_inline_figure_offsets() -> None:
    doc = importlib.import_module("docx").Document()
    doc.add_paragraph("BEFORE-BODY")
    cell = doc.add_table(rows=1, cols=1).cell(0, 0)
    cell.text = "BEFORE-OUTER"
    nested_cell = cell.add_table(rows=1, cols=1).cell(0, 0)
    nested_cell.text = "BEFORE-INNER"
    deepest = nested_cell.add_table(rows=1, cols=1).cell(0, 0)
    paragraph = deepest.paragraphs[0]
    paragraph.add_run("NESTED-BEFORE " + ("beneficiary instructions " * 20))
    shape = paragraph.add_run().add_picture(io.BytesIO(_PNG))
    shape._inline.docPr.set("descr", "Nested beneficiary diagram")
    paragraph.add_run(" NESTED-AFTER")
    nested_cell.add_paragraph("AFTER-INNER")
    cell.add_paragraph("AFTER-OUTER")
    doc.add_paragraph("AFTER-BODY")
    prepared = _prepare(_save(doc), "docx")
    markers = [
        "BEFORE-BODY",
        "BEFORE-OUTER",
        "BEFORE-INNER",
        "NESTED-BEFORE",
        "NESTED-AFTER",
        "AFTER-INNER",
        "AFTER-OUTER",
        "AFTER-BODY",
    ]
    positions = [prepared.markdown.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert all(prepared.markdown.count(marker) == 1 for marker in markers)
    assert prepared.metadata["needs_ocr"] is False
    figure = prepared.metadata["figure_items"][0]
    anchor = figure["metadata"]["text_anchor_start_char"]
    assert prepared.markdown[anchor:].startswith(" NESTED-AFTER")
    assert any(
        figure["figure_id"] == chunk.metadata.get("figure_id")
        for chunk in prepared.chunks
    )


def test_pptx_nested_groups_retain_text_and_table_on_their_slide() -> None:
    pptx = importlib.import_module("pptx")
    inch = importlib.import_module("pptx.util").Inches
    doc = pptx.Presentation()
    slide = doc.slides.add_slide(doc.slide_layouts[6])
    outer = slide.shapes.add_group_shape()
    outer.shapes.add_textbox(0, 0, inch(4), inch(1)).text = "BEFORE-GROUP"
    inner = outer.shapes.add_group_shape()
    inner.shapes.add_textbox(
        0, 0, inch(4), inch(1)
    ).text = "NESTED-INSTRUCTIONS: invoices may be paid by bank transfer."
    table = slide.shapes.add_table(1, 1, 0, inch(1), inch(4), inch(1))
    table.table.cell(0, 0).text = "NESTED-TABLE"
    inner.shapes._spTree.insert_element_before(table.element, "p:extLst")
    outer.shapes.add_textbox(0, inch(2), inch(4), inch(1)).text = "AFTER-GROUP"
    second = doc.slides.add_slide(doc.slide_layouts[6])
    second.shapes.add_textbox(0, 0, inch(4), inch(1)).text = "SECOND-SLIDE"
    prepared = _prepare(_save(doc), "pptx")
    markers = ["BEFORE-GROUP", "NESTED-INSTRUCTIONS", "NESTED-TABLE", "AFTER-GROUP"]
    positions = [prepared.markdown.index(marker) for marker in markers]
    assert positions == sorted(positions)
    for marker in [*markers, "SECOND-SLIDE"]:
        chunks = [chunk for chunk in prepared.chunks if marker in chunk.text]
        assert chunks
        assert all(
            chunk.metadata["slide_number"] == (2 if marker == "SECOND-SLIDE" else 1)
            for chunk in chunks
        )
    assert prepared.metadata["needs_ocr"] is False


def test_literal_slide_headings_cannot_forge_slide_locators() -> None:
    pptx = importlib.import_module("pptx")
    inch = importlib.import_module("pptx.util").Inches
    doc = pptx.Presentation()
    for number in (1, 2):
        slide = doc.slides.add_slide(doc.slide_layouts[6])
        slide.shapes.add_textbox(0, 0, inch(5), inch(2)).text = (
            f"## Slide 99\nSOURCE-{number}: Bank transfer is accepted.\n"
            "# Other heading\n" + (f"EVIDENCE-{number} " * 80)
        )
    prepared = _prepare(_save(doc), "pptx")
    assert prepared.metadata["slide_count"] == 2
    for chunk in prepared.chunks:
        assert prepared.markdown[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.metadata["slide_number"] in (1, 2)
        for number in (1, 2):
            if f"SOURCE-{number}" in chunk.text or f"EVIDENCE-{number}" in chunk.text:
                assert chunk.metadata["slide_number"] == number
    assert "## Slide 99" in prepared.markdown
    ordinary = prepare_text_chunks(
        "## Slide 99\n\nLiteral heading.", filename="notes.md"
    )
    assert all("slide_number" not in chunk.metadata for chunk in ordinary)


def test_ocr_replacement_cannot_reuse_native_slide_offsets() -> None:
    pptx = importlib.import_module("pptx")
    doc = pptx.Presentation()
    slide = doc.slides.add_slide(doc.slide_layouts[5])
    slide.shapes.title.text = "\ufffd" * 12

    class Replacement:
        async def extract_markdown(self, request: OcrRequest) -> OcrResult:
            return OcrResult(
                markdown="## Slide 99\n\nRecovered payment policy.",
                merge_mode="replace",
            )

    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=_save(doc),
            filename="review.pptx",
            mime_type=_PPTX,
            path=None,
            ocr_provider=Replacement(),  # type: ignore[arg-type]
        )
    )
    assert "Recovered payment policy" in prepared.markdown
    assert "slide_ranges" not in prepared.metadata
    assert all("slide_number" not in chunk.metadata for chunk in prepared.chunks)


@pytest.mark.parametrize("delimiter", [",", "\t", ";", "|"])
def test_csv_detection_respects_quoted_delimiters_and_multiline_fields(
    delimiter: str,
) -> None:
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter=delimiter)
    writer.writerow(["name", "description"])
    for number in range(12):
        writer.writerow(
            [f"Row{number}", 'one|two|three|four,semi;tab\t"quoted"\ncontinued']
        )
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=output.getvalue().encode(),
            filename="review.csv",
            mime_type="text/csv",
        )
    )
    assert metadata["delimiter"] == repr(delimiter)
    assert metadata["row_count"] == 13
    assert metadata["has_header"] is True
    assert "| name | description |" in markdown
    assert all(f"Row{number}" in markdown for number in range(12))
    assert '"quoted"\ncontinued' in markdown
    assert "Col 3" not in markdown


@pytest.mark.parametrize("extension", ["docx", "pptx"])
@pytest.mark.parametrize(
    "text, needs_ocr",
    [
        ("OK.", False),
        ("\ufffd" * 12, True),
        ("Valid\ufffd\ufffd", True),
        ("§" * 12, True),
    ],
)
def test_short_office_text_requires_clean_content_before_ocr_exemption(
    extension: str,
    text: str,
    needs_ocr: bool,
) -> None:
    if extension == "docx":
        doc = importlib.import_module("docx").Document()
        doc.add_paragraph(text)
    else:
        doc = importlib.import_module("pptx").Presentation()
        slide = doc.slides.add_slide(doc.slide_layouts[5])
        slide.shapes.title.text = text
    payload = _save(doc)
    _, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=payload,
            filename=f"review.{extension}",
            mime_type=_DOCX if extension == "docx" else _PPTX,
        )
    )
    assert metadata["needs_ocr"] is needs_ocr
    if needs_ocr:
        assert "quality_warning" not in metadata
        with pytest.raises(ValueError, match="requires OCR"):
            _prepare(payload, extension)
    else:
        assert metadata["quality_warning"] == "short_extracted_text"
        assert text in _prepare(payload, extension).markdown


def _running(pid: int) -> bool:
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
    )
    return bool(state.stdout.strip()) and not state.stdout.strip().startswith("Z")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract")
@pytest.mark.parametrize("cancel", ["timeout", "cancel", "repeated_cancel"])
def test_ocr_cleanup_kills_descendants_and_removes_input(
    tmp_path: Path, cancel: str
) -> None:
    async def run() -> None:
        marker = tmp_path / "processes.json"
        child_script = (
            "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "time.sleep(30)"
        )
        script = (
            "import json,os,signal,subprocess,sys,time; from pathlib import Path; "
            "p=json.load(sys.stdin); "
            f"child=subprocess.Popen([sys.executable,'-c',{child_script!r}]); "
            "Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),"
            "'child':child.pid,'input':p['file_path']})); time.sleep(30)"
        )
        provider = CommandOcrProvider(
            command=[sys.executable, "-c", script, str(marker)],
            timeout_seconds=0.6,
        )
        task = asyncio.create_task(
            provider.extract_markdown(
                OcrRequest(
                    file_bytes=b"ocr-input",
                    filename="scan.png",
                    mime_type="image/png",
                )
            )
        )
        started = monotonic()
        process_info: dict[str, Any] = {}
        try:
            async with asyncio.timeout(0.5):
                while not marker.exists():
                    await asyncio.sleep(0.01)
            process_info = json.loads(marker.read_text())
            assert Path(process_info["input"]).read_bytes() == b"ocr-input"
            if cancel != "timeout":
                task.cancel()
                if cancel == "repeated_cancel":
                    asyncio.get_running_loop().call_soon(task.cancel)
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(RuntimeError, match="timed out after 0.6s"):
                    await task
            assert monotonic() - started < 1.5
            assert not Path(process_info["input"]).exists()
            async with asyncio.timeout(0.5):
                while any(_running(process_info[key]) for key in ("pid", "child")):
                    await asyncio.sleep(0.01)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            for key in ("pid", "child"):
                if key in process_info:
                    try:
                        os.kill(process_info[key], signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    asyncio.run(run())
