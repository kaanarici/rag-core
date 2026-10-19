from __future__ import annotations

import asyncio
import logging
import sys
from types import SimpleNamespace
from typing import cast

import pytest

import rag_core.documents.converters.pdf_converter_extraction as pdf_extraction_module
from rag_core._engine.core_ocr_metadata import read_ocr_metadata
from rag_core._engine.core_prepare import (
    apply_ocr,
    prepare_text_chunks,
)
from rag_core.config import ChunkingConfig
from rag_core.core_models import (
    ParsedDocument,
    PreparedDocument,
)
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
        IndexRequest(document_id="headings.pdf", collection="pdf-fixtures", tenant_id="fixture", text=prepared.markdown, filename="headings.pdf", mime_type="application/pdf", source_type="file", document_key="file:headings.pdf", content_sha256="sha256:headings.pdf", processing_version="test-pdf-context", document_metadata=prepared.metadata, prepared_chunks=list(prepared.chunks))
    )
    return [
        payload_to_result(point_id=point.id, payload=point.payload, score=0.9)
        for points in store.upsert_calls
        for point in points
    ]
































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
