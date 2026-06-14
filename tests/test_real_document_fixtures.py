from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from rag_core._engine.core_prepare import prepare_document_bytes
from rag_core.documents.local_parse import parse_file_bytes
from rag_core.search.context_pack import ContextPack, build_context_pack
from rag_core.search.indexer import DocumentIndexer, IndexRequest
from rag_core.search.stored_payload import payload_to_result

from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder, RecordingVectorStore


_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "real_documents" / "apache_tika"


@dataclass(frozen=True)
class RealDocumentCase:
    filename: str
    mime_type: str
    expected_text: tuple[str, ...]
    parser_names: frozenset[str]
    expected_metadata: dict[str, object]


REAL_DOCUMENT_CASES = (
    RealDocumentCase(
        filename="testPDF.pdf",
        mime_type="application/pdf",
        expected_text=("Apache Tika", "Content Analysis Toolkit"),
        parser_names=frozenset({"local:pdf_inspector", "local:pymupdf"}),
        expected_metadata={"needs_ocr": False},
    ),
    RealDocumentCase(
        filename="testWORD.docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        expected_text=("Sample Word Document Title", "The table has things in it"),
        parser_names=frozenset({"local:python-docx"}),
        expected_metadata={"needs_ocr": False},
    ),
    RealDocumentCase(
        filename="testPPT.pptx",
        mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        expected_text=("## Slide 2", "Watershed", "Avalanche"),
        parser_names=frozenset({"local:python-pptx"}),
        expected_metadata={"needs_ocr": False, "slide_count": 3},
    ),
    RealDocumentCase(
        filename="testEXCEL.xlsx",
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        expected_text=("## Sheet: Feuil1", "Numbers and their Squares"),
        parser_names=frozenset({"local:openpyxl"}),
        expected_metadata={"needs_ocr": False, "sheet_count": 3},
    ),
)


@pytest.mark.parametrize(
    "case",
    REAL_DOCUMENT_CASES,
    ids=[case.filename for case in REAL_DOCUMENT_CASES],
)
def test_external_real_documents_parse_with_quality_metadata(
    case: RealDocumentCase,
) -> None:
    markdown, metadata = asyncio.run(
        parse_file_bytes(
            file_bytes=(_FIXTURE_DIR / case.filename).read_bytes(),
            filename=case.filename,
            mime_type=case.mime_type,
        )
    )

    assert markdown.strip()
    for text in case.expected_text:
        assert text in markdown
    assert metadata["parser"] in case.parser_names
    for key, value in case.expected_metadata.items():
        assert metadata[key] == value
    quality = metadata["quality"]
    assert isinstance(quality, dict)
    assert quality["verdict"] == "good"
    assert quality["char_count"] > 0


def test_external_pptx_preserves_slide_locators() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=(_FIXTURE_DIR / "testPPT.pptx").read_bytes(),
            filename="testPPT.pptx",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            path=None,
            ocr_provider=None,
        )
    )

    slide_chunks = {
        chunk.metadata.get("slide_number"): chunk
        for chunk in prepared.chunks
        if "slide_number" in chunk.metadata
    }
    assert set(slide_chunks) == {1, 2, 3}
    assert "Watershed" in slide_chunks[3].embedding_text


def test_external_xlsx_preserves_sheet_locators() -> None:
    prepared = asyncio.run(
        prepare_document_bytes(
            file_bytes=(_FIXTURE_DIR / "testEXCEL.xlsx").read_bytes(),
            filename="testEXCEL.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            path=None,
            ocr_provider=None,
        )
    )

    assert len(prepared.chunks) == 1
    [chunk] = prepared.chunks
    assert chunk.metadata["sheet_name"] == "Feuil1"
    assert "Numbers and their Squares" in chunk.embedding_text


def test_external_pptx_locators_survive_index_payload_and_context_pack() -> None:
    pack = asyncio.run(
        _build_real_document_context_pack(
            filename="testPPT.pptx",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            query="Watershed",
        )
    )

    watershed_snippet = next(
        snippet for snippet in pack.snippets if "Watershed" in snippet.text
    )
    assert watershed_snippet.locator.slide_number == 3
    assert watershed_snippet.source.title == "testPPT.pptx"
    assert "Slide 3" in watershed_snippet.header
    assert "Slide 3" in watershed_snippet.prompt_header
    assert pack.source_previews[watershed_snippet.rank - 1].locator_label == (
        "Slide 3, chunk 3"
    )
    assert pack.prompt_source_previews[watershed_snippet.rank - 1].locator_label == (
        "Slide 3, chunk 3"
    )


def test_external_xlsx_locators_survive_index_payload_and_context_pack() -> None:
    pack = asyncio.run(
        _build_real_document_context_pack(
            filename="testEXCEL.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            query="Numbers and their Squares",
        )
    )

    [snippet] = pack.snippets
    assert snippet.locator.sheet_name == "Feuil1"
    assert snippet.source.title == "testEXCEL.xlsx"
    assert "Sheet: Feuil1" in snippet.header
    assert "Sheet: Feuil1" in snippet.prompt_header
    assert pack.source_previews[0].locator_label == "Sheet: Feuil1, chunk 0"
    assert pack.prompt_source_previews[0].locator_label == "Sheet: Feuil1, chunk 0"


def test_external_pdf_sections_survive_index_payload_and_context_pack() -> None:
    pack = asyncio.run(
        _build_real_document_context_pack(
            filename="testPDF.pdf",
            mime_type="application/pdf",
            query="Content Analysis Toolkit",
        )
    )

    toolkit_snippet = next(
        snippet for snippet in pack.snippets if "Content Analysis Toolkit" in snippet.text
    )
    assert toolkit_snippet.locator.section_path is not None
    section_path = toolkit_snippet.locator.section_path.replace(" - ", "-")
    assert section_path == "Tika-Content Analysis Toolkit"
    assert toolkit_snippet.source.title == "testPDF.pdf"
    assert "Tika" in toolkit_snippet.header
    assert "Content Analysis Toolkit" in toolkit_snippet.header
    assert "Tika" in toolkit_snippet.prompt_header
    assert "Content Analysis Toolkit" in toolkit_snippet.prompt_header
    source_label = pack.source_previews[toolkit_snippet.rank - 1].locator_label
    prompt_source_label = pack.prompt_source_previews[toolkit_snippet.rank - 1].locator_label
    assert source_label is not None
    assert prompt_source_label is not None
    normalized_source_label = source_label.replace(" - ", "-")
    normalized_prompt_source_label = prompt_source_label.replace(" - ", "-")
    assert "Tika-Content Analysis Toolkit" in normalized_source_label
    assert "chunk" in normalized_source_label
    assert "Tika-Content Analysis Toolkit" in normalized_prompt_source_label
    assert "chunk" in normalized_prompt_source_label


async def _build_real_document_context_pack(
    *,
    filename: str,
    mime_type: str,
    query: str,
) -> ContextPack:
    prepared = await prepare_document_bytes(
        file_bytes=(_FIXTURE_DIR / filename).read_bytes(),
        filename=filename,
        mime_type=mime_type,
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
            document_id=filename,
            corpus_id="real-documents",
            namespace="fixture",
            text=prepared.markdown,
            filename=filename,
            mime_type=mime_type,
            source_type="file",
            document_key=f"file:{filename}",
            content_sha256=f"sha256:{filename}",
            processing_version="test-real-document-context",
            document_metadata=prepared.metadata,
            pre_chunked_texts=[chunk.text for chunk in prepared.chunks],
            embedding_chunk_texts=[chunk.embedding_text for chunk in prepared.chunks],
            chunk_metadata=[dict(chunk.metadata) for chunk in prepared.chunks],
            prepared_chunks=list(prepared.chunks),
        )
    )

    [points] = store.upsert_calls
    results = [
        payload_to_result(point_id=point.id, payload=point.payload, score=0.9)
        for point in points
    ]
    return build_context_pack(results, query=query)
