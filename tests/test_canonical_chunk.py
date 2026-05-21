import pytest

from rag_core.core_builders import build_index_request
from rag_core.core_models import PreparedChunk, PreparedDocument, ProcessingFingerprint
from rag_core.core_prepare import prepare_pre_chunked_texts, prepare_text_chunks
from rag_core.documents.chunking.markdown import MarkdownChunker
from rag_core.documents.chunking.protocol import ChunkConfig
from rag_core.documents.chunking.router import chunk_text


def _config() -> ChunkConfig:
    return ChunkConfig(max_chars=2000, overlap=200, strategy="markdown")


def test_chunkers_and_router_return_prepared_chunks() -> None:
    """No mapping layer between chunkers/router and the canonical PreparedChunk."""
    text = "# Heading\n\nFirst paragraph here.\n\n## Subhead\n\nSecond paragraph there.\n"
    direct = MarkdownChunker().chunk(text, _config())
    routed = chunk_text("# Title\n\nBody.\n", filename="note.md")

    assert direct and routed
    for chunk in direct:
        assert isinstance(chunk, PreparedChunk)
        assert chunk.text
        assert chunk.embedding_text == chunk.text
        assert chunk.chunking_strategy == "markdown"
    assert all(isinstance(chunk, PreparedChunk) for chunk in routed)


def test_prepare_text_chunks_populates_canonical_fields() -> None:
    text = "# Heading\n\nThis is a short paragraph.\n"
    [chunk] = prepare_text_chunks(text, filename="note.md")

    assert chunk.start_char == 0
    assert 0 < chunk.end_char <= len(text)
    assert chunk.text in text
    assert chunk.chunking_strategy == "markdown"
    assert chunk.word_count > 0


def test_markdown_chunks_include_section_path_metadata() -> None:
    text = (
        "# Guide\n\n"
        "Opening paragraph.\n\n"
        "## Install\n\n"
        "Install steps.\n\n"
        "### Qdrant\n\n"
        "Qdrant setup."
    )

    chunks = MarkdownChunker().chunk(text, _config())

    assert [chunk.metadata.get("section_path") for chunk in chunks] == [
        "Guide",
        "Guide > Install",
        "Guide > Install > Qdrant",
    ]
    assert [chunk.metadata.get("section_title") for chunk in chunks] == [
        "Guide",
        "Install",
        "Qdrant",
    ]


def test_markdown_chunks_normalize_office_style_locators() -> None:
    slide_chunks = MarkdownChunker().chunk(
        "## Slide 3\n\n### Summary\n\nSlide notes.",
        _config(),
    )
    sheet_chunks = MarkdownChunker().chunk(
        "## Sheet: Signals (Rows 11-20)\n\n| Signal | Value |\n| --- | --- |\n| a | b |",
        _config(),
    )

    assert {chunk.metadata["slide_number"] for chunk in slide_chunks} == {3}
    assert sheet_chunks[0].metadata["sheet_name"] == "Signals"
    assert sheet_chunks[0].metadata["row_range"] == "11-20"


def test_prepare_text_chunks_adds_pdf_page_locators() -> None:
    text = (
        "## Page 1\n\nAlpha page content for retrieval citations.\n\n"
        "## Page 2\n\nBeta page content for retrieval citations.\n"
    )

    prepared = prepare_text_chunks(
        text,
        mime_type="application/pdf",
        filename="report.pdf",
    )

    assert len(prepared) == 2
    assert [chunk.metadata.get("page_number") for chunk in prepared] == [1, 2]
    assert [chunk.metadata.get("page_index") for chunk in prepared] == [0, 1]
    assert [chunk.metadata.get("section_path") for chunk in prepared] == [
        "Page 1",
        "Page 2",
    ]


def test_prepare_text_chunks_assigns_preface_before_first_pdf_page_heading() -> None:
    text = (
        "Lead-in text emitted before the first explicit page heading.\n\n"
        "## Page 1\n\nAlpha page content.\n\n"
        "## Page 2\n\nBeta page content.\n"
    )

    prepared = prepare_text_chunks(
        text,
        mime_type="application/pdf",
        filename="report.pdf",
    )

    assert len(prepared) == 3
    assert [chunk.metadata.get("page_number") for chunk in prepared] == [1, 1, 2]
    assert [chunk.metadata.get("page_index") for chunk in prepared] == [0, 0, 1]


def test_prepare_text_chunks_leaves_pdf_without_page_markers_unannotated() -> None:
    prepared = prepare_text_chunks(
        "Plain extracted PDF text without page headings.",
        mime_type="application/pdf",
        filename="report.pdf",
    )

    assert prepared
    assert "page_number" not in prepared[0].metadata


def test_prepare_pre_chunked_texts_yields_canonical_chunks() -> None:
    chunks = prepare_pre_chunked_texts(
        ["alpha beta", "gamma"],
        chunking_strategy="prechunked",
    )

    assert [type(chunk) for chunk in chunks] == [PreparedChunk, PreparedChunk]
    assert chunks[0].text == "alpha beta"
    assert chunks[0].word_count == 2
    assert chunks[0].chunking_strategy == "prechunked"
    assert chunks[1].chunk_index == 1


def test_prepare_pre_chunked_texts_preserves_chunk_metadata() -> None:
    chunks = prepare_pre_chunked_texts(
        ["page one", "page two"],
        chunk_metadata=[
            {"page_number": 1, "page_index": 0},
            {"page_number": 2, "page_index": 1},
        ],
    )

    assert [chunk.metadata["page_number"] for chunk in chunks] == [1, 2]
    assert [chunk.metadata["page_index"] for chunk in chunks] == [0, 1]
    assert chunks[0].metadata["chunking_strategy"] == "prechunked"


def test_prepare_pre_chunked_texts_rejects_mismatched_chunk_metadata() -> None:
    with pytest.raises(ValueError, match="chunk_metadata length mismatch"):
        prepare_pre_chunked_texts(
            ["page one", "page two"],
            chunk_metadata=[{"page_number": 1}],
        )


def test_prepare_pre_chunked_texts_rejects_mismatched_embedding_texts() -> None:
    with pytest.raises(ValueError, match="embedding_texts length mismatch"):
        prepare_pre_chunked_texts(
            ["page one", "page two"],
            embedding_texts=["embedded page one"],
        )


def test_prepare_text_chunks_overrides_embedding_text_when_supplied() -> None:
    [chunk] = prepare_text_chunks(
        "# Title\n\nBody text.\n",
        filename="note.md",
        embedding_texts=["custom embedding text"],
    )
    assert chunk.text != chunk.embedding_text
    assert chunk.embedding_text == "custom embedding text"


def test_prepare_text_chunks_rejects_mismatched_embedding_texts() -> None:
    with pytest.raises(ValueError, match="embedding_texts length mismatch"):
        prepare_text_chunks(
            "# Title\n\nBody text.\n",
            filename="note.md",
            embedding_texts=["custom embedding text", "extra"],
        )


def test_build_index_request_preserves_prepared_chunk_metadata() -> None:
    prepared = PreparedDocument(
        filename="report.pdf",
        mime_type="application/pdf",
        markdown="## Page 1\n\nAlpha",
        chunks=[
            PreparedChunk(
                chunk_index=0,
                text="## Page 1\n\nAlpha",
                embedding_text="## Page 1\n\nAlpha",
                word_count=4,
                metadata={"page_number": 1, "page_index": 0},
            )
        ],
    )

    request = build_index_request(
        prepared=prepared,
        document_id="doc-1",
        document_key="report.pdf",
        content_sha256="abc123",
        processing_version=ProcessingFingerprint(base_version="v1", source_type="file"),
        existing=None,
        corpus_id="docs",
        namespace="acme",
        source_type="file",
        metadata=None,
        embedding_model="fake-model",
    )

    assert request.chunk_metadata == [{"page_number": 1, "page_index": 0}]
