import pytest

from rag_core.config import CODE_CHUNKING_STRATEGY, MARKDOWN_CHUNKING_STRATEGY
from rag_core._engine.core_prepare import prepare_text_chunks
from rag_core.documents.chunking.router import chunk_text, is_code_content


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({"filename": "query.sql"}, True, id="sql-by-extension"),
        pytest.param({"filename": "script.cc"}, True, id="cpp-by-extension"),
        pytest.param({"mime_type": "text/x-python"}, True, id="python-mime"),
        pytest.param(
            {"text": "def run():\n    return True\n\nclass Worker:\n    pass\n"},
            True,
            id="python-by-content",
        ),
        pytest.param(
            {"mime_type": "text/x-custom", "filename": "notes.txt", "allow_text_x_prefix": True},
            True,
            id="text-x-prefix-when-allowed",
        ),
        pytest.param(
            {"mime_type": "text/plain", "filename": "notes.txt"},
            False,
            id="plain-text-not-code",
        ),
    ],
)
def test_is_code_content_router_classification(
    kwargs: dict[str, object], expected: bool
) -> None:
    assert is_code_content(**kwargs) is expected  # type: ignore[arg-type]


def test_router_picks_code_strategy_for_source_files() -> None:
    chunks = chunk_text("def run():\n    return True\n", filename="script.py")

    assert chunks
    assert chunks[0].metadata.get("chunking_strategy") == CODE_CHUNKING_STRATEGY


def test_prepared_chunks_preserve_router_positions_and_strategy() -> None:
    text = "# Heading\n\nThis is a short paragraph.\n"
    prepared = prepare_text_chunks(text, filename="note.md")
    routed = chunk_text(text, filename="note.md")

    assert prepared and len(prepared) == len(routed)
    p, r = prepared[0], routed[0]
    assert (p.text, p.start_char, p.end_char) == (r.text, r.start_char, r.end_char)
    assert p.embedding_text == r.text
    assert p.word_count == len(r.text.split())
    assert p.token_count == len(r.text.split())
    assert p.chunking_strategy == MARKDOWN_CHUNKING_STRATEGY
