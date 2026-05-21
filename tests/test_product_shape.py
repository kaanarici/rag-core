from __future__ import annotations

from pathlib import Path

import pytest

import rag_core
from rag_core.cli import main
from rag_core.cli_output import search_hit_payload
from rag_core.core_file_io import detect_local_mime_type
from rag_core.documents.converters import get_converter
from tests.support import make_search_result


def test_demo_factory_is_not_root_public_api() -> None:
    assert "build_demo_core" not in rag_core.__all__
    assert not hasattr(rag_core, "build_demo_core")


def test_rag_core_facade_modules_live_under_facade_package() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "rag_core"

    assert not list(root.glob("core_facade_*.py"))
    assert sorted(path.name for path in (root / "facade").glob("*.py")) == [
        "__init__.py",
        "ingest.py",
        "ingest_batches.py",
        "ingest_sources.py",
        "manifest.py",
        "prepare.py",
        "retrieval.py",
    ]


@pytest.mark.parametrize(
    "command",
    ("doctor", "ingest", "ingest-url", "search"),
)
def test_cli_help_uses_vector_store_language(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([command, "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "configured Qdrant collection" not in output


def test_cli_top_level_help_excludes_removed_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for removed in ("eval", "trace-summary"):
        assert f"  {removed} " not in f"\n{output}\n"
    assert "  serve " in f"\n{output}\n"


def test_search_hit_payload_matches_ragie_scored_chunk_fields() -> None:
    hit = make_search_result(
        document_id="doc-1",
        document_key="docs/guide.md",
        score=0.88,
    )
    payload = search_hit_payload(hit)
    assert payload["id"] == hit.id
    assert payload["text"] == hit.text
    assert payload["score"] == 0.88
    assert payload["document_id"] == "doc-1"
    assert payload["document_key"] == "docs/guide.md"
    assert isinstance(payload["metadata"], dict)


def test_local_typescript_mime_detection_preserves_code_converter_routing() -> None:
    for filename in ("widget.ts", "widget.tsx"):
        mime_type = detect_local_mime_type(Path(filename))

        assert mime_type == "text/typescript"
        assert get_converter(mime_type=mime_type, filename=filename).format_name == "code"


def test_generic_text_mime_does_not_hide_known_code_extensions() -> None:
    for filename in (
        "main.zig",
        "component.vue",
        "query.graphql",
        "schema.tfvars",
        "build.gradle",
        "build.cmake",
        "module.cxx",
        "objc.m",
        "rules.make",
        "rules.mak",
    ):
        mime_type = detect_local_mime_type(Path(filename))

        assert not mime_type.startswith("video/")
        assert get_converter(mime_type=mime_type, filename=filename).format_name == "code"
