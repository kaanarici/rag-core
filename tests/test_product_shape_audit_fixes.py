from __future__ import annotations

from pathlib import Path

from rag_core.core_file_io import detect_local_mime_type
from rag_core.documents.converters import get_converter
from rag_core.search import SearchQuery, SparseVector


def test_search_query_is_public_from_search_namespace() -> None:
    query = SearchQuery(
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVector(indices=[], values=[]),
        namespace="acme",
        corpus_ids=["help"],
    )

    assert query.namespace == "acme"


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
