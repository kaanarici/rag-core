"""Converter routing behavior and converter-key single-owner contracts.

The routing/language tests are behavioral and asserted verbatim. The key-owner
test proves -- without pinning a file list -- that every converter key lives in
the ``converter_keys`` owner: the registry tables equal the owner tuple, the
table-owning modules import from the owner, and no converter class hardcodes a
raw key string for ``format_name``. (Previously this scraped hand-pinned lists
of registry and converter files for literal occurrences, which froze the
layout.)
"""

from __future__ import annotations

import ast
from pathlib import Path

from rag_core.file_io import detect_local_mime_type
from rag_core.documents.converters import get_converter
from rag_core.documents.converters.code_converter import detect_language
from rag_core.documents.converters.converter_keys import CONVERTER_KEYS
from rag_core.documents.converters.format_support import FORMAT_SUPPORT_MATRIX
from rag_core.documents.converters.registry_specs import CONVERTER_SPECS

from tests.support.source_graph import import_graph, iter_package_sources

CONVERTERS_ROOT = "src/rag_core/documents/converters"
CONVERTER_KEYS_OWNER = "rag_core.documents.converters.converter_keys"


def test_local_typescript_mime_detection_preserves_code_converter_routing() -> None:
    for filename in ("widget.ts", "widget.tsx"):
        mime_type = detect_local_mime_type(Path(filename))

        assert mime_type == "text/typescript"
        assert (
            get_converter(mime_type=mime_type, filename=filename).format_name == "code"
        )


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
        assert (
            get_converter(mime_type=mime_type, filename=filename).format_name == "code"
        )


def test_code_converter_and_chunking_share_language_extension_owner() -> None:
    assert detect_language("component.vue") == "vue"
    assert detect_language("schema.tfvars") == "terraform"
    assert detect_language("module.cxx") == "cpp"
    assert detect_language("unknown.note") == "text"


def _format_name_raw_literals() -> dict[str, list[str]]:
    """Converter classes whose ``format_name`` is a raw key string, not a constant."""
    keyset = set(CONVERTER_KEYS)
    offenders: dict[str, list[str]] = {}
    for _rel, dotted, source in iter_package_sources(CONVERTERS_ROOT):
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                if "format_name" not in targets:
                    continue
                value = stmt.value
                if isinstance(value, ast.Constant) and value.value in keyset:
                    offenders.setdefault(dotted, []).append(str(value.value))
    return offenders


def test_converter_registry_keys_have_single_owner() -> None:
    # The registry tables are built from the owner tuple, not from re-listed key
    # strings: assert their value equality.
    assert tuple(spec.key for spec in CONVERTER_SPECS) == CONVERTER_KEYS
    assert tuple(entry.key for entry in FORMAT_SUPPORT_MATRIX) == CONVERTER_KEYS
    assert (
        tuple(entry.converter_key for entry in FORMAT_SUPPORT_MATRIX) == CONVERTER_KEYS
    )

    # The table-owning modules import the keys from the owner rather than
    # embedding raw literals.
    graph = import_graph(CONVERTERS_ROOT)
    for table_owner in (
        "rag_core.documents.converters.registry_specs",
        "rag_core.documents.converters.format_support_matrix",
    ):
        assert any(
            imported == CONVERTER_KEYS_OWNER
            or imported.startswith(f"{CONVERTER_KEYS_OWNER}.")
            for imported in graph[table_owner]
        )

    # No converter class hardcodes a raw key string for its public ``format_name``;
    # each must bind a ``*_CONVERTER_KEY`` constant from the owner.
    assert _format_name_raw_literals() == {}
