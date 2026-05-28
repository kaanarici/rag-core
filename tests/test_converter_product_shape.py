from __future__ import annotations

import ast
from pathlib import Path

from rag_core.core_file_io import detect_local_mime_type
from rag_core.documents.converters import get_converter
from rag_core.documents.converters.converter_keys import CONVERTER_KEYS
from rag_core.documents.converters.format_support import FORMAT_SUPPORT_MATRIX
from rag_core.documents.converters.registry_specs import CONVERTER_SPECS

CANONICAL_LAUNCH_GATES = (
    "uv run ruff check .",
    "uv run mypy src tests examples",
    "uv run pytest -q",
    "./scripts/dx_smoke.sh",
    "./scripts/ci_self_host_smoke.sh",
    "uv build",
    "uv run python scripts/check_dist_artifacts.py",
    "uv run python scripts/wheel_smoke.py",
)


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




def test_converter_registry_keys_have_single_owner() -> None:
    root = Path(__file__).resolve().parents[1]
    owner = root / "src" / "rag_core" / "documents" / "converters" / "converter_keys.py"
    owner_source = owner.read_text(encoding="utf-8")

    for key in CONVERTER_KEYS:
        assert owner_source.count(f'= "{key}"') == 1

    assert tuple(spec.key for spec in CONVERTER_SPECS) == CONVERTER_KEYS
    assert tuple(entry.key for entry in FORMAT_SUPPORT_MATRIX) == CONVERTER_KEYS
    assert (
        tuple(entry.converter_key for entry in FORMAT_SUPPORT_MATRIX) == CONVERTER_KEYS
    )

    key_literals = set(CONVERTER_KEYS)
    for relative_path in (
        "src/rag_core/documents/converters/__init__.py",
        "src/rag_core/documents/converters/format_support.py",
        "src/rag_core/documents/converters/format_support_lookup.py",
        "src/rag_core/documents/converters/format_support_matrix.py",
        "src/rag_core/documents/converters/registry_maps.py",
        "src/rag_core/documents/converters/registry_specs.py",
    ):
        tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
        raw_key_literals = sorted(
            {
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in key_literals
            }
        )
        assert raw_key_literals == []

    for relative_path in (
        "src/rag_core/documents/converters/code_converter.py",
        "src/rag_core/documents/converters/csv_converter.py",
        "src/rag_core/documents/converters/docx_converter.py",
        "src/rag_core/documents/converters/html_converter.py",
        "src/rag_core/documents/converters/image_converter.py",
        "src/rag_core/documents/converters/json_converter.py",
        "src/rag_core/documents/converters/pdf_converter.py",
        "src/rag_core/documents/converters/pptx_converter.py",
        "src/rag_core/documents/converters/text_converter.py",
        "src/rag_core/documents/converters/xlsx_converter.py",
        "src/rag_core/documents/converters/xml_converter.py",
    ):
        source = (root / relative_path).read_text(encoding="utf-8")
        for key in CONVERTER_KEYS:
            assert f'format_name = "{key}"' not in source
