from __future__ import annotations

from pathlib import Path


def test_sdist_manifest_includes_readme_link_targets() -> None:
    manifest = Path("MANIFEST.in").read_text(encoding="utf-8")

    required_lines = (
        "recursive-include docs *.md",
        "recursive-include examples *.jsonl *.md *.py *.ts",
        "recursive-include scripts *.py",
    )
    for line in required_lines:
        assert line in manifest


def test_public_docs_and_examples_are_present_for_packaging() -> None:
    public_targets = (
        "docs/parsing/formats.md",
        "docs/providers.md",
        "docs/self-host.md",
        "examples/minimal_app.py",
        "examples/search_endpoint.py",
        "examples/source_ingest.py",
        "examples/vercel_ai_sdk_search_tool.ts",
        "examples/demo_corpus/corpus_lifecycle.md",
    )
    for target in public_targets:
        assert Path(target).is_file()
