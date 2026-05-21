from pathlib import Path

from scripts.architecture_pressure import build_report


def test_architecture_pressure_report_finds_hotspots_ignored_mypy_and_boundaries(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src" / "rag_core"
    (source_root / "search").mkdir(parents=True)
    (source_root / "documents" / "chunking").mkdir(parents=True)
    (source_root / "core.py").write_text(
        "one\ntwo\nthree\nfour\nfive\n", encoding="utf-8"
    )
    (source_root / "small.py").write_text("one\n", encoding="utf-8")
    (source_root / "search" / "chunking.py").write_text("one\n", encoding="utf-8")
    (source_root / "documents" / "chunking" / "markdown.py").write_text(
        "one\n", encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text(
        """
[[tool.mypy.overrides]]
module = ["rag_core.documents.chunking.code", "rag_core.search.providers.sparse"]
ignore_errors = true

[[tool.mypy.overrides]]
module = "missing_pkg"
ignore_missing_imports = true
""",
        encoding="utf-8",
    )

    report = build_report(tmp_path, limit=2, large_file_threshold=4)

    assert report["summary"] == {
        "python_file_count": 4,
        "large_file_count": 1,
        "mypy_ignore_errors_count": 2,
        "duplicate_boundary_warning_count": 1,
    }
    assert report["file_size_hotspots"] == [
        {"path": "src/rag_core/core.py", "lines": 5, "over_threshold": True},
        {
            "path": "src/rag_core/documents/chunking/markdown.py",
            "lines": 1,
            "over_threshold": False,
        },
    ]
    assert report["mypy_ignore_errors_modules"] == [
        "rag_core.documents.chunking.code",
        "rag_core.search.providers.sparse",
    ]
    assert report["duplicate_boundary_warnings"] == [
        {
            "concept": "chunking",
            "paths": [
                "src/rag_core/search/chunking.py",
                "src/rag_core/documents/chunking",
            ],
            "reason": (
                "chunking behavior spans search and document-understanding zones; "
                "keep ownership explicit before adding new chunk strategies"
            ),
        }
    ]
