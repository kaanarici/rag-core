from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import TypedDict

DEFAULT_LIMIT = 20
DEFAULT_LARGE_FILE_THRESHOLD = 600


class PythonFilePressure(TypedDict):
    path: str
    lines: int


class PythonFileHotspot(PythonFilePressure):
    over_threshold: bool


class BoundaryWarning(TypedDict):
    concept: str
    paths: list[str]
    reason: str


class Summary(TypedDict):
    python_file_count: int
    large_file_count: int
    mypy_ignore_errors_count: int
    duplicate_boundary_warning_count: int


class ArchitecturePressureReport(TypedDict):
    root: str
    large_file_threshold: int
    summary: Summary
    file_size_hotspots: list[PythonFileHotspot]
    mypy_ignore_errors_modules: list[str]
    duplicate_boundary_warnings: list[BoundaryWarning]


def build_report(
    root: Path,
    *,
    limit: int = DEFAULT_LIMIT,
    large_file_threshold: int = DEFAULT_LARGE_FILE_THRESHOLD,
) -> ArchitecturePressureReport:
    root = root.resolve()
    files = _python_file_pressures(root)
    hotspots = files[:limit]
    ignored_modules = _mypy_ignore_error_modules(root / "pyproject.toml")
    boundary_warnings = _duplicate_boundary_warnings(root)
    file_size_hotspots: list[PythonFileHotspot] = [
        {
            **file,
            "over_threshold": file["lines"] >= large_file_threshold,
        }
        for file in hotspots
    ]
    return {
        "root": str(root),
        "large_file_threshold": large_file_threshold,
        "summary": {
            "python_file_count": len(files),
            "large_file_count": sum(
                1 for file in files if file["lines"] >= large_file_threshold
            ),
            "mypy_ignore_errors_count": len(ignored_modules),
            "duplicate_boundary_warning_count": len(boundary_warnings),
        },
        "file_size_hotspots": file_size_hotspots,
        "mypy_ignore_errors_modules": ignored_modules,
        "duplicate_boundary_warnings": boundary_warnings,
    }


def _python_file_pressures(root: Path) -> list[PythonFilePressure]:
    source_root = root / "src" / "rag_core"
    files: list[PythonFilePressure] = [
        {
            "path": _relative_path(path, root),
            "lines": _line_count(path),
        }
        for path in source_root.rglob("*.py")
        if path.is_file()
    ]
    return sorted(files, key=lambda item: (-int(item["lines"]), str(item["path"])))


def _mypy_ignore_error_modules(pyproject_path: Path) -> list[str]:
    if not pyproject_path.exists():
        return []
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    overrides = (
        pyproject.get("tool", {})
        .get("mypy", {})
        .get("overrides", [])
    )
    modules: list[str] = []
    for override in overrides:
        if not isinstance(override, dict) or override.get("ignore_errors") is not True:
            continue
        raw_module = override.get("module", [])
        if isinstance(raw_module, str):
            modules.append(raw_module)
        elif isinstance(raw_module, list):
            modules.extend(str(module) for module in raw_module)
    return sorted(modules)


def _duplicate_boundary_warnings(root: Path) -> list[BoundaryWarning]:
    warnings: list[BoundaryWarning] = []
    _append_chunking_boundary_warning(root, warnings)
    return warnings


def _append_chunking_boundary_warning(
    root: Path,
    warnings: list[BoundaryWarning],
) -> None:
    search_chunking = root / "src" / "rag_core" / "search" / "chunking.py"
    document_chunking = root / "src" / "rag_core" / "documents" / "chunking"
    if not search_chunking.exists() or not document_chunking.exists():
        return
    document_files = sorted(
        path for path in document_chunking.rglob("*.py") if path.is_file()
    )
    if not document_files:
        return
    warnings.append(
        {
            "concept": "chunking",
            "paths": [
                _relative_path(search_chunking, root),
                _relative_path(document_chunking, root),
            ],
            "reason": (
                "chunking behavior spans search and document-understanding zones; "
                "keep ownership explicit before adding new chunk strategies"
            ),
        }
    )


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _emit_text(report: ArchitecturePressureReport) -> None:
    summary = report["summary"]
    print("Architecture Pressure")
    print(f"Root: {report['root']}")
    print(
        "Summary: "
        f"python_files={summary['python_file_count']} "
        f"large_files={summary['large_file_count']} "
        f"mypy_ignore_errors={summary['mypy_ignore_errors_count']} "
        f"duplicate_boundaries={summary['duplicate_boundary_warning_count']}"
    )
    print("\nFile Size Hotspots:")
    for file in report["file_size_hotspots"]:
        marker = " *" if file["over_threshold"] else ""
        print(f"  {file['lines']:>5} {file['path']}{marker}")
    print("\nMypy ignore_errors Modules:")
    for module in report["mypy_ignore_errors_modules"]:
        print(f"  {module}")
    print("\nDuplicate Boundary Warnings:")
    for warning in report["duplicate_boundary_warnings"]:
        print(f"  {warning['concept']}: {warning['reason']}")
        for path in warning["paths"]:
            print(f"    - {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report local architecture pressure for the rag-core checkout."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of largest source files to print. Default: {DEFAULT_LIMIT}.",
    )
    parser.add_argument(
        "--large-file-threshold",
        type=int,
        default=DEFAULT_LARGE_FILE_THRESHOLD,
        help=(
            "Line-count threshold used for the large-file summary. "
            f"Default: {DEFAULT_LARGE_FILE_THRESHOLD}."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.large_file_threshold <= 0:
        raise ValueError("--large-file-threshold must be positive")
    report = build_report(
        args.root,
        limit=args.limit,
        large_file_threshold=args.large_file_threshold,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _emit_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
