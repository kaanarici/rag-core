from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

REQUIRED_WHEEL_MEMBERS = frozenset(
    {
        "rag_core/py.typed",
        "rag_core/__init__.pyi",
        "rag_core/events/__init__.pyi",
        "rag_core/integrations/__init__.pyi",
        "rag_core/search/__init__.pyi",
        "rag_core/search/providers/__init__.pyi",
    }
)

REQUIRED_SDIST_SUFFIXES = frozenset(
    {
        "README.md",
        "docs/parsing/formats.md",
        "docs/providers/custom-providers.md",
        "docs/providers/provider-output-shapes.md",
        "docs/providers/vector-stores.md",
        "docs/evals/retrieval-quality.md",
        "docs/integrations/vercel-ai-sdk-tools.md",
        "examples/minimal_app.py",
        "examples/search_endpoint.py",
        "examples/retrieval_eval.py",
        "examples/source_ingest.py",
        "examples/vercel_ai_sdk_search_tool.ts",
        "examples/demo_corpus/corpus_lifecycle.md",
        "scripts/wheel_smoke.py",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check built rag-core wheel and sdist public artifacts."
    )
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()

    dist_dir = args.dist_dir
    wheel = _single(dist_dir, "*.whl")
    sdist = _single(dist_dir, "*.tar.gz")

    _check_wheel(wheel)
    _check_sdist(sdist)


def _single(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one {pattern} artifact in {dist_dir}, found {len(matches)}"
        )
    return matches[0]


def _check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
    missing = sorted(REQUIRED_WHEEL_MEMBERS - members)
    if missing:
        raise SystemExit(f"wheel is missing required typed artifacts: {missing}")


def _check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        suffixes = {"/".join(member.name.split("/")[1:]) for member in archive.getmembers()}
    missing = sorted(REQUIRED_SDIST_SUFFIXES - suffixes)
    if missing:
        raise SystemExit(f"sdist is missing required public artifacts: {missing}")


if __name__ == "__main__":
    main()
