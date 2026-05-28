"""Server-local path policy for the optional HTTP runtime."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rag_core.runtime.errors import RuntimeRequestError


def normalize_ingest_roots(roots: Sequence[Path] | None) -> tuple[Path, ...]:
    candidates = tuple(roots) if roots else (Path.cwd(),)
    return tuple(_normalize_path(root) for root in candidates)


def validate_ingest_path(path: str, *, roots: Sequence[Path]) -> Path:
    requested = _normalize_path(Path(path))
    for root in roots:
        if requested == root or root in requested.parents:
            if not requested.exists():
                raise RuntimeRequestError(
                    message="path does not exist",
                    details={"field": "path"},
                )
            return requested
    raise RuntimeRequestError(
        message="path is outside configured ingest roots",
        details={
            "field": "path",
            "allowed_roots": [str(root) for root in roots],
        },
    )


def _normalize_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


__all__ = ["normalize_ingest_roots", "validate_ingest_path"]
