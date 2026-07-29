from __future__ import annotations


def manifest_segment(label: str, value: str) -> str:
    segment = value.strip()
    if (
        not segment
        or segment != value
        or segment in {".", ".."}
        or "/" in segment
        or "\\" in segment
    ):
        raise ValueError(f"{label} must be a single non-empty path segment")
    if segment != segment.lower():
        raise ValueError(f"{label} must be lowercase")
    return segment


def manifest_scope_segments(namespace: str, collection: str) -> tuple[str, str]:
    return manifest_segment("namespace", namespace), manifest_segment(
        "collection", collection
    )
