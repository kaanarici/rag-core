"""Build text representations used for retrieval.

Dense and sparse indexing may use structured metadata as additional signal.
Stored payload text stays as the clean chunk body.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from rag_core.search.vector_models import ContentType


def build_sparse_text(
    chunk_text: str,
    metadata: dict[str, Any],
) -> str:
    """Build plain-word sparse input from metadata and chunk content."""
    parts: list[str] = []
    for _key, value in sorted(metadata.items()):
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, list):
            parts.extend(str(v).strip() for v in value if str(v).strip())
    parts.append(chunk_text)
    return " ".join(parts)


def build_textual_representation(
    content: str,
    source_type: str,
    name: str,
    content_type: Union[ContentType, str],
    *,
    path: Optional[str] = None,
    extra_fields: Optional[dict[str, str]] = None,
) -> str:
    """Build enriched text for retrieval and embedding.

    Code content is returned unchanged. Other content includes compact metadata
    lines as retrieval signal.
    Do not use this for display, prompt context, or ``SearchResult.text``.
    """
    if content_type == ContentType.CODE:
        return content

    return _build_document_representation(
        content=content,
        source_type=source_type,
        name=name,
        path=path,
        extra_fields=extra_fields,
    )


def _build_document_representation(
    content: str,
    source_type: str,
    name: str,
    path: Optional[str] = None,
    extra_fields: Optional[dict[str, str]] = None,
) -> str:
    lines = [
        f"source_type: {source_type}",
        "type: document",
        f"name: {name}",
    ]
    if path:
        lines.append(f"path: {path}")

    if extra_fields:
        for key, value in extra_fields.items():
            lines.append(f"{key}: {value}")

    lines.extend(["", content])
    return "\n".join(lines)
