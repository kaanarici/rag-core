from __future__ import annotations

_QUERY_KEY_MARKER = "|query_sha256:"


def private_remote_document_key(
    public_document_key: str, query_sha256: str | None
) -> str:
    if query_sha256:
        return f"{public_document_key}{_QUERY_KEY_MARKER}{query_sha256}"
    return public_document_key


def public_remote_document_key(document_key: str) -> str:
    return document_key.split(_QUERY_KEY_MARKER, 1)[0]


def has_private_query_identity(document_key: str) -> bool:
    return _QUERY_KEY_MARKER in document_key


__all__ = [
    "has_private_query_identity",
    "private_remote_document_key",
    "public_remote_document_key",
]
