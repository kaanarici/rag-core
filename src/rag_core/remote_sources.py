from __future__ import annotations

import posixpath
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import unquote

from rag_core.config import INGEST_SOURCE_TYPE_URL
from rag_core.documents.converters.format_support import (
    format_support_for_extension,
    format_support_for_mime_type,
)
from rag_core.fetch_security import (
    FetchLimits,
    FetchSecurityPolicy,
    ValidatedFetchUrl,
    validate_fetch_url,
)
from rag_core.fetch_security_url import (
    safe_remote_source_url,
)
from rag_core.fetching import (
    FetchClient as _FetchClient,
    FetchResponse,
    HttpFetchClient,
)
from rag_core.fetching_response import validate_fetch_response
from rag_core.remote_discovery_policy import validate_fetch_response_policy
from rag_core.remote_document_keys import (
    has_private_query_identity,
    private_remote_document_key,
    public_remote_document_key,
)

_CONTENT_TYPE_EXTENSIONS: Mapping[str, str] = {
    "application/json": ".json",
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/xhtml+xml": ".html",
    "application/xml": ".xml",
    "application/toml": ".toml",
    "application/x-yaml": ".yaml",
    "image/bmp": ".bmp",
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/svg+xml": ".svg",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
    "text/csv": ".csv",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/plain": ".txt",
    "text/x-markdown": ".md",
    "text/xml": ".xml",
    "text/yaml": ".yaml",
    "text/x-yaml": ".yaml",
}


@dataclass(frozen=True)
class RemoteSourceDocument:
    redacted_url: str
    document_key: str
    filename: str
    mime_type: str
    status_code: int
    content_length: int | None
    content_sha256: str
    byte_count: int
    redirect_count: int
    file_bytes: bytes = field(repr=False)
    source_type: str = INGEST_SOURCE_TYPE_URL
    requested_url: str | None = None

    def to_payload(self, *, include_private: bool = False) -> dict[str, object]:
        document_key = (
            self.document_key
            if include_private
            else public_remote_document_key(self.document_key)
        )
        payload: dict[str, object] = {
            "source_type": self.source_type,
            "url": self.redacted_url,
            "requested_url": self.requested_url or self.redacted_url,
            "document_key": document_key,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "status_code": self.status_code,
            "content_length": self.content_length,
            "content_sha256": self.content_sha256,
            "byte_count": self.byte_count,
            "redirect_count": self.redirect_count,
        }
        if not include_private and has_private_query_identity(self.document_key):
            payload["has_private_query_identity"] = True
        return payload

    def to_parse_kwargs(self) -> dict[str, object]:
        return {
            "file_bytes": self.file_bytes,
            "filename": self.filename,
            "mime_type": self.mime_type,
        }

    def to_source_metadata(self) -> dict[str, str]:
        metadata = {
            "source_type": self.source_type,
            "source_url": self.redacted_url,
        }
        if self.requested_url and self.requested_url != self.redacted_url:
            metadata["source_requested_url"] = self.requested_url
        return metadata


class RemoteUrlSourceReader:
    def __init__(
        self,
        *,
        fetch_client: _FetchClient | None = None,
        policy: FetchSecurityPolicy | None = None,
        limits: FetchLimits | None = None,
    ) -> None:
        if fetch_client is not None and limits is not None:
            raise ValueError("fetch_client cannot be combined with limits")
        if fetch_client is not None and policy is not None:
            raise ValueError("fetch_client cannot be combined with policy")
        self._limits = limits or FetchLimits()
        self._fetch_client = fetch_client or HttpFetchClient(
            policy=policy, limits=limits
        )
        self._policy = policy

    def read(self, url: str) -> RemoteSourceDocument:
        validate_fetch_url(url, policy=self._policy)
        response = self._fetch_client.fetch(url)
        validate_fetch_response(response, limits=self._limits)
        validate_fetch_response_policy(response, policy=self._policy)
        return remote_source_document(response)


def remote_source_document(response: FetchResponse) -> RemoteSourceDocument:
    mime_type = _media_type(response.content_type)
    redacted_url = safe_remote_source_url(response.url)
    requested_url = (
        response.redirect_chain[0] if response.redirect_chain else response.url
    )
    return RemoteSourceDocument(
        redacted_url=redacted_url,
        document_key=_document_key(response.url),
        filename=_remote_filename(path=response.url.path, mime_type=mime_type),
        mime_type=mime_type,
        status_code=response.status_code,
        content_length=response.content_length,
        content_sha256=response.content_sha256,
        byte_count=len(response.body),
        redirect_count=max(len(response.redirect_chain) - 1, 0),
        file_bytes=response.body,
        requested_url=safe_remote_source_url(requested_url),
    )


def _document_key(url: ValidatedFetchUrl) -> str:
    return private_remote_document_key(
        f"url:{safe_remote_source_url(url)}", url.query_sha256
    )


def _remote_filename(*, path: str, mime_type: str) -> str:
    extension = _extension_for_mime_type(mime_type)
    name = _safe_filename(unquote(posixpath.basename(path.strip("/"))))
    if not name:
        return f"index{extension}"
    stem, existing_extension = posixpath.splitext(name)
    if not stem:
        return f"index{extension}"
    if _extension_matches_mime(existing_extension, mime_type=mime_type):
        return f"{stem}{existing_extension.lower()}"
    return f"{stem}{extension}"


def _extension_for_mime_type(mime_type: str) -> str:
    resolved = mime_type.lower().strip()
    if resolved in _CONTENT_TYPE_EXTENSIONS:
        return _CONTENT_TYPE_EXTENSIONS[resolved]
    if resolved.startswith("text/"):
        return ".txt"
    support = format_support_for_mime_type(resolved)
    if support is not None and support.extensions:
        return support.extensions[0]
    raise ValueError(
        f"remote source MIME type is not supported: {resolved or '<missing>'}"
    )


def _extension_matches_mime(extension: str, *, mime_type: str) -> bool:
    if not extension:
        return False
    extension_support = format_support_for_extension(extension)
    mime_support = format_support_for_mime_type(mime_type)
    return (
        extension_support is not None
        and mime_support is not None
        and extension_support.converter_key == mime_support.converter_key
    )


def _safe_filename(name: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {".", "-", "_"} else "-"
        for char in name.strip()
    ).strip(".-")
    return safe if safe not in {"", ".", ".."} else ""


def _media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


__all__ = [
    "RemoteSourceDocument",
    "RemoteUrlSourceReader",
    "remote_source_document",
]
