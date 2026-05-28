"""URL parsing and redaction helpers for fetch security."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal
from urllib.parse import SplitResult, unquote, urlsplit, urlunsplit

from rag_core.fetch_security_network import (
    normalize_fetch_hostname,
    validate_fetch_host,
)
from rag_core.remote_document_keys import private_remote_document_key

FetchScheme = Literal["http", "https"]


@dataclass(frozen=True)
class ValidatedFetchUrl:
    redacted_url: str
    scheme: FetchScheme
    host: str
    port: int | None
    path: str
    has_query: bool
    query_sha256: str | None


def validate_fetch_url_parts(
    url: str,
    *,
    allowed_schemes: tuple[FetchScheme, ...],
    allow_private_addresses: bool,
) -> ValidatedFetchUrl:
    raw = url.strip()
    if not raw:
        raise ValueError("fetch URL must be non-empty")

    parsed = urlsplit(raw)
    scheme = _fetch_scheme(parsed.scheme, allowed_schemes=allowed_schemes)
    host = _normalized_host(parsed)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("fetch URL credentials are not allowed")
    validate_fetch_host(host, allow_private_addresses=allow_private_addresses)

    safe_url = _url_with_safe_netloc(parsed, scheme=scheme, host=host)
    return ValidatedFetchUrl(
        redacted_url=redact_fetch_url(safe_url),
        scheme=scheme,
        host=host,
        port=parsed.port,
        path=_canonical_url_path(parsed.path),
        has_query=bool(parsed.query),
        query_sha256=hashlib.sha256(parsed.query.encode("utf-8")).hexdigest()
        if parsed.query
        else None,
    )


def redact_fetch_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if not parsed.scheme or not parsed.hostname:
        return "<invalid-url>"
    try:
        port = parsed.port
    except ValueError:
        return "<invalid-url>"
    host = normalize_fetch_hostname(parsed.hostname)
    scheme = parsed.scheme.lower()
    netloc = _netloc(host=host, port=_canonical_url_port(scheme=scheme, port=port))
    path = _canonical_url_path(parsed.path)
    query = "redacted" if parsed.query else ""
    return urlunsplit((scheme, netloc, path, query, ""))


def safe_remote_event_url(url: ValidatedFetchUrl) -> str:
    netloc = _netloc(
        host=url.host,
        port=_canonical_url_port(scheme=url.scheme, port=url.port),
    )
    query = "redacted" if url.has_query else ""
    return urlunsplit((url.scheme, netloc, "/", query, ""))


def safe_remote_source_url(url: ValidatedFetchUrl) -> str:
    netloc = _netloc(
        host=url.host,
        port=_canonical_url_port(scheme=url.scheme, port=url.port),
    )
    query = "redacted" if url.has_query else ""
    return urlunsplit((url.scheme, netloc, url.path, query, ""))


def safe_remote_document_key(url: ValidatedFetchUrl) -> str:
    return private_remote_document_key(
        f"url:{safe_remote_source_url(url)}", url.query_sha256
    )


def _fetch_scheme(
    scheme: str,
    *,
    allowed_schemes: tuple[FetchScheme, ...],
) -> FetchScheme:
    normalized = scheme.lower()
    if normalized not in allowed_schemes:
        if normalized == "http":
            raise ValueError(
                "unsupported fetch URL scheme: http (HTTP requires explicit opt-in)"
            )
        raise ValueError(f"unsupported fetch URL scheme: {scheme or '<missing>'}")
    return "https" if normalized == "https" else "http"


def _normalized_host(parsed: SplitResult) -> str:
    if parsed.hostname is None:
        raise ValueError("fetch URL must include a host")
    return normalize_fetch_hostname(parsed.hostname)


def _url_with_safe_netloc(
    parsed: SplitResult, *, scheme: FetchScheme, host: str
) -> str:
    port = _canonical_url_port(scheme=scheme, port=parsed.port)
    path = _canonical_url_path(parsed.path)
    return urlunsplit((scheme, _netloc(host=host, port=port), path, parsed.query, ""))


def _canonical_url_port(*, scheme: str, port: int | None) -> int | None:
    if port is None:
        return None
    if (scheme == "https" and port == 443) or (scheme == "http" and port == 80):
        return None
    return port


def _canonical_url_path(path: str) -> str:
    if not path:
        return "/"

    raw_segments = path.split("/")
    segments: list[str] = []
    trailing_slash = path.endswith("/")
    for index, segment in enumerate(raw_segments):
        dot_segment = _dot_segment(segment)
        if dot_segment == ".":
            trailing_slash = index == len(raw_segments) - 1
            continue
        if dot_segment == "..":
            if len(segments) > 1 or (segments and segments[0] != ""):
                segments.pop()
            trailing_slash = index == len(raw_segments) - 1
            continue
        segments.append(segment)

    if not segments:
        return "/"
    canonical = "/".join(segments)
    if not canonical.startswith("/"):
        canonical = f"/{canonical}"
    if trailing_slash and canonical != "/" and not canonical.endswith("/"):
        canonical = f"{canonical}/"
    return canonical or "/"


def _dot_segment(segment: str) -> str | None:
    decoded = unquote(segment)
    if decoded in {".", ".."}:
        return decoded
    return None


def _netloc(*, host: str, port: int | None) -> str:
    rendered_host = f"[{host}]" if ":" in host else host
    return f"{rendered_host}:{port}" if port is not None else rendered_host


__all__ = [
    "FetchScheme",
    "ValidatedFetchUrl",
    "redact_fetch_url",
    "safe_remote_document_key",
    "safe_remote_event_url",
    "safe_remote_source_url",
    "validate_fetch_url_parts",
]
