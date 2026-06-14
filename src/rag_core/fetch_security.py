from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlunsplit

from rag_core.fetch_security_network import (
    IPAddress,
    parse_fetch_ip_address,
    validate_fetch_ip_address,
)
from rag_core.fetch_security_url import (
    FetchScheme,
    ValidatedFetchUrl,
    normalize_allowed_hosts,
    redact_fetch_url,
    validate_fetch_url_parts,
)

if TYPE_CHECKING:
    from rag_core.fetching_response import FetchResponse

DEFAULT_FETCH_MAX_BYTES = 25 * 1024 * 1024
DEFAULT_FETCH_TIMEOUT_SECONDS = 10.0
DEFAULT_FETCH_MAX_REDIRECTS = 5
FETCH_ALLOW_HTTP_ENV = "RAG_CORE_FETCH_ALLOW_HTTP"
FETCH_ALLOW_PRIVATE_ADDRESSES_ENV = "RAG_CORE_FETCH_ALLOW_PRIVATE_ADDRESSES"
FETCH_MAX_BYTES_ENV = "RAG_CORE_FETCH_MAX_BYTES"
FETCH_MAX_REDIRECTS_ENV = "RAG_CORE_FETCH_MAX_REDIRECTS"
FETCH_TIMEOUT_SECONDS_ENV = "RAG_CORE_FETCH_TIMEOUT_SECONDS"
DEFAULT_FETCH_ALLOWED_CONTENT_TYPES = (
    "application/json",
    "application/jsonl",
    "application/ndjson",
    "application/pdf",
    "application/toml",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/xml",
    "application/xhtml+xml",
    "application/x-ndjson",
    "application/x-yaml",
    "image/svg+xml",
    "text/csv",
    "text/html",
    "text/markdown",
    "text/plain",
    "text/tab-separated-values",
    "text/xml",
    "text/x-markdown",
    "text/x-yaml",
    "text/yaml",
)
_DEFAULT_SCHEMES: tuple[FetchScheme, ...] = ("https",)


@dataclass(frozen=True)
class FetchLimits:
    max_bytes: int = DEFAULT_FETCH_MAX_BYTES
    timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS
    max_redirects: int = DEFAULT_FETCH_MAX_REDIRECTS
    allowed_content_types: tuple[str, ...] = DEFAULT_FETCH_ALLOWED_CONTENT_TYPES

    def __post_init__(self) -> None:
        if self.max_bytes <= 0:
            raise ValueError("FetchLimits.max_bytes must be positive")
        if (
            not isinstance(self.timeout_seconds, int | float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("FetchLimits.timeout_seconds must be finite and positive")
        if self.max_redirects < 0:
            raise ValueError("FetchLimits.max_redirects must be non-negative")
        normalized = tuple(
            _normalize_content_type(content_type)
            for content_type in self.allowed_content_types
        )
        if not normalized:
            raise ValueError("FetchLimits.allowed_content_types must be non-empty")
        object.__setattr__(self, "allowed_content_types", normalized)


@dataclass(frozen=True)
class FetchSecurityPolicy:
    allowed_schemes: tuple[FetchScheme, ...] = _DEFAULT_SCHEMES
    allow_private_addresses: bool = False
    allowed_hosts: tuple[str, ...] | None = None
    """Optional egress allowlist applied to every URL and every redirect hop.

    ``None`` keeps the default private-IP-only block. An empty tuple is the
    canonical restricted-tier value: it denies every outbound HTTP fetch.
    Otherwise each entry is either an exact host (``docs.example.com``) or a
    leading-wildcard suffix (``*.example.com``).
    """

    def __post_init__(self) -> None:
        schemes = tuple(scheme.lower() for scheme in self.allowed_schemes)
        invalid = [scheme for scheme in schemes if scheme not in {"http", "https"}]
        if invalid:
            raise ValueError(f"unsupported fetch URL scheme: {invalid[0]}")
        if not schemes:
            raise ValueError("FetchSecurityPolicy.allowed_schemes must be non-empty")
        object.__setattr__(self, "allowed_schemes", schemes)
        object.__setattr__(
            self,
            "allowed_hosts",
            normalize_allowed_hosts(self.allowed_hosts),
        )


def validate_fetch_url(
    url: str,
    *,
    policy: FetchSecurityPolicy | None = None,
) -> ValidatedFetchUrl:
    active_policy = policy or FetchSecurityPolicy()
    return validate_fetch_url_parts(
        url,
        allowed_schemes=active_policy.allowed_schemes,
        allow_private_addresses=active_policy.allow_private_addresses,
        allowed_hosts=active_policy.allowed_hosts,
    )


def validate_fetch_redirects(
    urls: Sequence[str],
    *,
    policy: FetchSecurityPolicy | None = None,
    limits: FetchLimits | None = None,
) -> tuple[ValidatedFetchUrl, ...]:
    active_limits = limits or FetchLimits()
    if not urls:
        raise ValueError("fetch redirect chain must include at least one URL")
    redirect_count = len(urls) - 1
    if redirect_count > active_limits.max_redirects:
        raise ValueError(
            "fetch redirect chain exceeds max_redirects "
            f"({redirect_count} > {active_limits.max_redirects})"
        )
    return tuple(validate_fetch_url(url, policy=policy) for url in urls)


def validate_fetch_response_policy(
    response: "FetchResponse",
    *,
    policy: FetchSecurityPolicy | None,
) -> None:
    seen: set[tuple[str, str, int | None, str, bool]] = set()
    for url in (*response.redirect_chain, response.url):
        key = (url.scheme, url.host, url.port, url.path, url.has_query)
        if key in seen:
            continue
        seen.add(key)
        validate_fetch_url(_policy_check_url(url), policy=policy)


def validate_resolved_fetch_addresses(
    host: str,
    addresses: Iterable[str | IPAddress],
    *,
    policy: FetchSecurityPolicy | None = None,
) -> tuple[IPAddress, ...]:
    active_policy = policy or FetchSecurityPolicy()
    resolved = tuple(parse_fetch_ip_address(address) for address in addresses)
    if not resolved:
        raise ValueError(f"fetch host {host!r} resolved to no addresses")
    for address in resolved:
        validate_fetch_ip_address(
            address,
            allow_private_addresses=active_policy.allow_private_addresses,
        )
    return resolved


def _policy_check_url(url: ValidatedFetchUrl) -> str:
    path = url.path or "/"
    host = f"[{url.host}]" if ":" in url.host else url.host
    netloc = f"{host}:{url.port}" if url.port is not None else host
    query = "redacted" if url.has_query else ""
    return urlunsplit((url.scheme, netloc, path, query, ""))


def is_allowed_fetch_content_type(
    content_type: str | None,
    *,
    limits: FetchLimits | None = None,
) -> bool:
    if not content_type:
        return False
    active_limits = limits or FetchLimits()
    media_type = _normalize_content_type(content_type)
    return any(
        media_type == allowed
        or (allowed.endswith("/*") and media_type.startswith(f"{allowed[:-2]}/"))
        for allowed in active_limits.allowed_content_types
    )


def _normalize_content_type(content_type: str) -> str:
    media_type = content_type.split(";", 1)[0].strip().lower()
    if not media_type or "/" not in media_type:
        raise ValueError("fetch content type must be a media type")
    return media_type


__all__ = [
    "DEFAULT_FETCH_ALLOWED_CONTENT_TYPES",
    "DEFAULT_FETCH_MAX_BYTES",
    "DEFAULT_FETCH_MAX_REDIRECTS",
    "DEFAULT_FETCH_TIMEOUT_SECONDS",
    "FETCH_ALLOW_HTTP_ENV",
    "FETCH_ALLOW_PRIVATE_ADDRESSES_ENV",
    "FETCH_MAX_BYTES_ENV",
    "FETCH_MAX_REDIRECTS_ENV",
    "FETCH_TIMEOUT_SECONDS_ENV",
    "FetchLimits",
    "FetchSecurityPolicy",
    "ValidatedFetchUrl",
    "is_allowed_fetch_content_type",
    "redact_fetch_url",
    "validate_fetch_redirects",
    "validate_fetch_response_policy",
    "validate_fetch_url",
    "validate_resolved_fetch_addresses",
]
