"""URL/host/network/response fetch-security validation: SSRF guards, host pinning, redirect and content-type policy."""

from __future__ import annotations

import ipaddress
import re
from typing import TypeAlias
import hashlib
from dataclasses import dataclass
from typing import Literal
from urllib.parse import SplitResult, unquote, urlsplit, urlunsplit
from rag_core.ingest.urls.document_keys import private_remote_document_key
import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING


IPAddress: TypeAlias = ipaddress.IPv4Address | ipaddress.IPv6Address

_AMBIGUOUS_IPV4_RE = re.compile(r"(?:0x[0-9a-f]+|0[0-7]+|[0-9]+)", re.IGNORECASE)

_NAT64_WELL_KNOWN_PREFIX = ipaddress.ip_network("64:ff9b::/96")

_EXPLICITLY_NON_PUBLIC_IPV6_NETWORKS = (
    ipaddress.ip_network("100:0:0:1::/64"),
    ipaddress.ip_network("5f00::/16"),
    ipaddress.ip_network("fec0::/10"),
)

def normalize_fetch_hostname(hostname: str) -> str:
    host = hostname.strip().rstrip(".").lower()
    if not host:
        raise ValueError("fetch URL host must be non-empty")
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("fetch URL host is not valid IDNA") from exc

def validate_fetch_host(host: str, *, allow_private_addresses: bool) -> None:
    if host == "localhost" or host.endswith(".localhost"):
        if not allow_private_addresses:
            raise ValueError("fetch URL host resolves to a local address")
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if _looks_like_ambiguous_ipv4(host):
            raise ValueError("fetch URL host looks like an ambiguous IP address")
        return
    validate_fetch_ip_address(
        address,
        allow_private_addresses=allow_private_addresses,
    )

def validate_fetch_ip_address(
    address: IPAddress,
    *,
    allow_private_addresses: bool,
) -> None:
    embedded_ipv4 = _embedded_ipv4(address)
    if embedded_ipv4 is not None:
        validate_fetch_ip_address(
            embedded_ipv4,
            allow_private_addresses=allow_private_addresses,
        )
    if not allow_private_addresses and (
        _is_explicitly_non_public_ipv6(address) or not address.is_global
    ):
        raise ValueError(f"fetch URL address is not public: {address}")

def parse_fetch_ip_address(address: str | IPAddress) -> IPAddress:
    if isinstance(address, ipaddress.IPv4Address | ipaddress.IPv6Address):
        return address
    return ipaddress.ip_address(address)

def _embedded_ipv4(address: IPAddress) -> ipaddress.IPv4Address | None:
    if isinstance(address, ipaddress.IPv4Address):
        return None
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address in _NAT64_WELL_KNOWN_PREFIX:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return None

def _is_explicitly_non_public_ipv6(address: IPAddress) -> bool:
    return isinstance(address, ipaddress.IPv6Address) and any(
        address in network for network in _EXPLICITLY_NON_PUBLIC_IPV6_NETWORKS
    )

def _looks_like_ambiguous_ipv4(host: str) -> bool:
    parts = host.split(".")
    return 1 <= len(parts) <= 4 and all(
        bool(_AMBIGUOUS_IPV4_RE.fullmatch(part)) for part in parts
    )


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
    allowed_hosts: tuple[str, ...] | None = None,
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
    if allowed_hosts is not None and not host_matches_allowed_hosts(
        host, allowed_hosts=allowed_hosts
    ):
        raise ValueError(
            f"fetch URL host is not in allowed_hosts: {host}"
        )

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

def normalize_allowed_hosts(
    allowed_hosts: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    """Lowercase + IDNA-encode every allowed host pattern.

    ``None`` means unrestricted. An empty tuple means deny-all and is preserved.
    Each pattern is either an exact host or a leading-``*.`` wildcard suffix.
    Wildcard patterns must include at least one labeled suffix after ``*.``.
    """
    if allowed_hosts is None:
        return None
    if not allowed_hosts:
        return ()
    normalized: list[str] = []
    for raw in allowed_hosts:
        candidate = raw.strip().lower().rstrip(".")
        if not candidate:
            raise ValueError("fetch allowed_hosts entries must be non-empty")
        if candidate.startswith("*."):
            suffix = candidate[2:]
            if not suffix or "*" in suffix:
                raise ValueError(
                    "fetch allowed_hosts wildcard must be '*.<suffix>' with a labeled suffix"
                )
            try:
                encoded_suffix = suffix.encode("idna").decode("ascii")
            except UnicodeError as exc:
                raise ValueError(
                    f"fetch allowed_hosts wildcard suffix is not valid IDNA: {suffix!r}"
                ) from exc
            normalized.append(f"*.{encoded_suffix}")
            continue
        if "*" in candidate:
            raise ValueError(
                "fetch allowed_hosts entries may only use a leading '*.' wildcard"
            )
        try:
            encoded = candidate.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError(
                f"fetch allowed_hosts entry is not valid IDNA: {candidate!r}"
            ) from exc
        normalized.append(encoded)
    return tuple(normalized)

def host_matches_allowed_hosts(
    host: str,
    *,
    allowed_hosts: tuple[str, ...],
) -> bool:
    """Return True when ``host`` matches an exact entry or ``*.suffix`` rule."""
    normalized_host = host.strip().lower().rstrip(".")
    for pattern in allowed_hosts:
        if pattern.startswith("*."):
            suffix = pattern[2:]
            if normalized_host == suffix or normalized_host.endswith(f".{suffix}"):
                return True
            continue
        if normalized_host == pattern:
            return True
    return False

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


if TYPE_CHECKING:
    from rag_core.fetching import FetchResponse

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
    "IPAddress",
    "normalize_fetch_hostname",
    "parse_fetch_ip_address",
    "validate_fetch_host",
    "validate_fetch_ip_address",
    "FetchScheme",
    "ValidatedFetchUrl",
    "host_matches_allowed_hosts",
    "normalize_allowed_hosts",
    "redact_fetch_url",
    "safe_remote_document_key",
    "safe_remote_event_url",
    "safe_remote_source_url",
    "validate_fetch_url_parts",
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
    "is_allowed_fetch_content_type",
    "validate_fetch_redirects",
    "validate_fetch_response_policy",
    "validate_fetch_url",
    "validate_resolved_fetch_addresses",
]
