from __future__ import annotations

from dataclasses import asdict
import hashlib
import ipaddress
from pathlib import Path

import pytest

from rag_core.documents.converters.format_support import FORMAT_SUPPORT_MATRIX
from rag_core.fetch_security import (
    FetchLimits,
    FetchSecurityPolicy,
    is_allowed_fetch_content_type,
    redact_fetch_url,
    validate_fetch_redirects,
    validate_fetch_url,
    validate_resolved_fetch_addresses,
)
from rag_core.fetch_security_url import safe_remote_event_url

ALLOW_HTTP_POLICY = FetchSecurityPolicy(allowed_schemes=("https", "http"))
ALLOW_HTTP_PRIVATE_POLICY = FetchSecurityPolicy(
    allowed_schemes=("https", "http"),
    allow_private_addresses=True,
)


def test_validate_fetch_url_normalizes_and_redacts_public_http_url() -> None:
    validated = validate_fetch_url(
        "HTTPS://Example.COM:443/docs/index.html?private=alpha#section"
    )

    assert validated.scheme == "https"
    assert validated.host == "example.com"
    assert validated.port == 443
    assert validated.path == "/docs/index.html"
    assert validated.has_query is True
    assert validated.query_sha256 == hashlib.sha256(b"private=alpha").hexdigest()
    assert validated.redacted_url == "https://example.com/docs/index.html?redacted"
    assert "private=alpha" not in repr(validated)
    assert "private=alpha" not in repr(asdict(validated))


def test_validate_fetch_url_canonicalizes_default_ports_for_identity() -> None:
    https = validate_fetch_url("https://example.com:443/docs?private=alpha")
    http = validate_fetch_url("http://example.com:80/docs", policy=ALLOW_HTTP_POLICY)
    non_default = validate_fetch_url("https://example.com:8443/docs")
    root = validate_fetch_url("https://example.com")

    assert https.port == 443
    assert https.redacted_url == "https://example.com/docs?redacted"
    assert http.port == 80
    assert http.redacted_url == "http://example.com/docs"
    assert non_default.redacted_url == "https://example.com:8443/docs"
    assert root.path == "/"
    assert root.redacted_url == "https://example.com/"


def test_validate_fetch_url_rejects_http_by_default() -> None:
    with pytest.raises(ValueError, match="HTTP requires explicit opt-in"):
        validate_fetch_url("http://example.com/docs")

    assert (
        validate_fetch_url("http://example.com/docs", policy=ALLOW_HTTP_POLICY).scheme
        == "http"
    )


@pytest.mark.parametrize(
    ("url", "path", "redacted_url"),
    [
        (
            "https://example.com/docs/./guide",
            "/docs/guide",
            "https://example.com/docs/guide",
        ),
        (
            "https://example.com/docs/archive/../guide?private=alpha",
            "/docs/guide",
            "https://example.com/docs/guide?redacted",
        ),
        (
            "https://example.com/docs/%2e/guide",
            "/docs/guide",
            "https://example.com/docs/guide",
        ),
        (
            "https://example.com/docs/archive/%2E%2e/guide",
            "/docs/guide",
            "https://example.com/docs/guide",
        ),
        (
            "https://example.com/docs/%2econfig/guide",
            "/docs/%2econfig/guide",
            "https://example.com/docs/%2econfig/guide",
        ),
    ],
)
def test_validate_fetch_url_canonicalizes_safe_path_dot_segments(
    url: str,
    path: str,
    redacted_url: str,
) -> None:
    validated = validate_fetch_url(url)

    assert validated.path == path
    assert validated.redacted_url == redacted_url


def test_safe_remote_event_url_omits_document_path() -> None:
    validated = validate_fetch_url("https://example.com/docs/guide?private=alpha")

    assert safe_remote_event_url(validated) == "https://example.com/?redacted"
    assert validated.redacted_url == "https://example.com/docs/guide?redacted"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "file:///tmp/doc.md",
        "https:///missing-host",
        "https://user:pass@example.com/docs",
        "https://localhost/docs",
        "https://docs.localhost/guide",
        "https://127.0.0.1/docs",
        "https://10.0.0.5/docs",
        "https://172.16.0.1/docs",
        "https://192.168.1.10/docs",
        "https://169.254.169.254/latest/meta-data",
        "https://::1/docs",
        "https://[::1]/docs",
        "https://[100:0:0:1::1]/docs",
        "https://[5f00::1]/docs",
        "https://[::ffff:127.0.0.1]/docs",
        "https://[64:ff9b::7f00:1]/docs",
        "https://[fec0::1]/docs",
        "https://2130706433/docs",
        "https://127.1/docs",
        "https://0x7f.0.0.1/docs",
    ],
)
def test_validate_fetch_url_rejects_unsafe_url_shapes(url: str) -> None:
    with pytest.raises(ValueError):
        validate_fetch_url(url)


def test_validate_fetch_url_can_explicitly_allow_private_addresses() -> None:
    validated = validate_fetch_url(
        "http://127.0.0.1:8000/docs",
        policy=ALLOW_HTTP_PRIVATE_POLICY,
    )

    assert validated.redacted_url == "http://127.0.0.1:8000/docs"
    assert validated.host == "127.0.0.1"
    assert validated.port == 8000

    localhost = validate_fetch_url(
        "http://localhost:8000/docs?private=alpha",
        policy=ALLOW_HTTP_PRIVATE_POLICY,
    )
    assert localhost.redacted_url == "http://localhost:8000/docs?redacted"
    assert localhost.host == "localhost"


def test_validate_resolved_fetch_addresses_blocks_private_dns_results() -> None:
    with pytest.raises(ValueError, match="not public"):
        validate_resolved_fetch_addresses("example.test", ["93.184.216.34", "10.0.0.1"])

    with pytest.raises(ValueError, match="not public"):
        validate_resolved_fetch_addresses("example.test", ["64:ff9b::7f00:1"])

    with pytest.raises(ValueError, match="not public"):
        validate_resolved_fetch_addresses("example.test", ["5f00::1"])


def test_validate_resolved_fetch_addresses_accepts_public_addresses() -> None:
    resolved = validate_resolved_fetch_addresses(
        "example.com",
        ["93.184.216.34", ipaddress.ip_address("2606:2800:220:1:248:1893:25c8:1946")],
    )

    assert [str(address) for address in resolved] == [
        "93.184.216.34",
        "2606:2800:220:1:248:1893:25c8:1946",
    ]


def test_validate_resolved_fetch_addresses_requires_dns_results() -> None:
    with pytest.raises(ValueError, match="resolved to no addresses"):
        validate_resolved_fetch_addresses("example.com", [])


def test_validate_fetch_redirects_checks_every_target_and_limit() -> None:
    urls = [
        "https://example.com/start",
        "https://docs.example.com/guide",
    ]

    assert [item.host for item in validate_fetch_redirects(urls)] == [
        "example.com",
        "docs.example.com",
    ]

    with pytest.raises(ValueError, match="max_redirects"):
        validate_fetch_redirects(urls, limits=FetchLimits(max_redirects=0))

    with pytest.raises(ValueError, match="not public"):
        validate_fetch_redirects(["https://example.com", "https://169.254.169.254"])


def test_fetch_limits_validate_positive_bounds_and_content_type_allowlist() -> None:
    limits = FetchLimits(
        max_bytes=1,
        timeout_seconds=0.5,
        max_redirects=0,
        allowed_content_types=("Text/*", "application/pdf"),
    )

    assert (
        is_allowed_fetch_content_type("text/html; charset=utf-8", limits=limits) is True
    )
    assert is_allowed_fetch_content_type("application/pdf", limits=limits) is True
    assert is_allowed_fetch_content_type("image/png", limits=limits) is False
    assert is_allowed_fetch_content_type(None, limits=limits) is False

    with pytest.raises(ValueError, match="max_bytes"):
        FetchLimits(max_bytes=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        FetchLimits(timeout_seconds=0)
    with pytest.raises(ValueError, match="max_redirects"):
        FetchLimits(max_redirects=-1)
    with pytest.raises(ValueError, match="allowed_content_types"):
        FetchLimits(allowed_content_types=())


def test_default_fetch_content_types_cover_registered_non_ocr_mime_types() -> None:
    limits = FetchLimits()
    for entry in FORMAT_SUPPORT_MATRIX:
        expected = entry.key != "image"
        for mime_type in entry.mime_types:
            assert (
                is_allowed_fetch_content_type(mime_type, limits=limits) is expected
            ), mime_type


def test_remote_fetch_docs_name_svg_text_converter_exception() -> None:
    docs = Path("docs/parsing/formats.md").read_text(encoding="utf-8")

    assert is_allowed_fetch_content_type("image/svg+xml") is True
    assert is_allowed_fetch_content_type("image/png") is False
    assert "SVG (`image/svg+xml`) is allowed because it routes through the text" in docs
    assert "OCR-required image MIME types remain excluded" in docs
    assert "Image MIME types remain excluded" not in docs


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf")])
def test_fetch_limits_reject_non_finite_timeouts(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        FetchLimits(timeout_seconds=timeout)


def test_redact_fetch_url_strips_credentials_query_and_fragment() -> None:
    assert (
        redact_fetch_url("https://user:pass@example.com/path?private=alpha#fragment")
        == "https://example.com/path?redacted"
    )
    assert (
        redact_fetch_url("https://example.com:443/path") == "https://example.com/path"
    )
    assert redact_fetch_url("http://example.com:80/path") == "http://example.com/path"
    assert redact_fetch_url("https://example.com") == "https://example.com/"
    assert (
        redact_fetch_url("https://example.com:8443/path")
        == "https://example.com:8443/path"
    )
    assert redact_fetch_url("not a url") == "<invalid-url>"
    assert (
        redact_fetch_url("https://example.com:bad/path?private=alpha")
        == "<invalid-url>"
    )
