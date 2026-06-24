from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

import rag_core.ingest.sources.remote as remote_sources_module
from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy, validate_fetch_url
from rag_core.fetch_security import safe_remote_document_key
from rag_core.fetching import FetchError, FetchResponse
from rag_core.ingest.sources.remote import RemoteUrlSourceReader, remote_source_document

ALLOW_HTTP_PRIVATE_POLICY = FetchSecurityPolicy(
    allowed_schemes=("https", "http"),
    allow_private_addresses=True,
)


@dataclass
class FakeFetchClient:
    response: FetchResponse
    seen_url: str = ""

    def fetch(self, url: str) -> FetchResponse:
        self.seen_url = url
        return self.response


def test_remote_url_source_reader_builds_parse_ready_document_without_raw_url() -> None:
    body = b"# Guide\n\nHello"
    response = _fetch_response(
        url="https://example.com/docs/guide?private=alpha",
        content_type="text/markdown",
        body=body,
    )
    fake_client = FakeFetchClient(response=response)

    document = RemoteUrlSourceReader(fetch_client=fake_client).read(
        " https://example.com/docs/guide?private=alpha "
    )

    assert fake_client.seen_url == " https://example.com/docs/guide?private=alpha "
    assert document.redacted_url == "https://example.com/docs/guide?redacted"
    assert document.document_key == _url_key(
        "https",
        "example.com",
        "/docs/guide",
        query="private=alpha",
    )
    assert document.filename == "guide.md"
    assert document.mime_type == "text/markdown"
    assert document.status_code == 200
    assert document.content_length == len(body)
    assert document.content_sha256 == hashlib.sha256(body).hexdigest()
    assert document.byte_count == len(body)
    assert document.redirect_count == 0
    assert document.file_bytes == body
    assert document.source_type == "url"
    assert "private=alpha" not in repr(document)
    assert "private=alpha" not in repr(document.to_payload())
    assert document.to_payload()["status_code"] == 200
    assert document.to_payload()["content_length"] == len(body)
    assert document.to_payload()["redirect_count"] == 0

    assert document.to_parse_kwargs() == {
        "file_bytes": body,
        "filename": "guide.md",
        "mime_type": "text/markdown",
    }
    assert document.to_source_metadata() == {
        "source_type": "url",
        "source_url": "https://example.com/docs/guide?redacted",
    }


def test_remote_url_source_reader_accepts_fetch_policy_and_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"# Guide\n\nHello"
    response = _fetch_response(
        url="https://example.com/docs/guide",
        content_type="text/markdown",
        body=body,
    )
    calls: list[dict[str, object]] = []

    class RecordingHttpFetchClient:
        def __init__(
            self,
            *,
            policy: FetchSecurityPolicy | None = None,
            limits: FetchLimits | None = None,
        ) -> None:
            calls.append({"policy": policy, "limits": limits})

        def fetch(self, url: str) -> FetchResponse:
            return response

    monkeypatch.setattr(
        remote_sources_module, "HttpFetchClient", RecordingHttpFetchClient
    )
    policy = ALLOW_HTTP_PRIVATE_POLICY
    limits = FetchLimits(max_bytes=1234, timeout_seconds=2.5, max_redirects=1)

    document = RemoteUrlSourceReader(policy=policy, limits=limits).read(
        "https://example.com/docs/guide"
    )

    assert document.redacted_url == "https://example.com/docs/guide"
    assert calls == [{"policy": policy, "limits": limits}]


def test_remote_url_source_reader_rejects_ambiguous_fetch_configuration() -> None:
    fake_client = FakeFetchClient(
        response=_fetch_response(
            url="https://example.com/docs/guide",
            content_type="text/markdown",
            body=b"# Guide",
        )
    )

    with pytest.raises(ValueError, match="fetch_client cannot be combined with limits"):
        RemoteUrlSourceReader(
            fetch_client=fake_client,
            limits=FetchLimits(max_bytes=1024),
        )


def test_remote_url_source_reader_rejects_policy_with_custom_fetch_client() -> None:
    body = b"# Guide\n\nHello"
    response = _fetch_response(
        url="http://127.0.0.1/docs/guide?private=alpha",
        content_type="text/markdown",
        body=body,
        policy=ALLOW_HTTP_PRIVATE_POLICY,
    )
    fake_client = FakeFetchClient(response=response)

    with pytest.raises(ValueError, match="HTTP requires explicit opt-in"):
        RemoteUrlSourceReader(fetch_client=fake_client).read(
            "https://example.com/docs/guide"
        )
    assert fake_client.seen_url == "https://example.com/docs/guide"

    with pytest.raises(ValueError, match="fetch_client cannot be combined with policy"):
        RemoteUrlSourceReader(
            fetch_client=fake_client,
            policy=ALLOW_HTTP_PRIVATE_POLICY,
        )


def test_remote_url_source_reader_validates_final_response_url() -> None:
    response = _fetch_response(
        url="http://127.0.0.1/docs/guide?private=alpha",
        content_type="text/markdown",
        body=b"# Guide\n\nHello",
        policy=ALLOW_HTTP_PRIVATE_POLICY,
    )
    fake_client = FakeFetchClient(response=response)

    with pytest.raises(ValueError, match="HTTP requires explicit opt-in"):
        RemoteUrlSourceReader(fetch_client=fake_client).read(
            "https://example.com/docs/guide"
        )

    assert fake_client.seen_url == "https://example.com/docs/guide"


def test_remote_url_source_reader_validates_redirect_chain_urls() -> None:
    response = _fetch_response(
        url="https://example.com/docs/final?token=beta",
        content_type="text/plain",
        body=b"ok",
        policy=ALLOW_HTTP_PRIVATE_POLICY,
        redirect_urls=(
            "https://example.com/docs/start?token=alpha",
            "http://127.0.0.1/docs/redirect",
            "https://example.com/docs/final?token=beta",
        ),
    )
    fake_client = FakeFetchClient(response=response)

    with pytest.raises(ValueError, match="HTTP requires explicit opt-in"):
        RemoteUrlSourceReader(fetch_client=fake_client).read(
            "https://example.com/docs/start?token=alpha"
        )

    assert fake_client.seen_url == "https://example.com/docs/start?token=alpha"


def test_remote_url_source_reader_revalidates_custom_fetch_client_response_shape() -> (
    None
):
    oversized = FakeFetchClient(
        response=_fetch_response(
            url="https://example.com/docs/guide",
            content_type="text/plain",
            body=b"x" * (FetchLimits().max_bytes + 1),
        )
    )
    with pytest.raises(FetchError, match="max_bytes"):
        RemoteUrlSourceReader(fetch_client=oversized).read(
            "https://example.com/docs/guide"
        )

    image = FakeFetchClient(
        response=_fetch_response(
            url="https://example.com/docs/guide",
            content_type="image/png",
            body=b"png",
        )
    )
    with pytest.raises(FetchError, match="content type"):
        RemoteUrlSourceReader(fetch_client=image).read("https://example.com/docs/guide")


def test_remote_url_source_reader_revalidates_custom_fetch_redirect_limit() -> None:
    chain = tuple(
        validate_fetch_url(f"https://example.com/r{index}") for index in range(7)
    )
    fetch_client = FakeFetchClient(
        response=FetchResponse(
            url=chain[-1],
            status_code=200,
            content_type="text/plain",
            content_length=2,
            content_sha256=hashlib.sha256(b"ok").hexdigest(),
            body=b"ok",
            redirect_chain=chain,
        )
    )

    with pytest.raises(FetchError, match="max_redirects"):
        RemoteUrlSourceReader(fetch_client=fetch_client).read("https://example.com/r0")


def test_remote_source_document_keeps_query_fingerprint_without_raw_query() -> None:
    first = remote_source_document(
        _fetch_response(
            url="https://example.com/export?id=1",
            content_type="application/json",
            body=b"{}",
        )
    )
    second = remote_source_document(
        _fetch_response(
            url="https://example.com/export?id=2",
            content_type="application/json",
            body=b"{}",
        )
    )

    assert first.redacted_url == "https://example.com/export?redacted"
    assert second.redacted_url == "https://example.com/export?redacted"
    assert first.document_key != second.document_key
    assert "id=1" not in first.document_key
    assert "id=2" not in second.document_key
    assert first.document_key == _url_key(
        "https", "example.com", "/export", query="id=1"
    )
    assert second.document_key == _url_key(
        "https", "example.com", "/export", query="id=2"
    )


def test_remote_source_document_redacted_url_preserves_canonical_path() -> None:
    guide = remote_source_document(
        _fetch_response(
            url="https://example.com/docs/guide?token=alpha",
            content_type="text/markdown",
            body=b"# Guide",
        )
    )
    pricing = remote_source_document(
        _fetch_response(
            url="https://example.com/docs/pricing?token=beta",
            content_type="text/markdown",
            body=b"# Pricing",
        )
    )

    assert guide.redacted_url == "https://example.com/docs/guide?redacted"
    assert pricing.redacted_url == "https://example.com/docs/pricing?redacted"
    assert guide.redacted_url != pricing.redacted_url
    assert "token=alpha" not in guide.redacted_url
    assert "token=beta" not in pricing.redacted_url


def test_remote_source_document_canonicalizes_dot_segment_url_aliases() -> None:
    first = remote_source_document(
        _fetch_response(
            url="https://example.com/docs/guide",
            content_type="text/markdown",
            body=b"# Guide",
        )
    )
    second = remote_source_document(
        _fetch_response(
            url="https://example.com/docs/./archive/%2e%2E/guide",
            content_type="text/markdown",
            body=b"# Guide",
        )
    )

    assert first.redacted_url == "https://example.com/docs/guide"
    assert second.redacted_url == first.redacted_url
    assert second.document_key == first.document_key


def test_remote_source_document_uses_final_url_identity_after_redirect() -> None:
    first = remote_source_document(
        _fetch_response(
            url="https://example.com/final",
            content_type="text/plain",
            body=b"ok",
            redirect_urls=("https://example.com/start?token=alpha",),
        )
    )
    second = remote_source_document(
        _fetch_response(
            url="https://example.com/final",
            content_type="text/plain",
            body=b"ok",
            redirect_urls=("https://example.com/start?token=beta",),
        )
    )

    assert first.redacted_url == second.redacted_url
    assert first.document_key == second.document_key
    assert first.document_key == _url_key("https", "example.com", "/final")
    assert first.requested_url == "https://example.com/start?redacted"
    assert second.requested_url == "https://example.com/start?redacted"
    assert first.to_source_metadata()["source_url"] == "https://example.com/final"
    assert (
        first.to_source_metadata()["source_requested_url"]
        == "https://example.com/start?redacted"
    )


def test_remote_source_document_records_redirect_count_without_raw_redirect_query() -> (
    None
):
    document = remote_source_document(
        _fetch_response(
            url="https://example.com/final?token=beta",
            content_type="text/plain",
            body=b"ok",
            redirect_urls=("https://example.com/start?token=alpha",),
        )
    )

    assert document.redirect_count == 1
    assert "token=alpha" not in repr(document.to_payload())
    assert "token=beta" not in repr(document.to_payload())


def test_remote_source_payload_hides_query_identity_hash_by_default() -> None:
    document = remote_source_document(
        _fetch_response(
            url="https://example.com/export?id=1",
            content_type="application/json",
            body=b"{}",
        )
    )

    public_payload = document.to_payload()
    private_payload = document.to_payload(include_private=True)

    assert public_payload["document_key"] == "url:https://example.com/export?redacted"
    assert public_payload["has_private_query_identity"] is True
    assert "query_sha256" not in repr(public_payload)
    assert private_payload["document_key"] == document.document_key


def test_fetch_security_remote_document_key_uses_source_key_shape() -> None:
    validated = validate_fetch_url("https://example.com/docs/guide?token=secret")

    assert safe_remote_document_key(validated) == _url_key(
        "https",
        "example.com",
        "/docs/guide",
        query="token=secret",
    )


def test_remote_source_document_uses_safe_filename_fallbacks() -> None:
    assert (
        remote_source_document(
            _fetch_response(
                url="https://example.com/",
                content_type="text/html",
                body=b"<h1>Docs</h1>",
            )
        ).filename
        == "index.html"
    )
    assert (
        remote_source_document(
            _fetch_response(
                url="https://example.com/downloads/release%20notes",
                content_type="application/pdf",
                body=b"%PDF",
            )
        ).filename
        == "release-notes.pdf"
    )
    assert (
        remote_source_document(
            _fetch_response(
                url="https://example.com/download/report.doc",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                body=b"docx",
            )
        ).filename
        == "report.docx"
    )
    assert (
        remote_source_document(
            _fetch_response(
                url="https://example.com/README.md",
                content_type="text/plain",
                body=b"readme",
            )
        ).filename
        == "README.md"
    )


def _fetch_response(
    *,
    url: str,
    content_type: str,
    body: bytes,
    policy: FetchSecurityPolicy | None = None,
    redirect_urls: tuple[str, ...] = (),
) -> FetchResponse:
    validated_url = validate_fetch_url(url, policy=policy)
    redirect_chain = tuple(
        validate_fetch_url(redirect_url, policy=policy)
        for redirect_url in redirect_urls
    ) + (validated_url,)
    return FetchResponse(
        url=validated_url,
        status_code=200,
        content_type=content_type,
        content_length=len(body),
        content_sha256=hashlib.sha256(body).hexdigest(),
        body=body,
        redirect_chain=redirect_chain,
    )


def _url_key(scheme: str, host: str, path: str, *, query: str | None = None) -> str:
    query_string = f"?{query}" if query is not None else ""
    return safe_remote_document_key(
        validate_fetch_url(
            f"{scheme}://{host}{path}{query_string}",
            policy=ALLOW_HTTP_PRIVATE_POLICY,
        )
    )
