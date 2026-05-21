from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import rag_core.remote_discovery as remote_discovery_module
from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy, validate_fetch_url
from rag_core.fetching import FetchError, FetchResponse
from rag_core.remote_discovery import (
    RemoteDiscoveryReader,
    parse_llms_txt_urls,
    parse_sitemap_urls,
    write_discovered_url_file,
    write_raw_discovered_url_file,
)

ALLOW_HTTP_PRIVATE_POLICY = FetchSecurityPolicy(
    allowed_schemes=("https", "http"),
    allow_private_addresses=True,
)


@dataclass
class _FakeFetchClient:
    response: FetchResponse
    calls: list[str] = field(default_factory=list)

    def fetch(self, url: str) -> FetchResponse:
        self.calls.append(url)
        return self.response


@dataclass
class _MapFetchClient:
    responses: dict[str, FetchResponse]
    calls: list[str] = field(default_factory=list)

    def fetch(self, url: str) -> FetchResponse:
        self.calls.append(url)
        return self.responses[url]


def test_parse_sitemap_urls_extracts_redacted_page_urls() -> None:
    sitemap = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://example.com/docs?private=alpha&amp;page=1</loc>
        <lastmod>2026-05-01</lastmod>
      </url>
      <url>
        <loc>https://example.com/guide</loc>
      </url>
    </urlset>
    """

    result = parse_sitemap_urls(sitemap)

    assert result.source_kind == "sitemap"
    assert result.item_count == 2
    assert result.urls == (
        "https://example.com/docs?private=alpha&page=1",
        "https://example.com/guide",
    )
    assert result.redacted_urls == (
        "https://example.com/docs?redacted",
        "https://example.com/guide",
    )
    first = result.items[0]
    assert first.url == "https://example.com/docs?private=alpha&page=1"
    assert first.redacted_url == "https://example.com/docs?redacted"
    assert first.query_sha256 == hashlib.sha256(
        b"private=alpha&page=1"
    ).hexdigest()
    assert first.lastmod == "2026-05-01"
    assert first.to_payload()["url"] == "https://example.com/docs?redacted"
    assert "private=alpha" not in repr(first)
    assert "private=alpha" not in repr(result.to_payload())


def test_parse_sitemap_urls_deduplicates_canonical_source_identities() -> None:
    sitemap = """<urlset>
      <url><loc>https://example.com/docs</loc></url>
      <url><loc>https://example.com:443/docs</loc></url>
      <url><loc>https://example.com</loc></url>
      <url><loc>https://example.com/</loc></url>
      <url><loc>https://example.com/docs?private=alpha</loc></url>
      <url><loc>https://example.com/docs?private=beta</loc></url>
    </urlset>"""

    result = parse_sitemap_urls(sitemap)

    assert [item.redacted_url for item in result.items] == [
        "https://example.com/docs",
        "https://example.com/",
        "https://example.com/docs?redacted",
        "https://example.com/docs?redacted",
    ]
    assert result.items[2].query_sha256 != result.items[3].query_sha256


def test_parse_sitemap_index_extracts_sitemap_urls() -> None:
    sitemap_index = """<?xml version="1.0" encoding="UTF-8"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap>
        <loc>https://example.com/sitemap-docs.xml?token=alpha</loc>
        <lastmod>2026-05-02</lastmod>
      </sitemap>
    </sitemapindex>
    """

    result = parse_sitemap_urls(sitemap_index)

    assert result.source_kind == "sitemap_index"
    assert result.item_count == 1
    item = result.items[0]
    assert item.source_kind == "sitemap_index"
    assert item.redacted_url == "https://example.com/sitemap-docs.xml?redacted"
    assert item.lastmod == "2026-05-02"
    assert "token=alpha" not in repr(item.to_payload())


def test_parse_sitemap_urls_accepts_encoded_xml_bytes() -> None:
    sitemap = (
        "<?xml version='1.0' encoding='UTF-16'?>"
        "<urlset><url><loc>https://example.com/encoded</loc></url></urlset>"
    ).encode("utf-16")

    result = parse_sitemap_urls(sitemap)

    assert result.item_count == 1
    assert result.items[0].redacted_url == "https://example.com/encoded"


def test_parse_sitemap_urls_rejects_unsafe_xml_and_invalid_entries() -> None:
    with pytest.raises(ValueError, match="DTD or entity"):
        parse_sitemap_urls(
            "<!DOCTYPE sitemap [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>"
            "<urlset><url><loc>&xxe;</loc></url></urlset>"
        )
    with pytest.raises(ValueError, match="DTD or entity"):
        parse_sitemap_urls(
            (
                "<!DOCTYPE sitemap [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>"
                "<urlset><url><loc>&xxe;</loc></url></urlset>"
            ).encode("utf-16")
        )

    with pytest.raises(ValueError, match="missing <loc>"):
        parse_sitemap_urls("<urlset><url><lastmod>2026-05-01</lastmod></url></urlset>")

    with pytest.raises(ValueError, match="HTTP requires explicit opt-in"):
        parse_sitemap_urls("<urlset><url><loc>http://127.0.0.1/a</loc></url></urlset>")

    with pytest.raises(ValueError, match="supported Unicode"):
        parse_sitemap_urls(b"\xff\xfe\xff")

    allowed = parse_sitemap_urls(
        "<urlset><url><loc>http://127.0.0.1/a?secret=alpha</loc></url></urlset>",
        policy=ALLOW_HTTP_PRIVATE_POLICY,
    )
    assert allowed.items[0].redacted_url == "http://127.0.0.1/a?redacted"


def test_parse_sitemap_urls_enforces_max_urls() -> None:
    sitemap = """<urlset>
      <url><loc>https://example.com/a</loc></url>
      <url><loc>https://example.com/b</loc></url>
    </urlset>"""

    with pytest.raises(ValueError, match="max_urls"):
        parse_sitemap_urls(sitemap, max_urls=1)


def test_parse_sitemap_urls_enforces_max_urls_after_deduplication() -> None:
    sitemap = """<urlset>
      <url><loc>https://example.com/a</loc></url>
      <url><loc>https://example.com:443/a</loc></url>
      <url><loc>https://example.com/b</loc></url>
    </urlset>"""

    with pytest.raises(ValueError, match="max_urls"):
        parse_sitemap_urls(sitemap, max_urls=1)


def test_parse_llms_txt_urls_extracts_sections_and_optional_links() -> None:
    llms_txt = """# Example Docs

> Short guide.

## Docs

- [Quickstart](/docs/start?private=alpha): first-run path
- [Quickstart Duplicate](/docs/start?private=alpha): duplicate ignored
- [API](https://example.com/api)

## Optional

- [Archive](/archive): older docs
"""

    result = parse_llms_txt_urls(llms_txt, base_url="https://example.com/llms.txt")

    assert result.source_kind == "llms_txt"
    assert result.item_count == 3
    quickstart, api, archive = result.items
    assert quickstart.url == "https://example.com/docs/start?private=alpha"
    assert quickstart.redacted_url == "https://example.com/docs/start?redacted"
    assert quickstart.title == "Quickstart"
    assert quickstart.section == "Docs"
    assert quickstart.notes == "first-run path"
    assert quickstart.optional is False
    assert api.redacted_url == "https://example.com/api"
    assert archive.section == "Optional"
    assert archive.optional is True
    assert "private=alpha" not in repr(result)
    assert "private=alpha" not in repr(result.to_payload())


def test_parse_llms_txt_urls_deduplicates_canonical_source_identities() -> None:
    llms_txt = """## Docs
- [Root](https://example.com)
- [Root Slash](https://example.com/)
- [Root Port](https://example.com:443/)
- [Export Alpha](https://example.com/export?id=1)
- [Export Beta](https://example.com/export?id=2)
"""

    result = parse_llms_txt_urls(llms_txt, base_url="https://example.com/llms.txt")

    assert [item.redacted_url for item in result.items] == [
        "https://example.com/",
        "https://example.com/export?redacted",
        "https://example.com/export?redacted",
    ]
    assert result.items[1].query_sha256 != result.items[2].query_sha256


def test_parse_llms_txt_urls_validates_links_and_limits() -> None:
    with pytest.raises(ValueError, match="unsupported fetch URL scheme"):
        parse_llms_txt_urls(
            "- [Email](mailto:support@example.com)",
            base_url="https://example.com/llms.txt",
        )

    with pytest.raises(ValueError, match="max_urls"):
        parse_llms_txt_urls(
            "- [A](/a)\n- [B](/b)",
            base_url="https://example.com/llms.txt",
            max_urls=1,
        )


def test_remote_discovery_reader_fetches_one_sitemap_artifact() -> None:
    body = (
        "<urlset>"
        "<url><loc>https://example.com/a?secret=alpha</loc></url>"
        "<url><loc>https://example.com/b</loc></url>"
        "</urlset>"
    ).encode()
    client = _FakeFetchClient(
        _fetch_response("https://example.com/sitemap.xml?artifact=token", body)
    )

    result = RemoteDiscoveryReader(fetch_client=client).read_sitemap(
        "https://example.com/sitemap.xml?artifact=token"
    )

    assert client.calls == ["https://example.com/sitemap.xml?artifact=token"]
    assert result.source_kind == "sitemap"
    assert [item.redacted_url for item in result.items] == [
        "https://example.com/a?redacted",
        "https://example.com/b",
    ]
    assert "secret=alpha" not in repr(result.to_payload())


def test_remote_discovery_reader_revalidates_custom_fetch_response_shape() -> None:
    client = _FakeFetchClient(
        _fetch_response(
            "https://example.com/sitemap.xml",
            b"x" * (FetchLimits().max_bytes + 1),
        )
    )

    with pytest.raises(FetchError, match="max_bytes"):
        RemoteDiscoveryReader(fetch_client=client).read_sitemap(
            "https://example.com/sitemap.xml"
        )


def test_remote_discovery_reader_accepts_fetch_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"<urlset><url><loc>https://example.com/a</loc></url></urlset>"
    response = _fetch_response("https://example.com/sitemap.xml", body)
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
        remote_discovery_module,
        "HttpFetchClient",
        RecordingHttpFetchClient,
    )
    policy = ALLOW_HTTP_PRIVATE_POLICY
    limits = FetchLimits(max_bytes=1234, timeout_seconds=2.5, max_redirects=1)

    result = RemoteDiscoveryReader(policy=policy, limits=limits).read_sitemap(
        "https://example.com/sitemap.xml"
    )

    assert result.item_count == 1
    assert calls == [{"policy": policy, "limits": limits}]


def test_remote_discovery_reader_rejects_custom_client_with_fetch_limits() -> None:
    client = _FakeFetchClient(
        _fetch_response(
            "https://example.com/sitemap.xml",
            b"<urlset />",
        )
    )

    with pytest.raises(ValueError, match="fetch_client cannot be combined"):
        RemoteDiscoveryReader(fetch_client=client, limits=FetchLimits())


def test_remote_discovery_reader_expands_sitemap_indexes_to_page_urls() -> None:
    index_body = (
        "<sitemapindex>"
        "<sitemap><loc>https://example.com/nested.xml?secret=alpha</loc></sitemap>"
        "<sitemap><loc>https://example.com/extra.xml</loc></sitemap>"
        "</sitemapindex>"
    ).encode()
    nested_body = (
        "<urlset>"
        "<url><loc>https://example.com/a?private=one</loc></url>"
        "<url><loc>https://example.com/a?private=one</loc></url>"
        "</urlset>"
    ).encode()
    extra_body = (
        "<urlset><url><loc>https://example.com/b</loc></url></urlset>"
    ).encode()
    client = _MapFetchClient(
        {
            "https://example.com/sitemap-index.xml": _fetch_response(
                "https://example.com/sitemap-index.xml",
                index_body,
            ),
            "https://example.com/nested.xml?secret=alpha": _fetch_response(
                "https://example.com/nested.xml?secret=alpha",
                nested_body,
            ),
            "https://example.com/extra.xml": _fetch_response(
                "https://example.com/extra.xml",
                extra_body,
            ),
        }
    )

    result = RemoteDiscoveryReader(fetch_client=client).read_sitemap(
        "https://example.com/sitemap-index.xml"
    )

    assert client.calls == [
        "https://example.com/sitemap-index.xml",
        "https://example.com/nested.xml?secret=alpha",
        "https://example.com/extra.xml",
    ]
    assert result.source_kind == "sitemap"
    assert [item.redacted_url for item in result.items] == [
        "https://example.com/a?redacted",
        "https://example.com/b",
    ]
    assert "private=one" not in repr(result.to_payload())
    assert "secret=alpha" not in repr(result.to_payload())


def test_remote_discovery_reader_bounds_sitemap_index_expansion() -> None:
    index_body = (
        "<sitemapindex>"
        "<sitemap><loc>https://example.com/one.xml</loc></sitemap>"
        "<sitemap><loc>https://example.com/two.xml</loc></sitemap>"
        "</sitemapindex>"
    ).encode()
    client = _MapFetchClient(
        {
            "https://example.com/sitemap-index.xml": _fetch_response(
                "https://example.com/sitemap-index.xml",
                index_body,
            ),
            "https://example.com/one.xml": _fetch_response(
                "https://example.com/one.xml",
                b"<urlset />",
            ),
            "https://example.com/two.xml": _fetch_response(
                "https://example.com/two.xml",
                b"<urlset />",
            ),
        }
    )

    with pytest.raises(ValueError, match="max_sitemap_fetches"):
        RemoteDiscoveryReader(fetch_client=client).read_sitemap(
            "https://example.com/sitemap-index.xml",
            max_sitemap_fetches=1,
        )

    assert client.calls == [
        "https://example.com/sitemap-index.xml",
        "https://example.com/one.xml",
    ]


def test_remote_discovery_reader_resolves_llms_txt_links_from_final_url() -> None:
    body = b"## Docs\n- [Guide](guide?token=beta): relative child\n"
    client = _FakeFetchClient(
        _fetch_response(
            "https://example.com/docs/llms.txt?artifact=token",
            body,
            content_type="text/plain",
        )
    )

    result = RemoteDiscoveryReader(fetch_client=client).read_llms_txt(
        "https://example.com/llms.txt?artifact=token"
    )

    assert client.calls == ["https://example.com/llms.txt?artifact=token"]
    item = result.items[0]
    assert item.url == "https://example.com/docs/guide?token=beta"
    assert item.redacted_url == "https://example.com/docs/guide?redacted"
    assert item.notes == "relative child"
    assert "artifact=token" not in repr(result.to_payload())


def test_remote_discovery_reader_validates_artifact_request_url() -> None:
    client = _FakeFetchClient(
        _fetch_response(
            "http://127.0.0.1/llms.txt",
            b"## Docs\n- [Public](https://example.com/docs)\n",
            content_type="text/plain",
            policy=ALLOW_HTTP_PRIVATE_POLICY,
        )
    )

    with pytest.raises(ValueError, match="HTTP requires explicit opt-in"):
        RemoteDiscoveryReader(fetch_client=client).read_llms_txt(
            "http://127.0.0.1/llms.txt"
        )

    assert client.calls == []


def test_remote_discovery_reader_validates_final_response_url() -> None:
    client = _FakeFetchClient(
        _fetch_response(
            "http://127.0.0.1/llms.txt",
            b"## Docs\n- [Public](https://example.com/docs)\n",
            content_type="text/plain",
            policy=ALLOW_HTTP_PRIVATE_POLICY,
        )
    )

    with pytest.raises(ValueError, match="HTTP requires explicit opt-in"):
        RemoteDiscoveryReader(fetch_client=client).read_llms_txt(
            "https://example.com/llms.txt"
        )

    assert client.calls == ["https://example.com/llms.txt"]


def test_remote_discovery_reader_validates_redirect_chain_urls() -> None:
    client = _FakeFetchClient(
        _fetch_response(
            "https://example.com/llms.txt",
            b"## Docs\n- [Public](https://example.com/docs)\n",
            content_type="text/plain",
            policy=ALLOW_HTTP_PRIVATE_POLICY,
            redirect_chain=(
                "https://example.com/start",
                "http://127.0.0.1/redirect",
                "https://example.com/llms.txt",
            ),
        )
    )

    with pytest.raises(ValueError, match="HTTP requires explicit opt-in"):
        RemoteDiscoveryReader(fetch_client=client).read_llms_txt(
            "https://example.com/start"
        )

    assert client.calls == ["https://example.com/start"]


def test_remote_discovery_reader_policy_controls_discovered_urls() -> None:
    policy = ALLOW_HTTP_PRIVATE_POLICY
    body = b"## Internal\n- [Local](http://127.0.0.1/docs?secret=alpha)\n"
    client = _FakeFetchClient(
        _fetch_response(
            "http://127.0.0.1/llms.txt",
            body,
            content_type="text/plain",
            policy=policy,
        )
    )

    with pytest.raises(ValueError, match="HTTP requires explicit opt-in"):
        RemoteDiscoveryReader(fetch_client=client).read_llms_txt(
            "http://127.0.0.1/llms.txt"
        )

    client.calls.clear()
    result = RemoteDiscoveryReader(fetch_client=client, policy=policy).read_llms_txt(
        "http://127.0.0.1/llms.txt"
    )

    assert client.calls == ["http://127.0.0.1/llms.txt"]
    assert result.items[0].redacted_url == "http://127.0.0.1/docs?redacted"


def test_write_discovered_url_file_writes_redacted_urls_by_default(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "private" / "urls.txt"
    discovery = parse_llms_txt_urls(
        "- [Guide](/guide)\n"
        "- [Guide Copy](/guide)\n"
        "- [API](/api)\n",
        base_url="https://example.com/llms.txt",
    )

    written_path = write_discovered_url_file(discovery, output_path)

    assert written_path == output_path
    assert output_path.read_text(encoding="utf-8") == (
        "https://example.com/guide\n"
        "https://example.com/api\n"
    )
    if os.name != "nt":
        assert output_path.stat().st_mode & 0o777 == 0o600
        assert (tmp_path / "private").stat().st_mode & 0o777 == 0o700

    with pytest.raises(ValueError, match="already exists"):
        write_discovered_url_file(discovery, output_path)


def test_write_discovered_url_file_rejects_query_redacted_fetch_targets(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "private" / "urls.txt"
    discovery = parse_llms_txt_urls(
        "- [One](/export?id=1)\n"
        "- [Two](/export?id=2)\n",
        base_url="https://example.com/llms.txt",
    )

    with pytest.raises(ValueError, match="query-bearing URL"):
        write_discovered_url_file(discovery, output_path)

    assert not output_path.exists()


def test_write_raw_discovered_url_file_preserves_query_strings(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "private" / "urls.txt"
    discovery = parse_llms_txt_urls(
        "- [Guide](/guide?private=alpha)\n",
        base_url="https://example.com/llms.txt",
    )

    written_path = write_raw_discovered_url_file(discovery, output_path)

    assert written_path == output_path
    assert output_path.read_text(encoding="utf-8") == (
        "https://example.com/guide?private=alpha\n"
    )
    with pytest.raises(ValueError, match="already exists"):
        write_raw_discovered_url_file(discovery, output_path)


def test_write_discovered_url_file_normalizes_unwritable_parent_path(
    tmp_path: Path,
) -> None:
    parent_file = tmp_path / "parent-file"
    parent_file.write_text("not a directory", encoding="utf-8")
    discovery = parse_llms_txt_urls(
        "- [Guide](/guide)\n",
        base_url="https://example.com/llms.txt",
    )

    with pytest.raises(ValueError, match="not writable"):
        write_discovered_url_file(discovery, parent_file / "urls.txt")


def _fetch_response(
    url: str,
    body: bytes,
    *,
    content_type: str = "application/xml",
    policy: FetchSecurityPolicy | None = None,
    redirect_chain: tuple[str, ...] | None = None,
) -> FetchResponse:
    validated_url = validate_fetch_url(url, policy=policy)
    chain_urls = redirect_chain or (url,)
    return FetchResponse(
        url=validated_url,
        status_code=200,
        content_type=content_type,
        content_length=len(body),
        content_sha256=hashlib.sha256(body).hexdigest(),
        body=body,
        redirect_chain=tuple(
            validate_fetch_url(chain_url, policy=policy) for chain_url in chain_urls
        ),
    )
