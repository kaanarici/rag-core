from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pytest

from rag_core.cli import main
from rag_core.fetching import FetchError
from rag_core.remote_discovery import (
    RemoteDiscoveredUrl,
    RemoteDiscovery,
    parse_sitemap_urls,
)
from rag_core.remote_discovery_models import (
    DEFAULT_REMOTE_LLMS_TXT_MAX_URLS,
    DEFAULT_REMOTE_SITEMAP_INDEX_MAX_FETCHES,
    DEFAULT_REMOTE_SITEMAP_MAX_URLS,
)


@dataclass
class _FakeRemoteDiscoveryReader:
    sitemap_calls: list[dict[str, Any]] = field(default_factory=list)
    llms_txt_calls: list[dict[str, Any]] = field(default_factory=list)

    def read_sitemap(
        self,
        url: str,
        *,
        max_urls: int = DEFAULT_REMOTE_SITEMAP_MAX_URLS,
        max_sitemap_fetches: int = DEFAULT_REMOTE_SITEMAP_INDEX_MAX_FETCHES,
    ) -> RemoteDiscovery:
        self.sitemap_calls.append(
            {
                "url": url,
                "max_urls": max_urls,
                "max_sitemap_fetches": max_sitemap_fetches,
            }
        )
        return RemoteDiscovery(
            source_kind="sitemap",
            items=(
                RemoteDiscoveredUrl(
                    url="https://example.com/docs?private=alpha",
                    redacted_url="https://example.com/docs?redacted",
                    source_kind="sitemap",
                    query_sha256="query-hash",
                    lastmod="2026-05-17",
                ),
            ),
        )

    def read_llms_txt(
        self,
        url: str,
        *,
        max_urls: int = DEFAULT_REMOTE_LLMS_TXT_MAX_URLS,
    ) -> RemoteDiscovery:
        self.llms_txt_calls.append({"url": url, "max_urls": max_urls})
        return RemoteDiscovery(
            source_kind="llms_txt",
            items=(
                RemoteDiscoveredUrl(
                    url="https://example.com/guide?token=beta",
                    redacted_url="https://example.com/guide?redacted",
                    source_kind="llms_txt",
                    query_sha256="query-hash",
                    title="Guide",
                    section="Docs",
                    notes="first-run path",
                ),
            ),
        )


class _FailingRemoteDiscoveryReader:
    def __init__(self) -> None:
        self.sitemap_calls: list[dict[str, Any]] = []

    def read_sitemap(
        self,
        url: str,
        *,
        max_urls: int = DEFAULT_REMOTE_SITEMAP_MAX_URLS,
        max_sitemap_fetches: int = DEFAULT_REMOTE_SITEMAP_INDEX_MAX_FETCHES,
    ) -> RemoteDiscovery:
        self.sitemap_calls.append(
            {
                "url": url,
                "max_urls": max_urls,
                "max_sitemap_fetches": max_sitemap_fetches,
            }
        )
        raise FetchError(
            "fetch failed for https://example.com/sitemap.xml?redacted: HTTP 500"
        )

    def read_llms_txt(
        self,
        url: str,
        *,
        max_urls: int = DEFAULT_REMOTE_LLMS_TXT_MAX_URLS,
    ) -> RemoteDiscovery:
        raise AssertionError("llms.txt should not be read")


class _DuplicateSitemapRemoteDiscoveryReader:
    def read_sitemap(
        self,
        url: str,
        *,
        max_urls: int = DEFAULT_REMOTE_SITEMAP_MAX_URLS,
        max_sitemap_fetches: int = DEFAULT_REMOTE_SITEMAP_INDEX_MAX_FETCHES,
    ) -> RemoteDiscovery:
        del max_sitemap_fetches
        return parse_sitemap_urls(
            """<urlset>
              <url><loc>https://example.com/docs</loc></url>
              <url><loc>https://example.com:443/docs</loc></url>
              <url><loc>https://example.com</loc></url>
              <url><loc>https://example.com/</loc></url>
            </urlset>""",
            max_urls=max_urls,
        )

    def read_llms_txt(
        self,
        url: str,
        *,
        max_urls: int = DEFAULT_REMOTE_LLMS_TXT_MAX_URLS,
    ) -> RemoteDiscovery:
        raise AssertionError("llms.txt should not be read")


class _QuerySitemapRemoteDiscoveryReader:
    def read_sitemap(
        self,
        url: str,
        *,
        max_urls: int = DEFAULT_REMOTE_SITEMAP_MAX_URLS,
        max_sitemap_fetches: int = DEFAULT_REMOTE_SITEMAP_INDEX_MAX_FETCHES,
    ) -> RemoteDiscovery:
        del max_sitemap_fetches
        return parse_sitemap_urls(
            """<urlset>
              <url><loc>https://example.com/export?id=1</loc></url>
              <url><loc>https://example.com/export?id=2</loc></url>
            </urlset>""",
            max_urls=max_urls,
        )

    def read_llms_txt(
        self,
        url: str,
        *,
        max_urls: int = DEFAULT_REMOTE_LLMS_TXT_MAX_URLS,
    ) -> RemoteDiscovery:
        raise AssertionError("llms.txt should not be read")


class _ExplodingRemoteDiscoveryReader:
    def read_sitemap(
        self,
        url: str,
        *,
        max_urls: int = DEFAULT_REMOTE_SITEMAP_MAX_URLS,
        max_sitemap_fetches: int = DEFAULT_REMOTE_SITEMAP_INDEX_MAX_FETCHES,
    ) -> RemoteDiscovery:
        del max_urls, max_sitemap_fetches
        raise RuntimeError(f"parser exploded for {url}")

    def read_llms_txt(
        self,
        url: str,
        *,
        max_urls: int = DEFAULT_REMOTE_LLMS_TXT_MAX_URLS,
    ) -> RemoteDiscovery:
        del max_urls
        raise RuntimeError(f"parser exploded for {url}")


def test_discover_remote_sitemap_json_outputs_redacted_urls(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    reader = _FakeRemoteDiscoveryReader()
    monkeypatch.setattr(cli_module, "_remote_discovery_reader", lambda **_: reader)
    raw_url = "https://example.com/sitemap.xml?artifact=alpha"

    exit_code = main(
        [
            "discover-remote",
            raw_url,
            "--kind",
            "sitemap",
            "--max-urls",
            "5",
            "--max-sitemap-fetches",
            "7",
            "--json",
        ]
    )

    assert exit_code == 0
    assert reader.sitemap_calls == [
        {
            "url": raw_url,
            "max_urls": 5,
            "max_sitemap_fetches": 7,
        }
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_kind"] == "sitemap"
    assert payload["item_count"] == 1
    assert payload["items"][0]["url"] == "https://example.com/docs?redacted"
    assert "query_sha256" not in payload["items"][0]
    assert "private=alpha" not in repr(payload)
    assert "artifact=alpha" not in repr(payload)


def test_discover_remote_writes_url_file_without_printing_raw_urls(
    tmp_path: Any,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    reader = _FakeRemoteDiscoveryReader()
    monkeypatch.setattr(cli_module, "_remote_discovery_reader", lambda **_: reader)
    output_path = tmp_path / "urls.txt"

    exit_code = main(
        [
            "discover-remote",
            "https://example.com/sitemap.xml?artifact=alpha",
            "--kind",
            "sitemap",
            "--output-url-file",
            str(output_path),
            "--output-url-file-raw-queries",
            "--json",
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == (
        "https://example.com/docs?private=alpha\n"
    )
    if os.name != "nt":
        assert output_path.stat().st_mode & 0o777 == 0o600
    payload = json.loads(capsys.readouterr().out)
    assert payload["url_file"] == str(output_path)
    assert payload["items"][0]["url"] == "https://example.com/docs?redacted"
    assert "private=alpha" not in repr(payload)


def test_discover_remote_url_file_deduplicates_canonical_sitemap_entries(
    tmp_path: Any,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    reader = _DuplicateSitemapRemoteDiscoveryReader()
    monkeypatch.setattr(cli_module, "_remote_discovery_reader", lambda **_: reader)
    output_path = tmp_path / "urls.txt"

    exit_code = main(
        [
            "discover-remote",
            "https://example.com/sitemap.xml",
            "--kind",
            "sitemap",
            "--output-url-file",
            str(output_path),
            "--json",
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == (
        "https://example.com/docs\n"
        "https://example.com/\n"
    )
    capsys.readouterr()

    plan_exit_code = main(
        [
            "ingest-urls",
            str(output_path),
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--plan-json",
        ]
    )

    assert plan_exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["planned_count"] == 2
    document_keys = [item["document_key"] for item in payload["urls"]]
    assert document_keys == [
        "url:https://example.com/docs",
        "url:https://example.com/",
    ]


def test_discover_remote_url_file_refuses_lossy_redacted_query_sources(
    tmp_path: Any,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    reader = _QuerySitemapRemoteDiscoveryReader()
    monkeypatch.setattr(cli_module, "_remote_discovery_reader", lambda **_: reader)
    output_path = tmp_path / "urls.txt"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "discover-remote",
                "https://example.com/sitemap.xml",
                "--kind",
                "sitemap",
                "--output-url-file",
                str(output_path),
                "--json",
            ]
        )

    assert exc_info.value.code == 2
    assert not output_path.exists()
    error = capsys.readouterr().err
    assert "would omit 1 distinct query-bearing URL" in error
    assert "--output-url-file-raw-queries" in error
    assert "id=1" not in error
    assert "id=2" not in error


def test_discover_remote_url_file_can_opt_into_raw_query_sources(
    tmp_path: Any,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    reader = _QuerySitemapRemoteDiscoveryReader()
    monkeypatch.setattr(cli_module, "_remote_discovery_reader", lambda **_: reader)
    output_path = tmp_path / "urls.txt"

    exit_code = main(
        [
            "discover-remote",
            "https://example.com/sitemap.xml",
            "--kind",
            "sitemap",
            "--output-url-file",
            str(output_path),
            "--output-url-file-raw-queries",
            "--json",
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == (
        "https://example.com/export?id=1\n"
        "https://example.com/export?id=2\n"
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["items"][0]["url"] == "https://example.com/export?redacted"
    assert "id=1" not in repr(payload)


def test_discover_remote_raw_query_flag_requires_output_url_file(capsys: Any) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "discover-remote",
                "https://example.com/sitemap.xml",
                "--kind",
                "sitemap",
                "--output-url-file-raw-queries",
            ]
        )

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "--output-url-file-raw-queries requires --output-url-file" in error


def test_discover_remote_refuses_to_overwrite_url_file(
    tmp_path: Any,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    reader = _DuplicateSitemapRemoteDiscoveryReader()
    monkeypatch.setattr(cli_module, "_remote_discovery_reader", lambda **_: reader)
    output_path = tmp_path / "urls.txt"
    output_path.write_text("https://example.com/existing\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "discover-remote",
                "https://example.com/sitemap.xml",
                "--kind",
                "sitemap",
                "--output-url-file",
                str(output_path),
            ]
        )

    assert exc_info.value.code == 2
    assert output_path.read_text(encoding="utf-8") == "https://example.com/existing\n"
    error = capsys.readouterr().err
    assert "already exists" in error


def test_discover_remote_normalizes_unwritable_url_file_path(
    tmp_path: Any,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    reader = _DuplicateSitemapRemoteDiscoveryReader()
    monkeypatch.setattr(cli_module, "_remote_discovery_reader", lambda **_: reader)
    parent_file = tmp_path / "parent-file"
    parent_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "discover-remote",
                "https://example.com/sitemap.xml",
                "--kind",
                "sitemap",
                "--output-url-file",
                str(parent_file / "urls.txt"),
            ]
        )

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "URL output file is not writable" in error
    assert "Traceback" not in error


def test_discover_remote_llms_txt_text_output_uses_redacted_urls(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    reader = _FakeRemoteDiscoveryReader()
    monkeypatch.setattr(cli_module, "_remote_discovery_reader", lambda **_: reader)

    exit_code = main(
        [
            "discover-remote",
            "https://example.com/llms.txt?artifact=alpha",
            "--kind",
            "llms-txt",
        ]
    )

    assert exit_code == 0
    assert reader.llms_txt_calls == [
        {
            "url": "https://example.com/llms.txt?artifact=alpha",
            "max_urls": DEFAULT_REMOTE_LLMS_TXT_MAX_URLS,
        }
    ]
    output = capsys.readouterr().out
    assert "Discovery: llms-txt (1 URLs)" in output
    assert "https://example.com/guide?redacted [Guide] (Docs)" in output
    assert "token=beta" not in output
    assert "artifact=alpha" not in output


def test_discover_remote_fetch_error_exits_without_traceback_or_raw_query(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    reader = _FailingRemoteDiscoveryReader()
    monkeypatch.setattr(cli_module, "_remote_discovery_reader", lambda **_: reader)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "discover-remote",
                "https://example.com/sitemap.xml?artifact=alpha",
                "--kind",
                "sitemap",
            ]
        )

    assert exc_info.value.code == 2
    assert reader.sitemap_calls == [
        {
            "url": "https://example.com/sitemap.xml?artifact=alpha",
            "max_urls": DEFAULT_REMOTE_SITEMAP_MAX_URLS,
            "max_sitemap_fetches": DEFAULT_REMOTE_SITEMAP_INDEX_MAX_FETCHES,
        }
    ]
    error = capsys.readouterr().err
    assert (
        "rag-core: error: fetch failed for https://example.com/sitemap.xml?redacted"
        in error
    )
    assert "artifact=alpha" not in error
    assert "Traceback" not in error


def test_discover_remote_unexpected_reader_error_is_sanitized(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "_remote_discovery_reader",
        lambda **_: _ExplodingRemoteDiscoveryReader(),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "discover-remote",
                "https://example.com/sitemap.xml?artifact=alpha",
                "--kind",
                "sitemap",
            ]
        )

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "remote discovery failed" in error
    assert "artifact=alpha" not in error
    assert "Traceback" not in error


def test_discover_remote_rejects_invalid_url_before_reader_setup(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    def fail_reader(**_: Any) -> _FakeRemoteDiscoveryReader:
        raise AssertionError("reader should not be constructed for invalid URLs")

    monkeypatch.setattr(cli_module, "_remote_discovery_reader", fail_reader)

    with pytest.raises(SystemExit) as exc_info:
        main(["discover-remote", "file:///tmp/docs.xml", "--kind", "sitemap"])

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "unsupported fetch URL scheme" in error
    assert "Traceback" not in error


def test_discover_remote_private_address_requires_explicit_fetch_opt_in(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    reader = _FakeRemoteDiscoveryReader()
    captured: dict[str, Any] = {}

    def reader_factory(**kwargs: Any) -> _FakeRemoteDiscoveryReader:
        captured.update(kwargs)
        return reader

    monkeypatch.setattr(cli_module, "_remote_discovery_reader", reader_factory)
    private_url = "http://localhost:8000/sitemap.xml?artifact=alpha"

    with pytest.raises(SystemExit) as exc_info:
        main(["discover-remote", private_url, "--kind", "sitemap"])

    assert exc_info.value.code == 2
    assert reader.sitemap_calls == []
    error = capsys.readouterr().err
    assert "HTTP requires explicit opt-in" in error
    assert "artifact=alpha" not in error

    exit_code = main(
        [
            "discover-remote",
            private_url,
            "--kind",
            "sitemap",
            "--fetch-allow-http",
            "--fetch-allow-private-addresses",
            "--fetch-max-bytes",
            "123",
            "--fetch-timeout-seconds",
            "0.5",
            "--fetch-max-redirects",
            "0",
        ]
    )

    assert exit_code == 0
    assert reader.sitemap_calls == [
        {
            "url": private_url,
            "max_urls": DEFAULT_REMOTE_SITEMAP_MAX_URLS,
            "max_sitemap_fetches": DEFAULT_REMOTE_SITEMAP_INDEX_MAX_FETCHES,
        }
    ]
    assert captured["policy"].allowed_schemes == ("https", "http")
    assert captured["policy"].allow_private_addresses is True
    assert captured["limits"].max_bytes == 123
    assert captured["limits"].timeout_seconds == 0.5
    assert captured["limits"].max_redirects == 0


def test_discover_remote_rejects_non_positive_max_urls(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    reader = _FakeRemoteDiscoveryReader()
    monkeypatch.setattr(cli_module, "_remote_discovery_reader", lambda **_: reader)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "discover-remote",
                "https://example.com/sitemap.xml",
                "--kind",
                "sitemap",
                "--max-urls",
                "0",
            ]
        )

    assert exc_info.value.code == 2
    assert reader.sitemap_calls == []
    error = capsys.readouterr().err
    assert "--max-urls must be positive" in error
    assert "Traceback" not in error

    with pytest.raises(SystemExit) as sitemap_exc_info:
        main(
            [
                "discover-remote",
                "https://example.com/sitemap.xml",
                "--kind",
                "sitemap",
                "--max-sitemap-fetches",
                "0",
            ]
        )

    assert sitemap_exc_info.value.code == 2
    assert reader.sitemap_calls == []
    error = capsys.readouterr().err
    assert "--max-sitemap-fetches must be positive" in error
    assert "Traceback" not in error
