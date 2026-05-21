from __future__ import annotations

from pathlib import Path

from rag_core.fetching import FetchClient, FetchResponse, HttpFetchClient
from rag_core.fetching_response import validate_fetch_response
from rag_core.fetch_security import (
    FetchLimits,
    FetchSecurityPolicy,
    validate_fetch_url,
)
from rag_core.private_files import write_private_text_exclusive
from rag_core.remote_discovery_documents import (
    parse_llms_txt_urls as parse_llms_txt_urls,
    parse_sitemap_urls as parse_sitemap_urls,
)
from rag_core.remote_discovery_models import (
    RemoteDiscoveredUrl as RemoteDiscoveredUrl,
    RemoteDiscovery as RemoteDiscovery,
    RemoteDiscoveryKind as RemoteDiscoveryKind,
)
from rag_core.remote_discovery_policy import (
    response_base_url,
    validate_fetch_response_policy,
)
from rag_core.remote_discovery_sitemaps import expand_sitemap_index


class RemoteDiscoveryReader:
    def __init__(
        self,
        *,
        fetch_client: FetchClient | None = None,
        policy: FetchSecurityPolicy | None = None,
        limits: FetchLimits | None = None,
    ) -> None:
        if fetch_client is not None and limits is not None:
            raise ValueError("fetch_client cannot be combined with limits")
        self._limits = limits or FetchLimits()
        self._fetch_client = fetch_client or HttpFetchClient(
            policy=policy,
            limits=limits,
        )
        self._policy = policy

    def read_sitemap(
        self,
        url: str,
        *,
        max_urls: int = 50_000,
        max_sitemap_fetches: int = 128,
    ) -> RemoteDiscovery:
        response = self._fetch(url)
        discovery = parse_sitemap_urls(
            response.body,
            policy=self._policy,
            max_urls=max_urls,
        )
        if discovery.source_kind != "sitemap_index":
            return discovery
        return self._expand_sitemap_index(
            discovery,
            max_urls=max_urls,
            max_sitemap_fetches=max_sitemap_fetches,
        )

    def read_llms_txt(self, url: str, *, max_urls: int = 1_000) -> RemoteDiscovery:
        response = self._fetch(url)
        return parse_llms_txt_urls(
            response.body,
            base_url=response_base_url(response),
            policy=self._policy,
            max_urls=max_urls,
        )

    def _fetch(self, url: str) -> FetchResponse:
        validate_fetch_url(url, policy=self._policy)
        response = self._fetch_client.fetch(url)
        validate_fetch_response(response, limits=self._limits)
        validate_fetch_response_policy(response, policy=self._policy)
        return response

    def _expand_sitemap_index(
        self,
        discovery: RemoteDiscovery,
        *,
        max_urls: int,
        max_sitemap_fetches: int,
    ) -> RemoteDiscovery:
        return expand_sitemap_index(
            discovery,
            fetch_sitemap_body=lambda url: self._fetch(url).body,
            policy=self._policy,
            max_urls=max_urls,
            max_sitemap_fetches=max_sitemap_fetches,
        )


def write_discovered_url_file(
    discovery: RemoteDiscovery,
    path: str | Path,
) -> Path:
    return write_redacted_discovered_url_file(
        list(redacted_url_file_lines(discovery)),
        path,
    )


def write_raw_discovered_url_file(
    discovery: RemoteDiscovery,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    lines = "".join(f"{item.url}\n" for item in discovery.items)
    try:
        write_private_text_exclusive(output_path, lines, reject_symlink=True)
    except OSError as exc:
        if isinstance(exc, FileExistsError):
            raise ValueError(f"URL output file already exists: {str(output_path)!r}") from exc
        raise ValueError(f"URL output file is not writable: {str(output_path)!r}") from exc
    except ValueError as exc:
        if "symlink" in str(exc):
            raise ValueError(f"URL output file is not writable: {str(output_path)!r}") from exc
        raise
    return output_path


def write_redacted_discovered_url_file(
    redacted_urls: list[str],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    lines = "".join(f"{line}\n" for line in redacted_urls)
    try:
        write_private_text_exclusive(output_path, lines, reject_symlink=True)
    except OSError as exc:
        if isinstance(exc, FileExistsError):
            raise ValueError(f"URL output file already exists: {str(output_path)!r}") from exc
        raise ValueError(f"URL output file is not writable: {str(output_path)!r}") from exc
    except ValueError as exc:
        if "symlink" in str(exc):
            raise ValueError(f"URL output file is not writable: {str(output_path)!r}") from exc
        raise
    return output_path


def redacted_url_file_lines(discovery: RemoteDiscovery) -> tuple[str, ...]:
    query_redacted = {
        item.redacted_url for item in discovery.items if item.query_sha256 is not None
    }
    if query_redacted:
        count = len(query_redacted)
        noun = "URL" if count == 1 else "URLs"
        raise ValueError(
            f"redacted URL output would omit {count} distinct query-bearing {noun}; "
            "use write_raw_discovered_url_file() to preserve exact fetch targets"
        )

    seen_redacted: set[str] = set()
    lines: list[str] = []
    for item in discovery.items:
        if item.redacted_url in seen_redacted:
            continue
        seen_redacted.add(item.redacted_url)
        lines.append(item.redacted_url)
    return tuple(lines)


__all__ = [
    "RemoteDiscoveredUrl",
    "RemoteDiscovery",
    "RemoteDiscoveryKind",
    "RemoteDiscoveryReader",
    "parse_llms_txt_urls",
    "parse_sitemap_urls",
    "redacted_url_file_lines",
    "write_discovered_url_file",
    "write_raw_discovered_url_file",
    "write_redacted_discovered_url_file",
]
