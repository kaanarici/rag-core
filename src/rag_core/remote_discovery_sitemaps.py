from __future__ import annotations

from collections.abc import Callable

from rag_core.fetch_security import FetchSecurityPolicy
from rag_core.remote_discovery_documents import parse_sitemap_urls
from rag_core.remote_discovery_models import (
    REMOTE_DISCOVERY_KIND_SITEMAP,
    REMOTE_DISCOVERY_KIND_SITEMAP_INDEX,
    RemoteDiscoveredUrl,
    RemoteDiscovery,
    discovery_key,
)


def expand_sitemap_index(
    discovery: RemoteDiscovery,
    *,
    fetch_sitemap_body: Callable[[str], bytes],
    policy: FetchSecurityPolicy | None,
    max_urls: int,
    max_sitemap_fetches: int,
) -> RemoteDiscovery:
    if max_sitemap_fetches <= 0:
        raise ValueError("max_sitemap_fetches must be positive")
    sitemap_queue = list(discovery.items)
    fetched_sitemaps = 0
    seen_sitemaps: set[tuple[str, str | None]] = set()
    seen_urls: set[tuple[str, str | None]] = set()
    urls: list[RemoteDiscoveredUrl] = []
    while sitemap_queue:
        sitemap = sitemap_queue.pop(0)
        sitemap_key = discovery_key(sitemap)
        if sitemap_key in seen_sitemaps:
            continue
        if fetched_sitemaps >= max_sitemap_fetches:
            raise ValueError(
                f"sitemap discovery exceeds max_sitemap_fetches ({max_sitemap_fetches})"
            )
        seen_sitemaps.add(sitemap_key)
        fetched_sitemaps += 1
        nested = parse_sitemap_urls(
            fetch_sitemap_body(sitemap.url),
            policy=policy,
            max_urls=max_urls,
        )
        if nested.source_kind == REMOTE_DISCOVERY_KIND_SITEMAP_INDEX:
            sitemap_queue.extend(nested.items)
            continue
        for item in nested.items:
            key = discovery_key(item)
            if key in seen_urls:
                continue
            if len(urls) >= max_urls:
                raise ValueError(f"remote discovery exceeds max_urls ({max_urls})")
            seen_urls.add(key)
            urls.append(item)
    return RemoteDiscovery(source_kind=REMOTE_DISCOVERY_KIND_SITEMAP, items=tuple(urls))


__all__ = ["expand_sitemap_index"]
