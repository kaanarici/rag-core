from __future__ import annotations

import xml.etree.ElementTree as ET
from urllib.parse import urljoin

from rag_core.fetch_security import FetchSecurityPolicy, validate_fetch_url
from rag_core.remote_discovery_models import (
    DEFAULT_REMOTE_LLMS_TXT_MAX_URLS,
    DEFAULT_REMOTE_SITEMAP_MAX_URLS,
    REMOTE_DISCOVERY_KIND_LLMS_TXT,
    REMOTE_DISCOVERY_KIND_SITEMAP,
    REMOTE_DISCOVERY_KIND_SITEMAP_INDEX,
    RemoteDiscoveredUrl,
    RemoteDiscovery,
    RemoteDiscoveryKind,
    discovery_key,
)
from rag_core.remote_discovery_parsing import (
    LLMS_LINK_RE,
    child_text,
    children_named,
    decode_text,
    decode_xml_text,
    llms_notes,
    local_name,
    reject_xml_entities,
    validate_max_urls,
)


def parse_sitemap_urls(
    content: str | bytes,
    *,
    policy: FetchSecurityPolicy | None = None,
    max_urls: int = DEFAULT_REMOTE_SITEMAP_MAX_URLS,
) -> RemoteDiscovery:
    validate_max_urls(max_urls)
    text = decode_xml_text(content)
    reject_xml_entities(text)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"sitemap XML is not parseable: {exc}") from exc

    root_name = local_name(root.tag)
    if root_name == "urlset":
        items = _sitemap_items(
            children_named(root, "url"),
            policy=policy,
            max_urls=max_urls,
        )
        return RemoteDiscovery(source_kind=REMOTE_DISCOVERY_KIND_SITEMAP, items=items)
    if root_name == "sitemapindex":
        items = _sitemap_items(
            children_named(root, "sitemap"),
            policy=policy,
            max_urls=max_urls,
            source_kind=REMOTE_DISCOVERY_KIND_SITEMAP_INDEX,
        )
        return RemoteDiscovery(source_kind=REMOTE_DISCOVERY_KIND_SITEMAP_INDEX, items=items)
    raise ValueError(f"unsupported sitemap root element: {root_name or '<missing>'}")


def parse_llms_txt_urls(
    content: str | bytes,
    *,
    base_url: str,
    policy: FetchSecurityPolicy | None = None,
    max_urls: int = DEFAULT_REMOTE_LLMS_TXT_MAX_URLS,
) -> RemoteDiscovery:
    validate_max_urls(max_urls)
    section = ""
    items: list[RemoteDiscoveredUrl] = []
    seen: set[tuple[str, str | None]] = set()
    for line in decode_text(content).splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            section = stripped[3:].strip()
            continue
        match = LLMS_LINK_RE.match(line)
        if match is None:
            continue
        raw_url = urljoin(base_url, match.group("url").strip())
        item = _discovered_url(
            raw_url,
            source_kind=REMOTE_DISCOVERY_KIND_LLMS_TXT,
            policy=policy,
            title=match.group("title").strip(),
            section=section,
            notes=llms_notes(match.group("after")),
            optional=section.casefold() == "optional",
        )
        key = discovery_key(item)
        if key in seen:
            continue
        if len(items) >= max_urls:
            raise ValueError(f"remote discovery exceeds max_urls ({max_urls})")
        seen.add(key)
        items.append(item)
    return RemoteDiscovery(source_kind=REMOTE_DISCOVERY_KIND_LLMS_TXT, items=tuple(items))


def _sitemap_items(
    entries: list[ET.Element],
    *,
    policy: FetchSecurityPolicy | None,
    max_urls: int,
    source_kind: RemoteDiscoveryKind = REMOTE_DISCOVERY_KIND_SITEMAP,
) -> tuple[RemoteDiscoveredUrl, ...]:
    items: list[RemoteDiscoveredUrl] = []
    seen: set[tuple[str, str | None]] = set()
    for entry in entries:
        item = _sitemap_item(entry, policy=policy, source_kind=source_kind)
        key = discovery_key(item)
        if key in seen:
            continue
        if len(items) >= max_urls:
            raise ValueError(f"remote discovery exceeds max_urls ({max_urls})")
        seen.add(key)
        items.append(item)
    return tuple(items)


def _sitemap_item(
    entry: ET.Element,
    *,
    policy: FetchSecurityPolicy | None,
    source_kind: RemoteDiscoveryKind = REMOTE_DISCOVERY_KIND_SITEMAP,
) -> RemoteDiscoveredUrl:
    loc = child_text(entry, "loc")
    if not loc:
        raise ValueError(f"{source_kind} entry is missing <loc>")
    return _discovered_url(
        loc,
        source_kind=source_kind,
        policy=policy,
        lastmod=child_text(entry, "lastmod"),
    )


def _discovered_url(
    url: str,
    *,
    source_kind: RemoteDiscoveryKind,
    policy: FetchSecurityPolicy | None,
    title: str = "",
    section: str = "",
    notes: str = "",
    lastmod: str = "",
    optional: bool = False,
) -> RemoteDiscoveredUrl:
    validated = validate_fetch_url(url, policy=policy)
    return RemoteDiscoveredUrl(
        url=url.strip(),
        redacted_url=validated.redacted_url,
        source_kind=source_kind,
        query_sha256=validated.query_sha256,
        title=title,
        section=section,
        notes=notes,
        lastmod=lastmod.strip(),
        optional=optional,
    )
