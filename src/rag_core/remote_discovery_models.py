from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final
from typing import Literal

DEFAULT_REMOTE_LLMS_TXT_MAX_URLS: Final[int] = 1_000
DEFAULT_REMOTE_SITEMAP_INDEX_MAX_FETCHES: Final[int] = 128
DEFAULT_REMOTE_SITEMAP_MAX_URLS: Final[int] = 50_000

RemoteDiscoveryKind = Literal["sitemap", "sitemap_index", "llms_txt"]

REMOTE_DISCOVERY_KIND_SITEMAP: Final[RemoteDiscoveryKind] = "sitemap"
REMOTE_DISCOVERY_KIND_SITEMAP_INDEX: Final[RemoteDiscoveryKind] = "sitemap_index"
REMOTE_DISCOVERY_KIND_LLMS_TXT: Final[RemoteDiscoveryKind] = "llms_txt"

REMOTE_DISCOVERY_CLI_KIND_SITEMAP: Final[str] = "sitemap"
REMOTE_DISCOVERY_CLI_KIND_LLMS_TXT: Final[str] = "llms-txt"


@dataclass(frozen=True)
class RemoteDiscoveredUrl:
    url: str = field(repr=False)
    redacted_url: str
    source_kind: RemoteDiscoveryKind
    query_sha256: str | None = None
    title: str = ""
    section: str = ""
    notes: str = ""
    lastmod: str = ""
    optional: bool = False

    def to_payload(self) -> dict[str, object]:
        return {
            "url": self.redacted_url,
            "source_kind": self.source_kind,
            "title": self.title,
            "section": self.section,
            "notes": self.notes,
            "lastmod": self.lastmod,
            "optional": self.optional,
        }


@dataclass(frozen=True)
class RemoteDiscovery:
    source_kind: RemoteDiscoveryKind
    items: tuple[RemoteDiscoveredUrl, ...]

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(item.url for item in self.items)

    @property
    def redacted_urls(self) -> tuple[str, ...]:
        return tuple(item.redacted_url for item in self.items)

    def to_payload(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "item_count": self.item_count,
            "items": [item.to_payload() for item in self.items],
        }


def discovery_key(item: RemoteDiscoveredUrl) -> tuple[str, str | None]:
    return (item.redacted_url, item.query_sha256)


__all__ = [
    "DEFAULT_REMOTE_LLMS_TXT_MAX_URLS",
    "DEFAULT_REMOTE_SITEMAP_INDEX_MAX_FETCHES",
    "DEFAULT_REMOTE_SITEMAP_MAX_URLS",
    "REMOTE_DISCOVERY_CLI_KIND_LLMS_TXT",
    "REMOTE_DISCOVERY_CLI_KIND_SITEMAP",
    "REMOTE_DISCOVERY_KIND_LLMS_TXT",
    "REMOTE_DISCOVERY_KIND_SITEMAP",
    "REMOTE_DISCOVERY_KIND_SITEMAP_INDEX",
    "RemoteDiscoveredUrl",
    "RemoteDiscovery",
    "RemoteDiscoveryKind",
    "discovery_key",
]
