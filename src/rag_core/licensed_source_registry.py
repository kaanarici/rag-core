"""Licensed-source registry for URL ingest.

Applications may bind remote-source licenses to specific corpora. This registry
maps ``collection`` to the host allowlist authorized for that collection, so URL-list
ingest fails closed before any bytes are fetched.

This is caller-owned policy: rag-core does not own which hosts are licensed.
Callers provide the mapping via ``RemoteUrlIngestRequest``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

from rag_core.fetch_security import (
    host_matches_allowed_hosts,
    normalize_allowed_hosts,
)


class LicensedSourceMismatch(ValueError):
    """A URL's host is not authorized for the target collection."""


LicensedSourceMapping: TypeAlias = Mapping[str, Iterable[str]]


@dataclass(frozen=True)
class LicensedSourceRegistry:
    """Frozen ``collection -> allowed_hosts`` registry.

    Each value is normalized through ``normalize_allowed_hosts`` so an entry
    can be either an exact host or a leading ``*.`` wildcard suffix. A collection
    with an empty allowlist denies every URL for that collection.
    """

    entries: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_mapping(
        cls, mapping: LicensedSourceMapping | None
    ) -> LicensedSourceRegistry | None:
        if mapping is None:
            return None
        normalized: dict[str, tuple[str, ...]] = {}
        for raw_collection, raw_hosts in mapping.items():
            collection = str(raw_collection).strip()
            if not collection:
                raise ValueError(
                    "LicensedSourceRegistry collection keys must be non-empty"
                )
            host_tuple = tuple(str(host) for host in raw_hosts)
            normalized_hosts = normalize_allowed_hosts(host_tuple)
            # normalize_allowed_hosts only returns None for None input; here we
            # always pass a tuple so the result is also a tuple.
            assert normalized_hosts is not None
            normalized[collection] = normalized_hosts
        return cls(entries=normalized)

    def hosts_for(self, collection: str) -> tuple[str, ...] | None:
        return self.entries.get(collection)

    def assert_host_allowed(self, *, collection: str, host: str) -> None:
        allowed = self.hosts_for(collection)
        if allowed is None:
            raise LicensedSourceMismatch(
                f"licensed source registry has no entry for collection={collection!r}"
            )
        if not allowed:
            raise LicensedSourceMismatch(
                f"licensed source registry denies every host for collection={collection!r}"
            )
        if not host_matches_allowed_hosts(host, allowed_hosts=allowed):
            raise LicensedSourceMismatch(
                f"host {host!r} is not licensed for collection={collection!r}"
            )


__all__ = [
    "LicensedSourceMapping",
    "LicensedSourceMismatch",
    "LicensedSourceRegistry",
]
