"""Licensed-source registry for URL ingest.

Applications may bind remote-source licenses to specific corpora. This registry
maps ``corpus_id`` to the host allowlist authorized for that corpus, so URL-list
ingest fails closed before any bytes are fetched.

This is caller-owned policy: rag-core does not own which hosts are licensed.
Callers provide the mapping via ``RemoteUrlIngestRequest``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

from rag_core.fetch_security_url import (
    host_matches_allowed_hosts,
    normalize_allowed_hosts,
)


class LicensedSourceMismatch(ValueError):
    """A URL's host is not authorized for the target corpus."""


LicensedSourceMapping: TypeAlias = Mapping[str, Iterable[str]]


@dataclass(frozen=True)
class LicensedSourceRegistry:
    """Frozen ``corpus_id -> allowed_hosts`` registry.

    Each value is normalized through ``normalize_allowed_hosts`` so an entry
    can be either an exact host or a leading ``*.`` wildcard suffix. A corpus
    with an empty allowlist denies every URL for that corpus.
    """

    entries: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_mapping(
        cls, mapping: LicensedSourceMapping | None
    ) -> LicensedSourceRegistry | None:
        if mapping is None:
            return None
        normalized: dict[str, tuple[str, ...]] = {}
        for raw_corpus_id, raw_hosts in mapping.items():
            corpus_id = str(raw_corpus_id).strip()
            if not corpus_id:
                raise ValueError(
                    "LicensedSourceRegistry corpus_id keys must be non-empty"
                )
            host_tuple = tuple(str(host) for host in raw_hosts)
            normalized_hosts = normalize_allowed_hosts(host_tuple)
            # normalize_allowed_hosts only returns None for None input; here we
            # always pass a tuple so the result is also a tuple.
            assert normalized_hosts is not None
            normalized[corpus_id] = normalized_hosts
        return cls(entries=normalized)

    def hosts_for(self, corpus_id: str) -> tuple[str, ...] | None:
        return self.entries.get(corpus_id)

    def assert_host_allowed(self, *, corpus_id: str, host: str) -> None:
        allowed = self.hosts_for(corpus_id)
        if allowed is None:
            raise LicensedSourceMismatch(
                f"licensed source registry has no entry for corpus_id={corpus_id!r}"
            )
        if not allowed:
            raise LicensedSourceMismatch(
                f"licensed source registry denies every host for corpus_id={corpus_id!r}"
            )
        if not host_matches_allowed_hosts(host, allowed_hosts=allowed):
            raise LicensedSourceMismatch(
                f"host {host!r} is not licensed for corpus_id={corpus_id!r}"
            )


__all__ = [
    "LicensedSourceMapping",
    "LicensedSourceMismatch",
    "LicensedSourceRegistry",
]
