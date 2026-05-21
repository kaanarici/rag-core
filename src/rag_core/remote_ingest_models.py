from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy
from rag_core.manifest_persistence import (
    ManifestReconciliation,
    ManifestSource,
    manifest_reconciliation_payload,
)
from rag_core.remote_ingest_manifest import (
    remote_source_reconciliation_by_key,
    remote_url_plan_payload,
)
from rag_core.remote_ingest_results import (
    RemoteManifestStatus,
    RemoteUrlIngestFailure,
    RemoteUrlIngestResult,
    RemoteUrlIngestSuccess,
)
from rag_core.remote_document_keys import public_remote_document_key


@dataclass(frozen=True)
class RemoteUrlIngestRequest:
    namespace: str
    corpus_id: str
    url_file: str | Path | None = None
    urls: Sequence[str] = ()
    metadata: dict[str, str] | None = None
    force_reindex: bool = False
    max_concurrency: int = 1
    fetch_policy: FetchSecurityPolicy | None = None
    fetch_limits: FetchLimits | None = None


@dataclass(frozen=True)
class RemoteUrlSourceItem:
    url: str = field(repr=False)
    redacted_url: str
    document_key: str
    query_sha256: str | None = None
    source_line: int = 0
    raw_query: str = field(default="", repr=False)

    def to_payload(self, *, include_private: bool = False) -> dict[str, object]:
        document_key = (
            self.document_key
            if include_private
            else public_remote_document_key(self.document_key)
        )
        payload: dict[str, object] = {
            "url": self.redacted_url,
            "document_key": document_key,
            "source_line": self.source_line,
        }
        if not include_private and self.query_sha256:
            payload["has_private_query_identity"] = True
        return payload


@dataclass(frozen=True)
class RemoteUrlIngestPlan:
    url_source: str
    url_file: str | None
    namespace: str
    corpus_id: str
    urls: tuple[RemoteUrlSourceItem, ...]

    @property
    def url_count(self) -> int:
        return len(self.urls)

    @property
    def manifest_sources(self) -> tuple[ManifestSource, ...]:
        return tuple(
            ManifestSource(document_key=item.document_key) for item in self.urls
        )

    def to_payload(
        self,
        *,
        reconciliation: ManifestReconciliation | None = None,
        include_private: bool = False,
    ) -> dict[str, object]:
        reconciliation_by_key = remote_source_reconciliation_by_key(reconciliation)
        payload: dict[str, object] = {
            "source_type": "url",
            "url_source": self.url_source,
            "namespace": self.namespace,
            "corpus_id": self.corpus_id,
            "planned_count": self.url_count,
            "urls": [
                remote_url_plan_payload(
                    item,
                    reconciliation_by_key=reconciliation_by_key,
                    include_private=include_private,
                )
                for item in self.urls
            ],
        }
        if self.url_file is not None:
            payload["url_file"] = self.url_file
        if reconciliation is not None:
            payload["reconciliation"] = manifest_reconciliation_payload(reconciliation)
        return payload


__all__ = [
    "RemoteManifestStatus",
    "RemoteUrlIngestFailure",
    "RemoteUrlIngestPlan",
    "RemoteUrlIngestRequest",
    "RemoteUrlIngestResult",
    "RemoteUrlIngestSuccess",
    "RemoteUrlSourceItem",
]
