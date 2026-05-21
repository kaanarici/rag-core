"""Aggregate result shape for remote URL ingest batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from rag_core.manifest_persistence import ManifestReconciliationStatus
from rag_core.remote_document_keys import (
    has_private_query_identity,
    public_remote_document_key,
)

RemoteManifestStatus = ManifestReconciliationStatus | Literal[
    "unknown",
    "unknown_until_fetch",
]


@dataclass(frozen=True)
class RemoteUrlIngestSuccess:
    requested_url: str
    source_url: str
    document_key: str
    content_sha256: str | None
    document_id: str
    filename: str
    chunk_count: int
    ingest_state: str
    replaced_existing: bool
    manifest_status: RemoteManifestStatus = "unknown"
    manifest_reason: str = "manifest_not_checked"

    def to_payload(self, *, include_private: bool = False) -> dict[str, object]:
        payload = {"ok": True, **asdict(self)}
        if not include_private:
            payload["document_key"] = public_remote_document_key(self.document_key)
            if has_private_query_identity(self.document_key):
                payload["has_private_query_identity"] = True
        return payload


@dataclass(frozen=True)
class RemoteUrlIngestFailure:
    requested_url: str
    document_key: str
    error: str
    manifest_status: RemoteManifestStatus = "unknown"
    manifest_reason: str = "manifest_not_checked"

    def to_payload(self, *, include_private: bool = False) -> dict[str, object]:
        payload = {"ok": False, **asdict(self)}
        if not include_private:
            payload["document_key"] = public_remote_document_key(self.document_key)
            if has_private_query_identity(self.document_key):
                payload["has_private_query_identity"] = True
        return payload


@dataclass(frozen=True)
class RemoteUrlIngestResult:
    namespace: str
    corpus_id: str
    records: tuple[RemoteUrlIngestSuccess | RemoteUrlIngestFailure, ...]

    @property
    def planned_count(self) -> int:
        return len(self.records)

    @property
    def succeeded(self) -> tuple[RemoteUrlIngestSuccess, ...]:
        return tuple(
            record
            for record in self.records
            if isinstance(record, RemoteUrlIngestSuccess)
        )

    @property
    def written(self) -> tuple[RemoteUrlIngestSuccess, ...]:
        return tuple(
            record for record in self.succeeded if record.ingest_state != "unchanged"
        )

    @property
    def skipped(self) -> tuple[RemoteUrlIngestSuccess, ...]:
        return tuple(
            record for record in self.succeeded if record.ingest_state == "unchanged"
        )

    @property
    def failed(self) -> tuple[RemoteUrlIngestFailure, ...]:
        return tuple(
            record
            for record in self.records
            if isinstance(record, RemoteUrlIngestFailure)
        )

    @property
    def succeeded_count(self) -> int:
        return len(self.succeeded)

    @property
    def written_count(self) -> int:
        return len(self.written)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def failed_count(self) -> int:
        return len(self.failed)

    def to_payload(self, *, include_private: bool = False) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "corpus_id": self.corpus_id,
            "planned_count": self.planned_count,
            "succeeded_count": self.succeeded_count,
            "written_count": self.written_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "records": [
                record.to_payload(include_private=include_private)
                for record in self.records
            ],
            "succeeded": [
                record.to_payload(include_private=include_private)
                for record in self.succeeded
            ],
            "written": [
                record.to_payload(include_private=include_private)
                for record in self.written
            ],
            "skipped": [
                record.to_payload(include_private=include_private)
                for record in self.skipped
            ],
            "failed": [
                record.to_payload(include_private=include_private)
                for record in self.failed
            ],
        }


__all__ = [
    "RemoteManifestStatus",
    "RemoteUrlIngestFailure",
    "RemoteUrlIngestResult",
    "RemoteUrlIngestSuccess",
]
