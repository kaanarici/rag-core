"""Aggregate result shape for remote URL ingest batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rag_core.ingest.payloads import (
    failure_records,
    ingest_result_payload,
    skipped_records,
    success_records,
    written_records,
)
from rag_core.manifest.reconciliation.reasons import MANIFEST_REASON_NOT_CHECKED
from rag_core.manifest.reconciliation.statuses import (
    MANIFEST_STATUS_UNKNOWN,
    RemoteManifestStatus,
)
from rag_core.ingest.urls.document_keys import (
    has_private_query_identity,
    public_remote_document_key,
)


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
    manifest_status: RemoteManifestStatus = MANIFEST_STATUS_UNKNOWN
    manifest_reason: str = MANIFEST_REASON_NOT_CHECKED

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
    manifest_status: RemoteManifestStatus = MANIFEST_STATUS_UNKNOWN
    manifest_reason: str = MANIFEST_REASON_NOT_CHECKED

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
    collection: str
    records: tuple[RemoteUrlIngestSuccess | RemoteUrlIngestFailure, ...]

    @property
    def planned_count(self) -> int:
        return len(self.records)

    @property
    def succeeded(self) -> tuple[RemoteUrlIngestSuccess, ...]:
        return success_records(self.records, RemoteUrlIngestSuccess)

    @property
    def written(self) -> tuple[RemoteUrlIngestSuccess, ...]:
        return written_records(self.succeeded)

    @property
    def skipped(self) -> tuple[RemoteUrlIngestSuccess, ...]:
        return skipped_records(self.succeeded)

    @property
    def failed(self) -> tuple[RemoteUrlIngestFailure, ...]:
        return failure_records(self.records, RemoteUrlIngestFailure)

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
        return ingest_result_payload(
            namespace=self.namespace,
            collection=self.collection,
            records=self.records,
            succeeded=self.succeeded,
            written=self.written,
            skipped=self.skipped,
            failed=self.failed,
            include_private=include_private,
        )


__all__ = [
    "RemoteManifestStatus",
    "RemoteUrlIngestFailure",
    "RemoteUrlIngestResult",
    "RemoteUrlIngestSuccess",
]
