from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from rag_core.local_ingest_manifest import (
    manifest_status_for_document,
    source_reconciliation_by_key,
)
from rag_core.manifest_persistence import (
    ManifestReconciliation,
    ManifestReconciliationStatus,
    ManifestSource,
    manifest_reconciliation_payload,
)
from rag_core.local_ingest_result_payloads import (
    failure_records,
    ingest_result_payload,
    skipped_records,
    success_records,
    written_records,
)
from rag_core.sources import LocalSourceItem

LocalManifestStatus = ManifestReconciliationStatus | Literal["unknown"]


@dataclass(frozen=True)
class LocalIngestRequest:
    path: str | Path
    namespace: str
    corpus_id: str
    metadata: dict[str, str] | None = None
    force_reindex: bool = False
    max_concurrency: int = 1


@dataclass(frozen=True)
class LocalIngestPlan:
    path: str
    namespace: str
    corpus_id: str
    documents: tuple[LocalSourceItem, ...]

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def manifest_sources(self) -> tuple[ManifestSource, ...]:
        return tuple(
            ManifestSource(
                document_key=document.document_key,
                content_sha256=document.content_sha256,
            )
            for document in self.documents
        )

    def to_payload(
        self,
        *,
        reconciliation: ManifestReconciliation | None = None,
        include_private: bool = False,
    ) -> dict[str, object]:
        reconciliation_by_key = source_reconciliation_by_key(reconciliation)
        documents: list[dict[str, object]] = []
        for document in self.documents:
            if reconciliation is None:
                documents.append(document.to_payload(include_private=include_private))
                continue
            manifest_status, manifest_reason = manifest_status_for_document(
                document,
                reconciliation_by_key,
            )
            documents.append(
                document.to_payload(
                    manifest_status=manifest_status,
                    manifest_reason=manifest_reason,
                    include_private=include_private,
                )
            )
        payload: dict[str, object] = {
            "path": self.path if include_private else "<local-source>",
            "namespace": self.namespace,
            "corpus_id": self.corpus_id,
            "planned_count": self.document_count,
            "documents": documents,
        }
        if reconciliation is not None:
            payload["reconciliation"] = manifest_reconciliation_payload(
                reconciliation,
                include_private=include_private,
            )
        return payload


@dataclass(frozen=True)
class LocalIngestSuccess:
    path: str
    document_key: str
    content_sha256: str | None
    document_id: str
    filename: str
    chunk_count: int
    ingest_state: str
    replaced_existing: bool
    manifest_status: LocalManifestStatus = "unknown"
    manifest_reason: str = "manifest_not_checked"

    def to_payload(self, *, include_private: bool = False) -> dict[str, object]:
        if include_private:
            return {"ok": True, **asdict(self)}
        return {
            "ok": True,
            "path": _public_source_path(self.path),
            "filename": self.filename,
            "document_id": self.document_id,
            "chunk_count": self.chunk_count,
            "ingest_state": self.ingest_state,
            "replaced_existing": self.replaced_existing,
            "manifest_status": self.manifest_status,
            "manifest_reason": self.manifest_reason,
        }


@dataclass(frozen=True)
class LocalIngestFailure:
    path: str
    document_key: str
    content_sha256: str | None
    error: str
    manifest_status: LocalManifestStatus = "unknown"
    manifest_reason: str = "manifest_not_checked"

    def to_payload(self, *, include_private: bool = False) -> dict[str, object]:
        if include_private:
            return {"ok": False, **asdict(self)}
        return {
            "ok": False,
            "path": _public_source_path(self.path),
            "filename": Path(self.path).name,
            "error": self.error,
            "manifest_status": self.manifest_status,
            "manifest_reason": self.manifest_reason,
        }


@dataclass(frozen=True)
class LocalIngestResult:
    namespace: str
    corpus_id: str
    records: tuple[LocalIngestSuccess | LocalIngestFailure, ...]

    @property
    def planned_count(self) -> int:
        return len(self.records)

    @property
    def succeeded(self) -> tuple[LocalIngestSuccess, ...]:
        return success_records(self.records, LocalIngestSuccess)

    @property
    def written(self) -> tuple[LocalIngestSuccess, ...]:
        return written_records(self.succeeded)

    @property
    def skipped(self) -> tuple[LocalIngestSuccess, ...]:
        return skipped_records(self.succeeded)

    @property
    def failed(self) -> tuple[LocalIngestFailure, ...]:
        return failure_records(self.records, LocalIngestFailure)

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
            corpus_id=self.corpus_id,
            records=self.records,
            succeeded=self.succeeded,
            written=self.written,
            skipped=self.skipped,
            failed=self.failed,
            include_private=include_private,
        )


__all__ = [
    "LocalIngestFailure",
    "LocalIngestPlan",
    "LocalIngestRequest",
    "LocalIngestResult",
    "LocalIngestSuccess",
    "LocalManifestStatus",
]


def _public_source_path(path: str) -> str:
    archive_path, separator, member_path = path.partition("!/")
    if separator:
        return f"{Path(archive_path).name}!/{member_path}"
    return "<local-file>"
