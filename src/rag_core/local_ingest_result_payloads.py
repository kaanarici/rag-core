"""Payload and bucket helpers for local ingest results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar


class IngestPayloadRecord(Protocol):
    def to_payload(self, *, include_private: bool = False) -> dict[str, object]: ...


class IngestSuccessPayloadRecord(IngestPayloadRecord, Protocol):
    @property
    def ingest_state(self) -> str: ...


TSuccess = TypeVar("TSuccess", bound=IngestSuccessPayloadRecord)
TFailure = TypeVar("TFailure", bound=IngestPayloadRecord)


def success_records(
    records: Sequence[object],
    success_type: type[TSuccess],
) -> tuple[TSuccess, ...]:
    return tuple(record for record in records if isinstance(record, success_type))


def failure_records(
    records: Sequence[object],
    failure_type: type[TFailure],
) -> tuple[TFailure, ...]:
    return tuple(record for record in records if isinstance(record, failure_type))


def written_records(records: Sequence[TSuccess]) -> tuple[TSuccess, ...]:
    return tuple(record for record in records if record.ingest_state != "unchanged")


def skipped_records(records: Sequence[TSuccess]) -> tuple[TSuccess, ...]:
    return tuple(record for record in records if record.ingest_state == "unchanged")


def ingest_result_payload(
    *,
    namespace: str,
    corpus_id: str,
    records: Sequence[IngestPayloadRecord],
    succeeded: Sequence[IngestPayloadRecord],
    written: Sequence[IngestPayloadRecord],
    skipped: Sequence[IngestPayloadRecord],
    failed: Sequence[IngestPayloadRecord],
    include_private: bool = False,
) -> dict[str, object]:
    return {
        "namespace": namespace,
        "corpus_id": corpus_id,
        "planned_count": len(records),
        "succeeded_count": len(succeeded),
        "written_count": len(written),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "records": [
            record.to_payload(include_private=include_private) for record in records
        ],
        "succeeded": [
            record.to_payload(include_private=include_private) for record in succeeded
        ],
        "written": [
            record.to_payload(include_private=include_private) for record in written
        ],
        "skipped": [
            record.to_payload(include_private=include_private) for record in skipped
        ],
        "failed": [
            record.to_payload(include_private=include_private) for record in failed
        ],
    }
