from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rag_core.core_models import IngestedDocument
from rag_core.local_ingest_models import (
    LocalIngestFailure,
    LocalIngestResult,
    LocalIngestSuccess,
)
from rag_core.remote_ingest_models import (
    RemoteUrlIngestFailure,
    RemoteUrlIngestResult,
    RemoteUrlIngestSuccess,
)
from rag_core.remote_ingest_records import safe_remote_output_source_url


def emit_local_ingest_result(result: LocalIngestResult, *, as_json: bool) -> None:
    if as_json:
        for record in result.records:
            print(json.dumps(record.to_payload(), sort_keys=True))
        return
    for record in result.records:
        if isinstance(record, LocalIngestSuccess):
            source_label = _human_source_label(record.path, fallback=record.filename)
            print(
                f"{record.ingest_state}: {source_label} -> {record.document_id} "
                f"({record.chunk_count} chunks)"
            )
        elif isinstance(record, LocalIngestFailure):
            source_label = _human_source_label(record.path, fallback=Path(record.path).name)
            print(f"failed: {source_label} -> {record.error}")


def emit_ingested_document(document: IngestedDocument, *, as_json: bool) -> None:
    payload = {
        key: value for key, value in asdict(document).items() if value is not None
    }
    source_url = _safe_metadata_url(
        document.metadata.get("source_url"),
        fallback=document.document_key or document.filename,
    )
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        payload["metadata"] = _safe_source_metadata(metadata, source_url=source_url)
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(
        f"{document.ingest_state}: {source_url} -> {document.document_id} "
        f"({document.chunk_count} chunks)"
    )


def emit_remote_url_ingest(result: RemoteUrlIngestResult, *, as_json: bool) -> None:
    if as_json:
        for record in result.records:
            print(json.dumps(record.to_payload(), sort_keys=True))
        return
    for record in result.records:
        if isinstance(record, RemoteUrlIngestSuccess):
            print(
                f"{record.ingest_state}: {record.source_url} -> {record.document_id} "
                f"({record.chunk_count} chunks)"
            )
        elif isinstance(record, RemoteUrlIngestFailure):
            print(f"failed: {record.requested_url} -> {record.error}")


def _human_source_label(path: str, *, fallback: str) -> str:
    if "!/" not in path:
        return fallback
    _archive_name, member_path = path.split("!/", 1)
    if member_path:
        return member_path
    return fallback


def _safe_source_metadata(
    metadata: dict[object, object],
    *,
    source_url: str,
) -> dict[object, object]:
    sanitized: dict[object, object] = {}
    for key, value in metadata.items():
        if key == "source_url":
            sanitized[key] = source_url
        elif isinstance(key, str) and "url" in key.lower() and isinstance(value, str):
            sanitized[key] = _safe_metadata_url(value, fallback="<redacted-url>")
        else:
            sanitized[key] = value
    sanitized["source_url"] = source_url
    return sanitized


def _safe_metadata_url(value: object, *, fallback: str) -> str:
    if isinstance(value, str):
        redacted = safe_remote_output_source_url(value, fallback="")
        if redacted:
            return redacted
    return _safe_url_fallback(fallback)


def _safe_url_fallback(value: str) -> str:
    redacted = safe_remote_output_source_url(value, fallback="")
    if redacted:
        return redacted
    if "?" in value or "@" in value:
        return "<redacted-url>"
    return value


__all__ = [
    "emit_ingested_document",
    "emit_local_ingest_result",
    "emit_remote_url_ingest",
]
