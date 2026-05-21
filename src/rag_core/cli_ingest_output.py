from __future__ import annotations

import json
from pathlib import Path

from rag_core.local_ingest_models import (
    LocalIngestFailure,
    LocalIngestResult,
    LocalIngestSuccess,
)


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


def _human_source_label(path: str, *, fallback: str) -> str:
    if "!/" not in path:
        return fallback
    _archive_name, member_path = path.split("!/", 1)
    if member_path:
        return member_path
    return fallback


__all__ = ["emit_local_ingest_result"]
