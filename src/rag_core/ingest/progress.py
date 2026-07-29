from __future__ import annotations

from typing import Final, Literal

IngestProgressStatus = Literal["succeeded", "failed"]

INGEST_PROGRESS_SUCCEEDED: Final[IngestProgressStatus] = "succeeded"
INGEST_PROGRESS_FAILED: Final[IngestProgressStatus] = "failed"

__all__ = [
    "INGEST_PROGRESS_FAILED",
    "INGEST_PROGRESS_SUCCEEDED",
    "IngestProgressStatus",
]
