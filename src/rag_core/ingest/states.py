from __future__ import annotations

from typing import Final, Literal

IngestState = Literal["created", "preview", "reindexed", "replaced", "unchanged"]

INGEST_STATE_CREATED: Final[IngestState] = "created"
INGEST_STATE_PREVIEW: Final[IngestState] = "preview"
INGEST_STATE_REINDEXED: Final[IngestState] = "reindexed"
INGEST_STATE_REPLACED: Final[IngestState] = "replaced"
INGEST_STATE_UNCHANGED: Final[IngestState] = "unchanged"
