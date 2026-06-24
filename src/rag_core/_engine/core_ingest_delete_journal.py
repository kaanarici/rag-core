"""Delete-recovery journal: track partial right-to-forget purges to disk.

The right-to-forget contract requires that a multi-step delete
(vector store -> sidecar -> embedding cache -> chunk-context cache ->
manifest) either complete fully or leave a recoverable trail. This journal
records, per ``(namespace, collection, document_id)`` triple, which stages
have succeeded so a later replay (on next ingest of the same doc, or on
process start) can resume from the failed stage.

The journal is JSONL appended on every transition. The replay path reads
the file, folds entries by ``(namespace, collection, document_id)``, and
returns the latest state. A completed entry is logged with
``completed=true`` so the replay knows it has nothing to do.

Restricted-tier deployments care most about this seam: a partial purge that
silently strands sensitive bytes in the chunk-context cache or sidecar is the
exact failure mode this slice closes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from rag_core._engine.core_jsonl_journal import JsonlRecoveryJournal

# Canonical stage labels. Kept stable in the on-disk journal so replays
# survive code refactors. New stages must be appended, never reordered.
STAGE_VECTOR_STORE = "vector_store"
STAGE_LEXICAL_SIDECAR = "lexical_sidecar"
STAGE_EMBEDDING_CACHE = "embedding_cache"
STAGE_CHUNK_CONTEXT_CACHE = "chunk_context_cache"
STAGE_MANIFEST = "manifest"

DELETE_STAGES_IN_ORDER: tuple[str, ...] = (
    STAGE_VECTOR_STORE,
    STAGE_LEXICAL_SIDECAR,
    STAGE_EMBEDDING_CACHE,
    STAGE_CHUNK_CONTEXT_CACHE,
    STAGE_MANIFEST,
)

JOURNAL_FILE_NAME = ".delete_recovery.jsonl"

# Compaction trigger: once the journal file exceeds this size, recording a
# completed purge rewrites the file keeping only the still-incomplete tail.
_COMPACT_MIN_BYTES = 64 * 1024


@dataclass(frozen=True)
class DeleteRecoveryEntry:
    """One audit-shaped record about the right-to-forget purge for a doc."""

    namespace: str
    collection: str
    document_id: str
    stages_completed: tuple[str, ...]
    completed: bool
    last_error_type: str | None = None
    last_error_stage: str | None = None
    created_at_ns: int = 0
    updated_at_ns: int = 0

    _CHARSET: ClassVar[str] = "utf-8"

    def to_jsonl(self) -> str:
        payload = {
            "namespace": self.namespace,
            "collection": self.collection,
            "document_id": self.document_id,
            "stages_completed": list(self.stages_completed),
            "completed": self.completed,
            "last_error_type": self.last_error_type,
            "last_error_stage": self.last_error_stage,
            "created_at_ns": self.created_at_ns,
            "updated_at_ns": self.updated_at_ns,
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


class DeleteRecoveryJournal(JsonlRecoveryJournal[DeleteRecoveryEntry]):
    """Append-only JSONL journal under ``directory / .delete_recovery.jsonl``.

    The journal is opportunistic: a missing or unreadable file is not fatal.
    Right-to-forget recovery prefers a noisy fresh start over a hidden state.
    Callers feed ``record(...)`` after each delete stage; ``latest_entry(...)``
    returns the most recent recorded state for a ``(ns, collection, doc)`` triple.
    """

    _FILE_NAME: ClassVar[str] = JOURNAL_FILE_NAME
    _CHARSET: ClassVar[str] = DeleteRecoveryEntry._CHARSET

    def record(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
        stages_completed: tuple[str, ...],
        completed: bool,
        last_error_type: str | None = None,
        last_error_stage: str | None = None,
        created_at_ns: int | None = None,
    ) -> DeleteRecoveryEntry:
        # Wall clock, informational only: entry ordering is file append order,
        # which survives process restarts and OS reboots (monotonic does not).
        now_ns = time.time_ns()
        entry = DeleteRecoveryEntry(
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            stages_completed=tuple(stages_completed),
            completed=completed,
            last_error_type=last_error_type,
            last_error_stage=last_error_stage,
            created_at_ns=created_at_ns or now_ns,
            updated_at_ns=now_ns,
        )
        self._append(entry)
        if completed:
            self._compact_if_oversized()
        return entry

    def _parse(self, payload: Mapping[str, object]) -> DeleteRecoveryEntry:
        return DeleteRecoveryEntry(
            namespace=str(payload.get("namespace", "")),
            collection=str(payload.get("collection", "")),
            document_id=str(payload.get("document_id", "")),
            stages_completed=tuple(
                str(stage)
                for stage in cast(
                    Iterable[object], payload.get("stages_completed", [])
                )
            ),
            completed=bool(payload.get("completed", False)),
            last_error_type=cast("str | None", payload.get("last_error_type")),
            last_error_stage=cast("str | None", payload.get("last_error_stage")),
            created_at_ns=int(cast(Any, payload.get("created_at_ns", 0))),
            updated_at_ns=int(cast(Any, payload.get("updated_at_ns", 0))),
        )

    def _is_pending(self, entry: DeleteRecoveryEntry) -> bool:
        return not entry.completed

    def _compact_min_bytes(self) -> int:
        return _COMPACT_MIN_BYTES


__all__ = [
    "DELETE_STAGES_IN_ORDER",
    "DeleteRecoveryEntry",
    "DeleteRecoveryJournal",
    "JOURNAL_FILE_NAME",
    "STAGE_CHUNK_CONTEXT_CACHE",
    "STAGE_EMBEDDING_CACHE",
    "STAGE_LEXICAL_SIDECAR",
    "STAGE_MANIFEST",
    "STAGE_VECTOR_STORE",
]
