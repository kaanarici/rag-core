"""Delete-recovery journal: track partial right-to-forget purges to disk.

The right-to-forget contract requires that a multi-step delete
(vector store -> sidecar -> embedding cache -> chunk-context cache ->
manifest) either complete fully or leave a recoverable trail. This journal
records, per ``(namespace, corpus_id, document_id)`` triple, which stages
have succeeded so a later replay (on next ingest of the same doc, or on
process start) can resume from the failed stage.

The journal is JSONL appended on every transition. The replay path reads
the file, folds entries by ``(namespace, corpus_id, document_id)``, and
returns the latest state. A completed entry is logged with
``completed=true`` so the replay knows it has nothing to do.

Restricted-tier deployments care most about this seam: a partial purge that
silently strands sensitive bytes in the chunk-context cache or sidecar is the
exact failure mode this slice closes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


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
    corpus_id: str
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
            "corpus_id": self.corpus_id,
            "document_id": self.document_id,
            "stages_completed": list(self.stages_completed),
            "completed": self.completed,
            "last_error_type": self.last_error_type,
            "last_error_stage": self.last_error_stage,
            "created_at_ns": self.created_at_ns,
            "updated_at_ns": self.updated_at_ns,
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


@dataclass
class DeleteRecoveryJournal:
    """Append-only JSONL journal under ``directory / .delete_recovery.jsonl``.

    The journal is opportunistic: a missing or unreadable file is not fatal.
    Right-to-forget recovery prefers a noisy fresh start over a hidden state.
    Callers feed ``record(...)`` after each delete stage; ``latest_entry(...)``
    returns the most recent recorded state for a ``(ns, corpus, doc)`` triple.
    """

    directory: Path

    @property
    def path(self) -> Path:
        # Directory creation is deferred to first write so a read-only
        # inspect path does not make the disk dirty.
        return self.directory / JOURNAL_FILE_NAME

    def record(
        self,
        *,
        namespace: str,
        corpus_id: str,
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
            corpus_id=corpus_id,
            document_id=document_id,
            stages_completed=tuple(stages_completed),
            completed=completed,
            last_error_type=last_error_type,
            last_error_stage=last_error_stage,
            created_at_ns=created_at_ns or now_ns,
            updated_at_ns=now_ns,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding=DeleteRecoveryEntry._CHARSET) as handle:
            handle.write(entry.to_jsonl())
            handle.write("\n")
        if completed:
            self._compact_if_oversized()
        return entry

    def _iter_entries(self) -> list[DeleteRecoveryEntry]:
        if not self.path.exists():
            return []
        results: list[DeleteRecoveryEntry] = []
        with self.path.open("r", encoding=DeleteRecoveryEntry._CHARSET) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    # A torn write or operator edit shouldn't crash recovery;
                    # the worst-case is missed pending state which the next
                    # delete attempt re-records.
                    continue
                results.append(
                    DeleteRecoveryEntry(
                        namespace=str(payload.get("namespace", "")),
                        corpus_id=str(payload.get("corpus_id", "")),
                        document_id=str(payload.get("document_id", "")),
                        stages_completed=tuple(
                            str(stage)
                            for stage in payload.get("stages_completed", [])
                        ),
                        completed=bool(payload.get("completed", False)),
                        last_error_type=payload.get("last_error_type"),
                        last_error_stage=payload.get("last_error_stage"),
                        created_at_ns=int(payload.get("created_at_ns", 0)),
                        updated_at_ns=int(payload.get("updated_at_ns", 0)),
                    )
                )
        return results

    def all_entries_for(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str,
    ) -> list[DeleteRecoveryEntry]:
        return [
            entry
            for entry in self._iter_entries()
            if entry.namespace == namespace
            and entry.corpus_id == corpus_id
            and entry.document_id == document_id
        ]

    def latest_entry(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str,
    ) -> DeleteRecoveryEntry | None:
        # File append order is the authoritative ordering: the last matching
        # line wins. Timestamps are informational and must not decide replay.
        entries = self.all_entries_for(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
        )
        if not entries:
            return None
        return entries[-1]

    def pending_entries(self) -> list[DeleteRecoveryEntry]:
        """Return every entry whose latest record is not ``completed``."""
        return [
            entry
            for entry in self._latest_by_key().values()
            if not entry.completed
        ]

    def _latest_by_key(self) -> dict[tuple[str, str, str], DeleteRecoveryEntry]:
        latest_by_key: dict[tuple[str, str, str], DeleteRecoveryEntry] = {}
        for entry in self._iter_entries():
            latest_by_key[(entry.namespace, entry.corpus_id, entry.document_id)] = entry
        return latest_by_key

    def _compact_if_oversized(self) -> None:
        """Rewrite the journal keeping only still-incomplete latest entries.

        Completed purges replay as no-ops, so dropping them preserves
        recovery semantics while bounding file growth. Atomic via temp +
        replace; safe in-process because journal calls never await
        mid-operation.
        """
        try:
            if self.path.stat().st_size < _COMPACT_MIN_BYTES:
                return
        except OSError:
            return
        pending = [
            entry for entry in self._latest_by_key().values() if not entry.completed
        ]
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        with tmp_path.open("w", encoding=DeleteRecoveryEntry._CHARSET) as handle:
            for entry in pending:
                handle.write(entry.to_jsonl())
                handle.write("\n")
        tmp_path.replace(self.path)


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
