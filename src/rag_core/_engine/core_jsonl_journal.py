"""Shared append-only JSONL machinery for crash-recovery journals."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Generic, Protocol, TypeVar, cast


class JsonlJournalEntry(Protocol):
    @property
    def namespace(self) -> str: ...

    @property
    def collection(self) -> str: ...

    @property
    def document_id(self) -> str: ...

    def to_jsonl(self) -> str: ...


EntryT = TypeVar("EntryT", bound=JsonlJournalEntry)


@dataclass
class JsonlRecoveryJournal(Generic[EntryT]):
    """Append-only JSONL journal folded by document identity."""

    directory: Path

    _FILE_NAME: ClassVar[str]
    _CHARSET: ClassVar[str] = "utf-8"

    @property
    def path(self) -> Path:
        # Directory creation is deferred to first write so a read-only
        # inspect path does not make the disk dirty.
        return self.directory / self._FILE_NAME

    def _append(self, entry: EntryT) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        charset = cast(str, getattr(type(entry), "_CHARSET", self._CHARSET))
        with self.path.open("a", encoding=charset) as handle:
            handle.write(entry.to_jsonl())
            handle.write("\n")

    def _iter_entries(self) -> list[EntryT]:
        if not self.path.exists():
            return []
        results: list[EntryT] = []
        with self.path.open("r", encoding=self._CHARSET) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry = self._parse(cast(Mapping[str, object], payload))
                if entry is not None:
                    results.append(entry)
        return results

    def _latest_by_key(self) -> dict[tuple[str, str, str], EntryT]:
        latest_by_key: dict[tuple[str, str, str], EntryT] = {}
        for entry in self._iter_entries():
            latest_by_key[(entry.namespace, entry.collection, entry.document_id)] = entry
        return latest_by_key

    def latest_entry(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
    ) -> EntryT | None:
        # File append order is authoritative: the last matching line wins.
        # Timestamps are informational and must not decide replay.
        matching = [
            entry
            for entry in self._iter_entries()
            if entry.namespace == namespace
            and entry.collection == collection
            and entry.document_id == document_id
        ]
        if not matching:
            return None
        return matching[-1]

    def pending_entries(self) -> list[EntryT]:
        return [
            entry for entry in self._latest_by_key().values() if self._is_pending(entry)
        ]

    def _compact_if_oversized(self) -> None:
        """Rewrite the journal keeping only still-pending latest entries."""
        try:
            if self.path.stat().st_size < self._compact_min_bytes():
                return
        except OSError:
            return
        pending = [
            entry for entry in self._latest_by_key().values() if self._is_pending(entry)
        ]
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        with tmp_path.open("w", encoding=self._CHARSET) as handle:
            for entry in pending:
                handle.write(entry.to_jsonl())
                handle.write("\n")
        tmp_path.replace(self.path)

    def _parse(self, payload: Mapping[str, object]) -> EntryT | None:
        raise NotImplementedError

    def _is_pending(self, entry: EntryT) -> bool:
        raise NotImplementedError

    def _compact_min_bytes(self) -> int:
        raise NotImplementedError


__all__ = ["JsonlRecoveryJournal"]
