from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from rag_core.retrieval_defaults import DEFAULT_LOCAL_SEARCH_LIMIT

DEFAULT_LOCAL_SEARCH_COLLECTION = "local_search"
DEFAULT_LOCAL_MAX_FILES = 200
DEFAULT_LOCAL_SEARCH_NAMESPACE = "local"


@dataclass(frozen=True)
class LocalSearchRequest:
    path: Path
    query: str
    namespace: str = DEFAULT_LOCAL_SEARCH_NAMESPACE
    corpus_id: str | None = None
    limit: int = DEFAULT_LOCAL_SEARCH_LIMIT
    max_files: int = DEFAULT_LOCAL_MAX_FILES


@dataclass(frozen=True)
class LocalSearchSkippedFailure:
    path: str
    error: str


@dataclass(frozen=True)
class LocalSearchFileSet:
    files: list[Path]
    skipped_unsupported_count: int
    skipped_empty_count: int
    truncated: bool


@dataclass(frozen=True)
class LocalSearchResult:
    query: str
    namespace: str
    corpus_id: str
    indexed: list[dict[str, object]]
    skipped_unsupported_count: int
    skipped_empty_count: int
    skipped_failed: list[LocalSearchSkippedFailure]
    truncated: bool
    hits: list[dict[str, object]]

    @property
    def indexed_count(self) -> int:
        return len(self.indexed)

    @property
    def skipped_count(self) -> int:
        return (
            self.skipped_unsupported_count
            + self.skipped_empty_count
            + len(self.skipped_failed)
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "query": self.query,
            "namespace": self.namespace,
            "corpus_id": self.corpus_id,
            "indexed_count": self.indexed_count,
            "indexed": list(self.indexed),
            "skipped_count": self.skipped_count,
            "skipped_empty_count": self.skipped_empty_count,
            "skipped_unsupported_count": self.skipped_unsupported_count,
            "skipped_failed": [asdict(item) for item in self.skipped_failed],
            "truncated": self.truncated,
            "hits": list(self.hits),
        }


__all__ = [
    "DEFAULT_LOCAL_SEARCH_COLLECTION",
    "DEFAULT_LOCAL_SEARCH_LIMIT",
    "DEFAULT_LOCAL_MAX_FILES",
    "DEFAULT_LOCAL_SEARCH_NAMESPACE",
    "LocalSearchFileSet",
    "LocalSearchRequest",
    "LocalSearchResult",
    "LocalSearchSkippedFailure",
]
