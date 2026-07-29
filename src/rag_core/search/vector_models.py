"""Vector and result value objects for search infrastructure."""

from __future__ import annotations

import enum
import math
import numbers
from dataclasses import dataclass, field
from typing import Final, Literal, Optional, Sequence, overload

from rag_core.search.sparse_channels import merge_sparse_channels

SEARCH_RESULT_TYPE_TEXT: Final[str] = "text"


class ContentType(str, enum.Enum):
    DOCUMENT = "document"
    CODE = "code"


@dataclass(frozen=True)
class SparseVector:
    """Sparse vector representation (BM25 indices + values)."""

    indices: list[int]
    values: list[float]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                "SparseVector.indices and values must have the same length"
            )
        for index in self.indices:
            if not _is_sparse_index(index):
                raise ValueError("SparseVector.indices must be non-negative integers")
        for value in self.values:
            if not _is_finite_real_number(value):
                raise ValueError("SparseVector.values must be finite numbers")


@dataclass(frozen=True)
class TextualRepresentation:
    """Enriched metadata + content text for embedding or lexical indexing."""

    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class VectorPoint:
    """A point to upsert into the vector store."""

    id: str
    dense_vector: list[float]
    sparse_vector: SparseVector
    payload: dict[str, object]
    sparse_vectors: dict[str, SparseVector] = field(default_factory=dict)
    sparse_text: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_id(self.id, "VectorPoint.id")
        _validate_dense_vector(self.dense_vector, "VectorPoint.dense_vector")

    def all_sparse_vectors(self) -> dict[str, SparseVector]:
        """Return sparse vectors keyed by channel name (always includes bm25)."""
        return merge_sparse_channels(self.sparse_vector, self.sparse_vectors)


@dataclass(frozen=True)
class Evidence:
    """One ranked piece of source evidence.

    ``score`` is the retrieval or fusion score returned by the vector store.
    Provider rerank scores, when present, live under ``metadata["rerank"]`` so
    callers can compare retrieval and rerank signals explicitly.

    Example: Evidence(id="uuid5-hex", text="Clean retrieved chunk body",
             score=0.87, content_type="document", source_type="<source-type>",
             document_id="doc_123", collection="help_center",
             document_key="docs/report.pdf", title="Q1 Report",
             chunk_index=3, section_title="Introduction")
    """

    id: str
    text: str
    score: float
    content_type: str
    source_type: str
    document_id: Optional[str] = None
    collection: Optional[str] = None
    document_key: Optional[str] = None
    content_sha256: Optional[str] = None
    title: Optional[str] = None
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    document_path: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_word_count: Optional[int] = None
    chunk_token_estimate: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    embedding_model: Optional[str] = None
    chunker_strategy: Optional[str] = None
    result_type: Optional[str] = None
    figure_id: Optional[str] = None
    figure_thumbnail_url: Optional[str] = None
    namespace: Optional[str] = None
    metadata: dict[str, object] = field(default_factory=dict)
    equivalent_sources: tuple[dict[str, object], ...] = ()

    def __post_init__(self) -> None:
        _validate_non_empty_id(self.id, "SearchResult.id")

    @property
    def chunk_id(self) -> str:
        return self.id

    @property
    def section(self) -> str | None:
        return self.section_title or self.section_path or self.section_id

    @property
    def locator(self) -> dict[str, object]:
        metadata = self.metadata
        values: dict[str, object | None] = {
            "chunk_index": self.chunk_index,
            "section_path": self.section_path,
            "page_number": _metadata_int(metadata, "page_number"),
            "page_index": _metadata_int(metadata, "page_index"),
            "slide_number": _metadata_int(metadata, "slide_number"),
            "sheet_name": _metadata_str(metadata, "sheet_name"),
            "row_range": _metadata_str(metadata, "row_range"),
            "line_start": _metadata_int(metadata, "line_start"),
            "line_end": _metadata_int(metadata, "line_end"),
            "start_char": self.start_char,
            "end_char": self.end_char,
            "figure_id": self.figure_id,
        }
        return {key: value for key, value in values.items() if value is not None}

    @property
    def retrieval_signals(self) -> dict[str, object]:
        signals: dict[str, object] = {"score": self.score}
        rerank = self.metadata.get("rerank")
        if isinstance(rerank, dict):
            signals["rerank"] = dict(rerank)
        return signals


# Internal provider/store code historically used ``SearchResult``. Keep one
# value object while the common path names what it represents: evidence.
SearchResult = Evidence

AnswerabilityStatus = Literal["sufficient", "insufficient", "unknown"]


@dataclass(frozen=True)
class Answerability:
    status: AnswerabilityStatus = "unknown"
    reason: str = "not_calibrated"
    calibration: str | None = None
    signals: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"sufficient", "insufficient", "unknown"}:
            raise ValueError(
                "Answerability.status must be 'sufficient', 'insufficient', or 'unknown'"
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Answerability.reason must be a non-empty string")
        if self.calibration is not None and (
            not isinstance(self.calibration, str) or not self.calibration.strip()
        ):
            raise ValueError(
                "Answerability.calibration must be a non-empty string when set"
            )


@dataclass(frozen=True)
class RetrievalResult:
    evidence: tuple[Evidence, ...]
    answerability: Answerability = field(default_factory=Answerability)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.evidence)

    def __len__(self) -> int:
        return len(self.evidence)

    @overload
    def __getitem__(self, index: int) -> Evidence: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Evidence, ...]: ...

    def __getitem__(self, index: int | slice) -> Evidence | tuple[Evidence, ...]:
        return self.evidence[index]


def _validate_dense_vector(values: Sequence[object], field_name: str) -> None:
    for value in values:
        if not _is_finite_real_number(value):
            raise ValueError(f"{field_name} must contain finite numbers")


def _validate_non_empty_id(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _is_sparse_index(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, numbers.Integral)
        and int(value) >= 0
    )


def _is_finite_real_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, numbers.Real)
        and math.isfinite(float(value))
    )


def _metadata_int(metadata: dict[str, object], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _metadata_str(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    resolved = str(value).strip()
    return resolved or None
