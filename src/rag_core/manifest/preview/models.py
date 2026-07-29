from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

from rag_core.core_models import CollectionManifestEntry, IngestedDocument


@dataclass(frozen=True)
class ManifestPreviewRequest:
    path: Path
    namespace: str
    collection: str
    document_id: str | None = None
    document_key: str | None = None
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class ManifestPreviewResult:
    document: IngestedDocument
    manifest_entry: CollectionManifestEntry

    def to_payload(self) -> dict[str, object]:
        return {
            "document": _dataclass_payload(self.document),
            "manifest_entry": _dataclass_payload(self.manifest_entry),
        }


def _dataclass_payload(value: object) -> dict[str, object]:
    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
        return {str(key): item for key, item in payload.items() if item is not None}
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if item is not None}
    return {}


__all__ = [
    "ManifestPreviewRequest",
    "ManifestPreviewResult",
]
