from __future__ import annotations

import pytest

from rag_core.core_models import ProcessingFingerprint
from rag_core.config import ChunkingConfig
from rag_core._engine.core_runtime import describe_source_processing_versions


def test_processing_fingerprint_parse_round_trips_valid_payload() -> None:
    fingerprint = ProcessingFingerprint(
        base_version="rag-core:2026-05-17",
        source_type="file",
        contextualizer_id="anthropic:claude",
        contextualizer_chunk_cap=250,
        chunking=ChunkingConfig(strategy="markdown", max_chars=500, overlap=50),
        sparse_index=True,
    )

    parsed = ProcessingFingerprint.parse(fingerprint.serialize())

    assert parsed == fingerprint


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        "{}",
        '{"base_version":"","source_type":"file"}',
        '{"base_version":"v1","source_type":""}',
        '{"base_version":"v1","source_type":"file","contextualizer_id":1}',
        '{"base_version":"v1","source_type":"file","contextualizer_chunk_cap":0}',
        '{"base_version":"v1","source_type":"file","contextualizer_chunk_cap":true}',
        '{"base_version":"v1","source_type":"file","contextualizer_chunk_cap":"10"}',
        '{"base_version":"v1","source_type":"file","chunking":[]}',
        '{"base_version":"v1","source_type":"file","chunking":{"max_chars":0}}',
        '{"base_version":"v1","source_type":"file","chunking":{"max_chars":500,"overlap":500}}',
        '{"base_version":"v1","source_type":"file","sparse_index":"true"}',
    ],
)
def test_processing_fingerprint_parse_rejects_invalid_payloads(raw: str) -> None:
    with pytest.raises(ValueError, match="ProcessingFingerprint"):
        ProcessingFingerprint.parse(raw)


def test_source_versions_preserve_chunking_and_sparse_identity() -> None:
    fingerprint = ProcessingFingerprint(
        base_version="v1",
        source_type="file",
        chunking=ChunkingConfig(max_chars=500, overlap=50),
        sparse_index=True,
    )
    for raw in describe_source_processing_versions(fingerprint).values():
        parsed = ProcessingFingerprint.parse(raw)
        assert parsed.chunking == fingerprint.chunking
        assert parsed.sparse_index is True


def test_legacy_fingerprints_are_readable_but_no_longer_match_current_processing() -> None:
    raw = '{"base_version":"v1","source_type":"file"}'
    parsed = ProcessingFingerprint.parse(raw)
    assert parsed.base_version == "v1"
    assert parsed.serialize() != raw
