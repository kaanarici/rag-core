from __future__ import annotations

import pytest

from rag_core.core_models import ProcessingFingerprint


def test_processing_fingerprint_parse_round_trips_valid_payload() -> None:
    fingerprint = ProcessingFingerprint(
        base_version="rag-core:2026-05-17",
        source_type="file",
        contextualizer_id="anthropic:claude",
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
    ],
)
def test_processing_fingerprint_parse_rejects_invalid_payloads(raw: str) -> None:
    with pytest.raises(ValueError, match="ProcessingFingerprint"):
        ProcessingFingerprint.parse(raw)
