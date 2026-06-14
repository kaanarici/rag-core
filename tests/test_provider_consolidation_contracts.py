from __future__ import annotations

from enum import Enum
from pathlib import Path

import pytest

from rag_core.search.policy import DEFAULT_POLICY


class _PayloadKey(Enum):
    VALUE = "enum_key"


class _PayloadValue(Enum):
    VALUE = "enum_value"


def test_stored_payload_owns_json_payload_normalization() -> None:
    from rag_core.search.stored_payload import (
        JsonPayloadKey,
        json_payload_value,
        validate_json_payload,
    )

    payload: dict[JsonPayloadKey, object] = {
        _PayloadKey.VALUE: {
            "nested": [_PayloadValue.VALUE, 3],
        },
        7: True,
    }

    assert validate_json_payload(payload) == {
        "enum_key": {"nested": ["enum_value", 3]},
        "7": True,
    }

    with pytest.raises(ValueError, match="unsupported value type: object"):
        json_payload_value(object())


def test_provider_modules_do_not_define_json_payload_normalizers() -> None:
    for path in (
        Path("src/rag_core/search/providers/qdrant_payloads.py"),
        Path("src/rag_core/search/providers/pgvector_payloads.py"),
        Path("src/rag_core/search/providers/turbopuffer_payloads.py"),
    ):
        source = path.read_text(encoding="utf-8")

        assert "def validate_json_payload" not in source
        assert "def json_payload_value" not in source
        assert "def _jsonish" not in source
        assert "def _payload_key" not in source


def test_document_records_own_lookup_validation_and_payload_extraction() -> None:
    from rag_core.search.document_records import (
        resolve_document_id_from_payload,
        stored_document_record_from_payload,
        validate_document_lookup_inputs,
    )

    namespace, corpus_id = validate_document_lookup_inputs(
        namespace=" team ",
        corpus_id=" corpus ",
        document_id=None,
        document_key="/docs/a.md",
    )

    assert (namespace, corpus_id) == ("team", "corpus")

    payload = {
        "document_id": "doc-1",
        "document_key": "/docs/a.md",
        "content_sha256": "sha",
        "processing_version": "v1",
    }
    document_id = resolve_document_id_from_payload(
        payload=payload,
        document_id_field=DEFAULT_POLICY.document_id_field,
        fallback_document_id=None,
        invalid_message="document lookup returned invalid document_id",
    )

    assert document_id == "doc-1"
    assert stored_document_record_from_payload(
        payload=payload,
        namespace=namespace,
        corpus_id=corpus_id,
        document_id=document_id,
        chunk_count=2,
        policy=DEFAULT_POLICY,
        invalid_field_message=(
            "document record payload field {field!r} must be a string"
        ),
    ).chunk_count == 2


def test_provider_modules_do_not_define_document_record_lookup_contracts() -> None:
    for path in (
        Path("src/rag_core/search/providers/qdrant_payloads.py"),
        Path("src/rag_core/search/providers/pgvector_store.py"),
        Path("src/rag_core/search/providers/turbopuffer_documents.py"),
        Path("src/rag_core/search/providers/memory_documents.py"),
    ):
        source = path.read_text(encoding="utf-8")

        assert "def _validate_document_lookup_inputs" not in source
        assert "def _payload_document_id" not in source
        assert "def _resolve_document_id" not in source
        assert "def stored_document_record_from_payload" not in source
        assert "def _stored_document_record_from_lookup" not in source
