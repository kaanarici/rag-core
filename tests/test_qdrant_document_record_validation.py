from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from rag_core.search.providers.qdrant_payloads import _count_result_value
from rag_core.search.providers.qdrant_store import QdrantVectorStore

SECRET = "sk-test-secret"


class _FakeQdrantClient:
    def __init__(self, count: object) -> None:
        self.count_result = count

    async def get_collections(self) -> object:
        return SimpleNamespace(collections=[SimpleNamespace(name="docs")])

    async def get_collection(self, *, collection_name: str) -> object:
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(size=3072),
                    sparse_vectors=None,
                )
            )
        )

    async def scroll(self, **kwargs: Any) -> tuple[list[object], None]:
        return (
            [
                SimpleNamespace(
                    payload={
                        "document_id": "doc-1",
                        "document_key": "/docs/private.md",
                        "content_sha256": "sha",
                        "processing_version": "v1",
                        "text": "private " + SECRET,
                    }
                )
            ],
            None,
        )

    async def count(self, **kwargs: Any) -> object:
        return SimpleNamespace(count=self.count_result)

    async def close(self) -> None:
        return None


def _store(client: object) -> QdrantVectorStore:
    store = QdrantVectorStore(
        url=None,
        api_key=None,
        collection_name="docs",
        location=":memory:",
    )
    store._client = cast(Any, client)
    store._collection_state.ready = True
    return store


@pytest.mark.parametrize(
    "count",
    [
        pytest.param(True, id="bool"),
        pytest.param(-1, id="negative-int"),
        pytest.param("-1", id="negative-string"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(1.5, id="fractional-float"),
        pytest.param(
            "not-a-number\nTraceback (most recent call last): " + SECRET,
            id="secret-string",
        ),
    ],
)
def test_qdrant_document_record_rejects_malformed_count_without_leaking(
    count: object,
) -> None:
    async def _run() -> None:
        with pytest.raises(ValueError, match="document count response"):
            await _store(_FakeQdrantClient(count)).get_document_record(
                namespace="team-space",
                corpus_id="corpus-a",
                document_id="doc-1",
            )

    asyncio.run(_run())


def test_qdrant_count_result_value_accepts_zero() -> None:
    assert _count_result_value(0) == 0


def test_qdrant_count_result_value_rejects_fractional_float() -> None:
    with pytest.raises(ValueError, match="document count response"):
        _count_result_value(1.5)


def test_qdrant_document_record_uses_valid_count() -> None:
    async def _run() -> None:
        record = await _store(_FakeQdrantClient(7)).get_document_record(
            namespace="team-space",
            corpus_id="corpus-a",
            document_id="doc-1",
        )

        assert record is not None
        assert record.chunk_count == 7

    asyncio.run(_run())
