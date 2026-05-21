import asyncio
import json

from rag_core import RAGCore
from rag_core.cli_doctor import _planned_runtime_payload

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_test_config,
)


def test_describe_runtime_reports_standard_source_processing_versions() -> None:
    async def scenario() -> dict[str, object]:
        core = RAGCore(
            make_test_config(
                qdrant_collection="rag_core_runtime_source_versions",
                embedding_dimensions=4,
                source_type="url",
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(),
        )
        try:
            return core.describe_runtime()
        finally:
            await core.close()

    payload = asyncio.run(scenario())

    assert json.loads(str(payload["processing_version"]))["source_type"] == "url"
    source_versions = payload["source_processing_versions"]
    assert isinstance(source_versions, dict)
    assert json.loads(str(source_versions["default"]))["source_type"] == "url"
    assert json.loads(str(source_versions["file"]))["source_type"] == "file"
    assert json.loads(str(source_versions["url"]))["source_type"] == "url"
    assert json.loads(str(source_versions["archive"]))["source_type"] == "archive"


def test_doctor_payload_reports_standard_source_processing_versions() -> None:
    payload = _planned_runtime_payload(
        make_test_config(
            qdrant_collection="rag_core_doctor_source_versions",
            embedding_dimensions=4,
            source_type="archive",
        )
    )

    assert json.loads(str(payload["processing_version"]))["source_type"] == "archive"
    source_versions = payload["source_processing_versions"]
    assert isinstance(source_versions, dict)
    assert json.loads(str(source_versions["default"]))["source_type"] == "archive"
    assert json.loads(str(source_versions["file"]))["source_type"] == "file"
    assert json.loads(str(source_versions["url"]))["source_type"] == "url"
    assert json.loads(str(source_versions["archive"]))["source_type"] == "archive"
