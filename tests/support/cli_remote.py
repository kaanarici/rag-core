from __future__ import annotations

from typing import Any, ClassVar

import pytest

from rag_core.core_models import IngestedDocument, RAGCoreConfig
from rag_core.fetch_security import ValidatedFetchUrl, validate_fetch_url
from rag_core.fetch_security_url import safe_remote_document_key, safe_remote_source_url


class FakeRemoteRAGCore:
    _last_instance: ClassVar[FakeRemoteRAGCore | None] = None

    def __init__(self, config: RAGCoreConfig, **kwargs: Any) -> None:
        self.config = config
        self.event_sink = kwargs.get("event_sink")
        self.ensure_ready_called = False
        self.closed = False
        self.ingest_url_calls: list[dict[str, Any]] = []
        type(self)._last_instance = self

    async def ensure_ready(self) -> None:
        self.ensure_ready_called = True

    async def ingest_url(self, url: str, **kwargs: Any) -> IngestedDocument:
        self.ingest_url_calls.append({"url": url, **kwargs})
        if "/fail" in url:
            raise RuntimeError(f"fetch exploded for {url}")
        validated_url = validate_fetch_url(url, policy=kwargs.get("fetch_policy"))
        redacted_url = safe_remote_source_url(validated_url)
        index = len(self.ingest_url_calls)
        return IngestedDocument(
            document_id=f"doc-{index}",
            corpus_id=kwargs["corpus_id"],
            namespace=kwargs["namespace"],
            chunk_count=1,
            filename="guide.txt",
            mime_type="text/plain",
            document_key=remote_url_key(validated_url),
            content_sha256=f"hash-{index}",
            ingest_state="created",
            processing_version='{"base_version":"rag_core_processing_v3","source_type":"url"}',
            metadata={"source_type": "url", "source_url": redacted_url},
        )

    async def close(self) -> None:
        self.closed = True


class FakeOpenAIError(Exception):
    __module__ = "openai"


def install_fake_remote_core(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_core import cli as cli_module

    FakeRemoteRAGCore._last_instance = None
    monkeypatch.setattr(cli_module, "RAGCore", FakeRemoteRAGCore)


def require_fake_remote_core() -> FakeRemoteRAGCore:
    instance = FakeRemoteRAGCore._last_instance
    if instance is None:
        raise AssertionError("FakeRemoteRAGCore was not initialized")
    return instance


def remote_url_key(url: ValidatedFetchUrl) -> str:
    return safe_remote_document_key(url)
