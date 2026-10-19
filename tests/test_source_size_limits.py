"""Bounded source reads and non-destructive rejection of oversized replacements."""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
import zipfile

import pytest

from rag_core import Config
from rag_core.config import IngestConfig
from rag_core.core import Engine
from rag_core.demo import DemoEmbeddingProvider
from rag_core.file_io import read_file_bytes
from rag_core.fetch_security import validate_fetch_url
from rag_core.fetching import FetchError, FetchResponse


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5, None])
def test_document_size_limit_requires_a_positive_integer(limit: Any) -> None:
    with pytest.raises(ValueError, match="max_document_bytes"):
        IngestConfig(max_document_bytes=limit)


def test_file_read_never_requests_more_than_limit_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * (1024 * 1024))
    original = Path.open
    requests: list[int] = []

    @contextmanager
    def tracked(self: Path, *args: Any, **kwargs: Any):
        with original(self, *args, **kwargs) as file:
            class Reader:
                def read(self, size: int = -1) -> bytes:
                    requests.append(size)
                    return cast(bytes, file.read(size))
            yield Reader()

    monkeypatch.setattr(Path, "open", tracked)

    async def run() -> None:
        with pytest.raises(ValueError, match="max_document_bytes"):
            await read_file_bytes(path, max_bytes=1024)
        assert requests == [1025]

    asyncio.run(run())


def test_injected_fetch_response_obeys_the_engine_document_cap() -> None:
    body = b"x" * 65

    class Client:
        def fetch(self, url: str) -> FetchResponse:
            validated = validate_fetch_url(url)
            return FetchResponse(
                url=validated, status_code=200, content_type="text/plain",
                content_length=len(body), content_sha256=hashlib.sha256(body).hexdigest(),
                body=body, redirect_chain=(validated,),
            )

    async def run() -> None:
        config = Config.local()
        config = replace(config, ingest=replace(config.ingest, max_document_bytes=64))
        async with Engine(config, embedding_provider=DemoEmbeddingProvider()) as core:
            with pytest.raises(FetchError, match="max_bytes"):
                await core.add_url("https://example.com/large.txt", fetch_client=Client(),
                                   tenant_id="acme", collection="docs")

    asyncio.run(run())


def test_all_local_entry_points_reject_oversize_without_losing_good_data(tmp_path: Path) -> None:
    async def run() -> None:
        base = Config.local()
        config = replace(base, ingest=replace(base.ingest, max_document_bytes=64))
        small = tmp_path / "small.txt"
        large = tmp_path / "large.txt"
        small.write_bytes(b"Invoices accept card.")
        large.write_bytes(b"x" * 65)
        scope: dict[str, Any] = dict(tenant_id="acme", collection="docs")
        async with Engine(config, embedding_provider=DemoEmbeddingProvider()) as core:
            original = await core.add_file(small, document_key="billing", **scope)
            with pytest.raises(ValueError, match="max_document_bytes"):
                await core.add_bytes(
                    file_bytes=b"x" * 65, filename="small.txt", mime_type="text/plain",
                    document_key="billing", **scope,
                )
            with pytest.raises(ValueError, match="max_document_bytes"):
                await core.add_file(large, document_key="billing", **scope)
            for operation in (core.parse_bytes, core.prepare_bytes):
                with pytest.raises(ValueError, match="max_document_bytes"):
                    await operation(file_bytes=b"x" * 65, filename="large.txt", mime_type="text/plain")
            with pytest.raises(ValueError, match="max_document_bytes"):
                await core.prepare_file(large)
            hits = await core.search(query="invoices", rerank=False, **scope)
            assert [hit.text for hit in hits] == ["Invoices accept card."]
            record = await core._store.get_document_record(document_id=original.document_id, **scope)
            assert record is not None and record.is_complete
            result = await core.add(tmp_path, **scope)
            assert result.succeeded_count == result.failed_count == 1
            archive = tmp_path / "oversized.zip"
            with zipfile.ZipFile(archive, "w") as file:
                file.writestr("large.txt", b"x" * 65)
            with pytest.raises(ValueError):
                await core.add_archive(archive, **scope)

    asyncio.run(run())
