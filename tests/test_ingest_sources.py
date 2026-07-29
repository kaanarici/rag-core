from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from rag_core.core_models import IngestedDocument
from rag_core.facade.ingest_sources import ingest_local_file_source
from rag_core.ingest.sources.local import LocalFileSourceReader
from rag_core.ingest.sources.local import document_key as local_document_key


def _ingested_document(*, document_id: str, document_key: str) -> IngestedDocument:
    return IngestedDocument(
        document_id=document_id,
        collection="help",
        namespace="acme",
        chunk_count=1,
        filename="single.md",
        mime_type="text/markdown",
        document_key=document_key,
    )


def test_ingest_local_file_source_defaults_document_key_to_canonical_path_across_path_spellings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    file_path = docs / "single.md"
    file_path.write_text("single", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    captured: list[dict[str, Any]] = []

    async def fake_ingest_bytes(**kwargs: Any) -> IngestedDocument:
        captured.append(kwargs)
        return _ingested_document(
            document_id=f"doc-{len(captured)}",
            document_key=str(kwargs["document_key"]),
        )

    async def scenario() -> None:
        await ingest_local_file_source(
            "docs/single.md",
            ingest_bytes=fake_ingest_bytes,
            namespace="acme",
            collection="help",
        )
        await ingest_local_file_source(
            "./docs/single.md",
            ingest_bytes=fake_ingest_bytes,
            namespace="acme",
            collection="help",
        )
        await ingest_local_file_source(
            file_path.resolve(),
            ingest_bytes=fake_ingest_bytes,
            namespace="acme",
            collection="help",
        )

    asyncio.run(scenario())

    expected_key = local_document_key(docs, file_path)
    assert [call["document_key"] for call in captured] == [
        expected_key,
        expected_key,
        expected_key,
    ]


def test_ingest_local_file_source_default_document_key_does_not_collide_on_basename(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    left_file = left / "same.md"
    right_file = right / "same.md"
    left_file.write_text("left", encoding="utf-8")
    right_file.write_text("right", encoding="utf-8")
    captured: list[dict[str, Any]] = []

    async def fake_ingest_bytes(**kwargs: Any) -> IngestedDocument:
        captured.append(kwargs)
        return _ingested_document(
            document_id=f"doc-{len(captured)}",
            document_key=str(kwargs["document_key"]),
        )

    async def scenario() -> None:
        await ingest_local_file_source(
            left_file,
            ingest_bytes=fake_ingest_bytes,
            namespace="acme",
            collection="help",
        )
        await ingest_local_file_source(
            right_file,
            ingest_bytes=fake_ingest_bytes,
            namespace="acme",
            collection="help",
        )

    asyncio.run(scenario())

    assert [call["document_key"] for call in captured] == [
        local_document_key(left, left_file),
        local_document_key(right, right_file),
    ]


def test_ingest_local_file_source_nested_path_matches_local_source_reader_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "docs"
    nested = docs / "nested"
    nested.mkdir(parents=True)
    file_path = nested / "item.md"
    file_path.write_text("nested", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    captured: list[dict[str, Any]] = []

    async def fake_ingest_bytes(**kwargs: Any) -> IngestedDocument:
        captured.append(kwargs)
        return _ingested_document(
            document_id="doc-nested",
            document_key=str(kwargs["document_key"]),
        )

    async def scenario() -> None:
        await ingest_local_file_source(
            "docs/nested/item.md",
            ingest_bytes=fake_ingest_bytes,
            namespace="acme",
            collection="help",
        )
        await ingest_local_file_source(
            file_path.resolve(),
            ingest_bytes=fake_ingest_bytes,
            namespace="acme",
            collection="help",
        )

    asyncio.run(scenario())

    assert [call["document_key"] for call in captured] == [
        LocalFileSourceReader().read("docs/nested/item.md").items[0].document_key,
        LocalFileSourceReader().read(file_path.resolve()).items[0].document_key,
    ]


def test_ingest_local_file_source_respects_explicit_document_key(tmp_path: Path) -> None:
    file_path = tmp_path / "single.md"
    file_path.write_text("single", encoding="utf-8")
    captured: list[dict[str, Any]] = []

    async def fake_ingest_bytes(**kwargs: Any) -> IngestedDocument:
        captured.append(kwargs)
        return _ingested_document(document_id="doc-explicit", document_key="explicit")

    asyncio.run(
        ingest_local_file_source(
            file_path,
            ingest_bytes=fake_ingest_bytes,
            namespace="acme",
            collection="help",
            document_key=" explicit-key ",
        )
    )

    assert captured[0]["document_key"] == "explicit-key"


def test_ingest_local_file_source_requires_document_key_when_path_has_parent_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace" / "project"
    workspace.mkdir(parents=True)
    file_path = tmp_path / "workspace" / "shared.md"
    file_path.write_text("single", encoding="utf-8")
    monkeypatch.chdir(workspace)

    async def fake_ingest_bytes(**kwargs: Any) -> IngestedDocument:
        return _ingested_document(
            document_id="doc-ambiguous",
            document_key=str(kwargs["document_key"]),
        )

    with pytest.raises(ValueError, match="document_key is required"):
        asyncio.run(
            ingest_local_file_source(
                "../shared.md",
                ingest_bytes=fake_ingest_bytes,
                namespace="acme",
                collection="help",
            )
        )


def test_ingest_local_file_source_rejects_symlink_file_path(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "external.md"
    target.write_text("external", encoding="utf-8")
    alias = docs / "alias.md"
    alias.symlink_to(target)

    async def fake_ingest_bytes(**kwargs: Any) -> IngestedDocument:
        return _ingested_document(
            document_id="doc-symlink",
            document_key=str(kwargs["document_key"]),
        )

    with pytest.raises(ValueError, match="does not allow symlink"):
        asyncio.run(
            ingest_local_file_source(
                alias,
                ingest_bytes=fake_ingest_bytes,
                namespace="acme",
                collection="help",
            )
        )


def test_ingest_local_file_source_rejects_symlink_parent_path(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    target = real / "external.md"
    target.write_text("external", encoding="utf-8")
    alias_dir = tmp_path / "alias"
    alias_dir.symlink_to(real, target_is_directory=True)

    async def fake_ingest_bytes(**kwargs: Any) -> IngestedDocument:
        return _ingested_document(
            document_id="doc-symlink-parent",
            document_key=str(kwargs["document_key"]),
        )

    with pytest.raises(ValueError, match="does not allow symlink"):
        asyncio.run(
            ingest_local_file_source(
                alias_dir / "external.md",
                ingest_bytes=fake_ingest_bytes,
                namespace="acme",
                collection="help",
            )
        )


def test_ingest_local_file_source_rejects_hardlinked_file_path(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "link"):
        pytest.skip("hardlinks are unavailable on this platform")
    source = tmp_path / "source.md"
    source.write_text("external", encoding="utf-8")
    alias = tmp_path / "alias.md"
    try:
        os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"hardlink support unavailable: {exc}")

    async def fake_ingest_bytes(**kwargs: Any) -> IngestedDocument:
        return _ingested_document(
            document_id="doc-hardlink",
            document_key=str(kwargs["document_key"]),
        )

    with pytest.raises(ValueError, match="does not allow multi-link"):
        asyncio.run(
            ingest_local_file_source(
                alias,
                ingest_bytes=fake_ingest_bytes,
                namespace="acme",
                collection="help",
            )
        )
