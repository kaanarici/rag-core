"""Targeted unit tests for the ingest and query CLI surface.

Full ingest+search end-to-end is already covered by ``test_local_ingest``
and ``test_indexer_behavior`` against fake providers. This file covers the
CLI-side seams:

- glob / directory expansion (``expand_supported_local_files``)
- argparse wiring (subcommands accept the documented flags)
- ``CoreIngestor`` writes a manifest entry when ``manifest_directory`` is set
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from rag_core.cli import _build_parser
from rag_core.core_ingest import CoreIngestor
from rag_core.core_lifecycle import compute_content_sha256
from rag_core.core_models import (
    CorpusManifestEntry,
    PreparedChunk,
    PreparedDocument,
    ProcessingFingerprint,
)
from rag_core.manifest_persistence import read_entries, write_entry
from rag_core.search.indexer import IndexResult
from rag_core.search.policy import DEFAULT_POLICY
from rag_core.search.types import StoredDocumentRecord
from rag_core.sources import expand_supported_local_files
from tests.support import RecordingVectorStore


# ---------------------------------------------------------------------------
# expand_supported_local_files
# ---------------------------------------------------------------------------


def test_expand_supported_local_files_literal_file(tmp_path: Path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hi", encoding="utf-8")
    assert expand_supported_local_files(str(target)) == [target]


def test_expand_supported_local_files_rejects_literal_symlink_file(tmp_path: Path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hi", encoding="utf-8")
    alias = tmp_path / "doc-alias.md"
    alias.symlink_to(target)
    assert expand_supported_local_files(str(alias)) == []


def test_expand_supported_local_files_literal_file_filters_unsupported_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "scan.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert expand_supported_local_files(str(target)) == []


def test_expand_supported_local_files_glob(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("c", encoding="utf-8")

    expanded = expand_supported_local_files(str(tmp_path / "*.md"))
    assert sorted(p.name for p in expanded) == ["a.md", "b.md"]


def test_expand_supported_local_files_recursive_glob(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "deep.md").write_text("d", encoding="utf-8")
    (tmp_path / "top.md").write_text("t", encoding="utf-8")

    expanded = expand_supported_local_files(str(tmp_path / "**" / "*.md"))
    assert sorted(p.name for p in expanded) == ["deep.md", "top.md"]


def test_expand_supported_local_files_directory_walks_recursive(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "archive.bin").write_bytes(b"\x00\x01")
    (tmp_path / "~$draft.docx").write_bytes(b"not a real docx")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.cpython-314.pyc").write_bytes(b"generated")

    expanded = expand_supported_local_files(str(tmp_path))
    assert sorted(p.name for p in expanded) == ["a.md", "b.md"]


def test_expand_supported_local_files_wildcard_filters_unsupported_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "archive.bin").write_bytes(b"\x00\x01")

    expanded = expand_supported_local_files(str(tmp_path / "*"))
    assert sorted(p.name for p in expanded) == ["a.md"]


@pytest.mark.parametrize("missing", ("nope.md", ""))
def test_expand_supported_local_files_missing_returns_empty(
    tmp_path: Path, missing: str
) -> None:
    target = str(tmp_path / missing) if missing else missing
    assert expand_supported_local_files(target) == []


# ---------------------------------------------------------------------------
# argparse wiring — focused on flags that drive behavior elsewhere in the CLI.
# ---------------------------------------------------------------------------


def test_ingest_subparser_accepts_documented_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "ingest",
            "./docs",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--force-reindex",
            "--manifest-dir",
            "/tmp/manifest",
            "--metadata",
            "team=search",
            "--qdrant-url",
            "http://localhost:6333",
            "--vector-store",
            "qdrant",
            "--embedding-dimensions",
            "1536",
            "--events-jsonl",
            "/tmp/events.jsonl",
            "--plan-json",
            "--json",
        ]
    )
    assert args.command == "ingest"
    assert args.path == "./docs"
    assert args.namespace == "acme"
    assert args.corpus_id == "help"
    assert args.force_reindex is True
    assert args.manifest_dir == "/tmp/manifest"
    assert args.metadata == ["team=search"]
    assert args.embedding_dimensions == 1536
    assert args.plan_json is True
    assert args.json is True


def test_local_search_subparser_accepts_events_jsonl() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["local-search", "./docs", "billing policy", "--events-jsonl", "/tmp/events.jsonl"]
    )
    assert args.command == "local-search"
    assert args.events_jsonl == "/tmp/events.jsonl"


def test_ingest_help_describes_supported_file_filter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["ingest", "--help"])

    help_text = capsys.readouterr().out
    assert "Ingest supported local files" in help_text
    assert "Supported file, directory, or shell glob" in help_text
    assert "--plan-json" in help_text


def test_query_subparser_accepts_repeating_corpus_id() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "search",
            "billing policy",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--corpus-id",
            "internal",
            "--limit",
            "20",
            "--rerank",
            "--qdrant-url",
            "http://localhost:6333",
            "--events-jsonl",
            "/tmp/events.jsonl",
        ]
    )
    assert args.command == "search"
    assert args.text == "billing policy"
    assert args.corpus_id == ["help", "internal"]
    assert args.limit == 20
    assert args.rerank is True


# ---------------------------------------------------------------------------
# CoreIngestor manifest behavior
# ---------------------------------------------------------------------------


def _make_ingestor(
    *,
    manifest_directory: Path | None,
    store: RecordingVectorStore | None = None,
) -> tuple[CoreIngestor, AsyncMock]:
    indexer = AsyncMock()
    indexer.index_document = AsyncMock(
        return_value=IndexResult(
            document_id="doc-1",
            chunk_count=2,
            point_ids=["p-0", "p-1"],
            point_payloads=[{"text": "x"}, {"text": "y"}],
            document_key="doc.md",
            content_sha256="cafe",
        )
    )

    async def _prepare_bytes(
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
    ) -> PreparedDocument:
        return PreparedDocument(
            filename=filename,
            mime_type=mime_type,
            markdown="hi",
            chunks=[
                PreparedChunk(
                    chunk_index=0, text="hi", embedding_text="hi", word_count=1
                ),
            ],
        )

    ingestor = CoreIngestor(
        collection_name="rag_core_test",
        source_type="file",
        embedding_model="fake-embedding",
        processing_version=ProcessingFingerprint(
            base_version="rag_core_processing_v1", source_type="file"
        ),
        store=store or RecordingVectorStore(),
        indexer=indexer,
        sidecar=None,
        prepare_bytes=_prepare_bytes,
        manifest_directory=manifest_directory,
    )
    return ingestor, indexer


def test_core_ingestor_writes_manifest_entry(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifest"

    async def go() -> None:
        ingestor, _ = _make_ingestor(manifest_directory=manifest_dir)
        await ingestor.ingest_bytes(
            file_bytes=b"hello",
            filename="doc.md",
            mime_type="text/markdown",
            namespace="acme",
            corpus_id="help",
        )

    asyncio.run(go())
    entries = read_entries(manifest_dir, namespace="acme", corpus_id="help")
    assert len(entries) == 1
    assert entries[0].document_id.startswith("doc_")
    assert entries[0].filename == "doc.md"
    assert entries[0].chunk_count == 2


def test_core_ingestor_heals_missing_manifest_without_preparing_unchanged_retry(
    tmp_path: Path,
) -> None:
    manifest_dir = tmp_path / "manifest"
    document_id = DEFAULT_POLICY.make_document_id(
        namespace="acme",
        corpus_id="help",
        document_key="doc.md",
    )
    processing_version = ProcessingFingerprint(
        base_version="rag_core_processing_v1",
        source_type="file",
    ).serialize()
    store = RecordingVectorStore(
        document_records={
            ("acme", "help", document_id): StoredDocumentRecord(
                document_id=document_id,
                namespace="acme",
                corpus_id="help",
                document_key="doc.md",
                content_sha256=compute_content_sha256(b"hello"),
                processing_version=processing_version,
                chunk_count=2,
            )
        }
    )

    prepare_calls = 0

    async def prepare_bytes(
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
    ) -> PreparedDocument:
        nonlocal prepare_calls
        prepare_calls += 1
        return PreparedDocument(
            filename=filename,
            mime_type=mime_type,
            markdown="hello",
            chunks=[
                PreparedChunk(
                    chunk_index=0,
                    text="hello",
                    embedding_text="hello",
                    word_count=1,
                )
            ],
        )

    async def go() -> AsyncMock:
        ingestor, indexer = _make_ingestor(
            manifest_directory=manifest_dir,
            store=store,
        )
        ingestor._prepare_bytes = prepare_bytes
        await ingestor.ingest_bytes(
            file_bytes=b"hello",
            filename="doc.md",
            mime_type="text/markdown",
            namespace="acme",
            corpus_id="help",
        )
        return indexer

    indexer = asyncio.run(go())
    indexer.index_document.assert_not_awaited()
    assert prepare_calls == 1
    entries = read_entries(manifest_dir, namespace="acme", corpus_id="help")
    assert len(entries) == 1
    assert entries[0].document_id == document_id
    assert entries[0].document_key == "doc.md"
    assert entries[0].content_sha256 == compute_content_sha256(b"hello")


def test_core_ingestor_unchanged_retry_preserves_manifest_metadata(
    tmp_path: Path,
) -> None:
    manifest_dir = tmp_path / "manifest"
    document_id = DEFAULT_POLICY.make_document_id(
        namespace="acme",
        corpus_id="help",
        document_key="doc.md",
    )
    content_sha256 = compute_content_sha256(b"hello")
    processing_version = ProcessingFingerprint(
        base_version="rag_core_processing_v1",
        source_type="file",
    ).serialize()
    write_entry(
        manifest_dir,
        CorpusManifestEntry(
            document_id=document_id,
            namespace="acme",
            corpus_id="help",
            document_key="doc.md",
            content_sha256=content_sha256,
            filename="doc.md",
            mime_type="text/markdown",
            chunk_count=2,
            parser="local:text",
            needs_ocr=True,
            metadata={"title": "First Title", "ocr_page_indices": [0, 2]},
        ),
    )
    store = RecordingVectorStore(
        document_records={
            ("acme", "help", document_id): StoredDocumentRecord(
                document_id=document_id,
                namespace="acme",
                corpus_id="help",
                document_key="doc.md",
                content_sha256=content_sha256,
                processing_version=processing_version,
                chunk_count=2,
            )
        }
    )

    async def go() -> AsyncMock:
        ingestor, indexer = _make_ingestor(
            manifest_directory=manifest_dir,
            store=store,
        )
        await ingestor.ingest_bytes(
            file_bytes=b"hello",
            filename="doc.md",
            mime_type="text/markdown",
            namespace="acme",
            corpus_id="help",
            metadata={"title": "Second Title"},
        )
        return indexer

    indexer = asyncio.run(go())
    indexer.index_document.assert_not_awaited()
    entries = read_entries(manifest_dir, namespace="acme", corpus_id="help")
    assert len(entries) == 1
    assert entries[0].metadata == {"title": "Second Title", "ocr_page_indices": [0, 2]}
    assert entries[0].parser == "local:text"
    assert entries[0].needs_ocr is True


def test_core_ingestor_heals_stale_manifest_identity_without_metadata_drift(
    tmp_path: Path,
) -> None:
    manifest_dir = tmp_path / "manifest"
    document_id = DEFAULT_POLICY.make_document_id(
        namespace="acme",
        corpus_id="help",
        document_key="doc.md",
    )
    content_sha256 = compute_content_sha256(b"hello")
    processing_version = ProcessingFingerprint(
        base_version="rag_core_processing_v1",
        source_type="file",
    ).serialize()
    write_entry(
        manifest_dir,
        CorpusManifestEntry(
            document_id=document_id,
            namespace="acme",
            corpus_id="help",
            document_key="doc.md",
            content_sha256="old-sha",
            filename="doc.md",
            mime_type="text/markdown",
            chunk_count=2,
            metadata={"title": "Old Title"},
        ),
    )
    store = RecordingVectorStore(
        document_records={
            ("acme", "help", document_id): StoredDocumentRecord(
                document_id=document_id,
                namespace="acme",
                corpus_id="help",
                document_key="doc.md",
                content_sha256=content_sha256,
                processing_version=processing_version,
                chunk_count=2,
            )
        }
    )

    async def go() -> AsyncMock:
        ingestor, indexer = _make_ingestor(
            manifest_directory=manifest_dir,
            store=store,
        )
        await ingestor.ingest_bytes(
            file_bytes=b"hello",
            filename="doc.md",
            mime_type="text/markdown",
            namespace="acme",
            corpus_id="help",
            metadata={"title": "Current Title"},
        )
        return indexer

    indexer = asyncio.run(go())
    indexer.index_document.assert_not_awaited()
    entries = read_entries(manifest_dir, namespace="acme", corpus_id="help")
    assert len(entries) == 1
    assert entries[0].content_sha256 == content_sha256
    assert entries[0].metadata == {"title": "Current Title"}


def test_core_ingestor_skips_manifest_when_directory_unset(tmp_path: Path) -> None:
    async def go() -> None:
        ingestor, _ = _make_ingestor(manifest_directory=None)
        await ingestor.ingest_bytes(
            file_bytes=b"hello",
            filename="doc.md",
            mime_type="text/markdown",
            namespace="acme",
            corpus_id="help",
        )

    asyncio.run(go())
    # Nothing was passed in, so nothing should land on disk.
    assert not list(tmp_path.iterdir())


def test_core_ingestor_rejects_bad_manifest_scope_before_indexing(tmp_path: Path) -> None:
    async def go() -> None:
        ingestor, indexer = _make_ingestor(manifest_directory=tmp_path / "manifest")
        with pytest.raises(ValueError, match="single non-empty path segment"):
            await ingestor.ingest_bytes(
                file_bytes=b"hello",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="../escape",
                corpus_id="help",
            )
        indexer.index_document.assert_not_awaited()

    asyncio.run(go())
    assert not list(tmp_path.iterdir())
