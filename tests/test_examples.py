"""Examples must keep running against the public API."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest

from examples.chatbot_context import run_chatbot_context_demo
from examples.corpus_lifecycle import (
    CorpusManifestEntry,
    delete_from_manifest,
    ingest_into_manifest,
    manifest_key,
    manifest_row,
    preview_text,
    search_corpus,
)
from examples.minimal_app import run_demo as run_minimal_app_demo
from examples.pdf_ocr_path import describe_pdf_runtime, inspect_pdf_route, prepare_pdf_for_ingest
from examples.retrieval_eval import run_demo as run_retrieval_eval_demo
from examples.source_ingest import run_demo as run_source_ingest_demo
from rag_core import OcrRoutingSignal, ParsedDocument, PreparedChunk, PreparedDocument, RAGCore
from rag_core.documents import build_mistral_ocr_provider
from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
    make_test_config,
)


def _module_env() -> dict[str, str]:
    env = os.environ.copy()
    root = Path.cwd()
    pythonpath = [str(root / "src"), str(root)]
    if existing := env.get("PYTHONPATH"):
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def test_minimal_app_runs_as_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.minimal_app"],
        check=True,
        cwd=Path.cwd(),
        env=_module_env(),
        capture_output=True,
        text=True,
    )

    assert "Context to pass into your model call:" in result.stdout
    assert "Citations:" in result.stdout
    assert "billing.txt" in result.stdout


def test_search_endpoint_runs_as_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.search_endpoint"],
        check=True,
        cwd=Path.cwd(),
        env=_module_env(),
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["citations"]
    assert payload["source_previews"]


def test_vercel_ai_sdk_example_matches_public_tool_payload() -> None:
    source = Path("examples/vercel_ai_sdk_search_tool.ts").read_text(encoding="utf-8")
    assert "document_path" not in source
    assert "isSearchSnippet" in source
    assert "isSourcePreview" in source
    assert "typeof payload.citation_summary" in source
    assert "pattern: \"\\\\S\"" in source
    assert "retrieval_metadata" in source
    assert "Number.isFinite" in source
    assert "value.bbox.every(isFiniteNumber)" in source
    assert "summarizeSearchResult" in source
    assert "console.log(result.toolResults)" not in source


def test_source_ingest_runs_as_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.source_ingest"],
        check=True,
        cwd=Path.cwd(),
        env=_module_env(),
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["archive_written_count"] == 1
    assert payload["citation_count"] >= 3
    assert payload["local_document_key"] == "local-guide.md"
    assert str(payload["remote_document_key"]).startswith("url:https://example.com/")
    assert str(payload["context_text"]).count("**Path**: /") == 0
    assert "docs.zip!/" not in str(payload["context_text"])
    assert "token=secret" not in result.stdout


def test_corpus_lifecycle_example_tracks_manifest_and_delete() -> None:
    async def go() -> None:
        store = RecordingVectorStore(
            search_results=[
                make_search_result(
                    id="hit-1",
                    text="billing answers stay searchable",
                    score=0.92,
                    document_id="doc-from-search",
                    corpus_id="help-center",
                )
            ]
        )
        core = RAGCore(
            make_test_config(
                qdrant_collection="rag_core_examples_lifecycle",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        manifest: dict[str, CorpusManifestEntry] = {}

        try:
            entry = await ingest_into_manifest(
                core,
                manifest=manifest,
                file_bytes=b"billing answers stay searchable",
                filename="faq.txt",
                mime_type="text/plain",
                namespace="acme",
                corpus_id="help-center",
                metadata={"source": "seed"},
            )
            key = manifest_key(
                namespace="acme",
                corpus_id="help-center",
                document_key=entry.document_key or "faq.txt",
            )
            assert manifest[key] == entry
            assert manifest_row(entry)["parser"] == "local:text"

            hits = await search_corpus(core, entry=entry, query="billing", limit=3)
            assert hits[0].document_id == "doc-from-search"
            assert preview_text(hits[0]) == "billing answers stay searchable"
            assert store.search_calls[0].corpus_ids == ["help-center"]

            deleted = await delete_from_manifest(core, manifest=manifest, key=key)
            assert deleted.document_id == entry.document_id
            assert key not in manifest
            assert store.delete_calls[0].document_id == entry.document_id
        finally:
            await core.close()

    asyncio.run(go())


def test_pdf_example_reports_route_and_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_parse_bytes(
        *, file_bytes: bytes, filename: str, mime_type: str, path: str | None = None
    ) -> ParsedDocument:
        return ParsedDocument(
            filename=filename,
            mime_type=mime_type,
            markdown="",
            metadata={"parser": "local:pdf_inspector", "needs_ocr": True, "ocr_page_indices": [2, 0, 2]},
            path=path,
        )

    async def fake_prepare_bytes(
        *, file_bytes: bytes, filename: str, mime_type: str, path: str | None = None
    ) -> PreparedDocument:
        return PreparedDocument(
            filename=filename,
            mime_type=mime_type,
            markdown="# OCR text",
            chunks=[
                PreparedChunk(
                    chunk_index=0,
                    text="# OCR text",
                    embedding_text="# OCR text",
                    word_count=3,
                )
            ],
            metadata={"parser": "local:pdf_inspector", "needs_ocr": False},
            path=path,
            ocr=OcrRoutingSignal(needed=False, page_indices=[], parser="local:pdf_inspector"),
        )

    async def go() -> None:
        core = RAGCore(
            make_test_config(
                qdrant_collection="rag_core_examples_pdf",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(),
            ocr_provider=build_mistral_ocr_provider(python_executable="/tmp/python"),
        )
        try:
            monkeypatch.setattr(core, "parse_bytes", fake_parse_bytes)
            monkeypatch.setattr(core, "prepare_bytes", fake_prepare_bytes)

            route = await inspect_pdf_route(
                core, file_bytes=b"%PDF", filename="scan.pdf", path="/tmp/scan.pdf"
            )
            assert route == {
                "parser": "local:pdf_inspector",
                "needs_ocr": True,
                "ocr_page_indices": [0, 2],
            }

            prepared = await prepare_pdf_for_ingest(
                core, file_bytes=b"%PDF", filename="scan.pdf"
            )
            assert prepared.markdown == "# OCR text"
            assert prepared.ocr.needed is False

            runtime = describe_pdf_runtime(core)
            assert runtime["ocr"] == {
                "provider": "mistral",
                "model": "mistral-ocr-latest",
                "supports_page_selection": True,
            }
            assert isinstance(runtime["pdf_inspector"], dict)
        finally:
            await core.close()

    asyncio.run(go())


def test_minimal_app_demo_runs() -> None:
    asyncio.run(run_minimal_app_demo())


def test_chatbot_context_example_returns_model_context() -> None:
    context = asyncio.run(run_chatbot_context_demo("How do I pay an invoice?"))

    assert context.snippets
    assert context.snippets[0].citation_id
    assert "billing" in context.as_text().lower()


def test_retrieval_eval_example_reports_balanced_profile() -> None:
    report = asyncio.run(run_retrieval_eval_demo())
    run = cast(dict[str, object], report["run"])

    assert run["search_profile"] == "balanced"


def test_retrieval_eval_module_prints_redacted_report() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.retrieval_eval"],
        cwd=Path.cwd(),
        env=_module_env(),
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["cases"][0]["case_label"] == "case-1"
    assert "case_id" not in payload["cases"][0]
    assert "query" not in payload["cases"][0]
    assert "expected_chunk_ids" not in payload["cases"][0]
    assert "How can a customer update payment details?" not in result.stdout


def test_source_ingest_example_covers_local_archive_and_url_sources() -> None:
    payload = asyncio.run(run_source_ingest_demo())

    assert payload["local_document_key"] == "local-guide.md"
    assert payload["archive_written_count"] == 1
    remote_document_key = str(payload["remote_document_key"])
    assert remote_document_key == "url:https://example.com/docs/remote-guide?redacted"
    assert cast(int, payload["citation_count"]) >= 3
    context_text = str(payload["context_text"]).lower()
    assert "local files" in context_text
    assert "zip members" in context_text
    assert "remote guide" in context_text
    assert "token=secret" not in context_text
