"""Examples must keep running against the public surface."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

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
from examples.pdf_ocr_path import (
    describe_pdf_runtime,
    inspect_pdf_route,
    prepare_pdf_for_ingest,
)
from rag_core import (
    OcrRoutingSignal,
    ParsedDocument,
    PreparedChunk,
    PreparedDocument,
    RAGCore,
)
from rag_core.contracts import (
    SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
    SEARCH_USER_DOCUMENTS_DEFAULT_RERANK,
    SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH,
    SEARCH_USER_DOCUMENTS_LIMIT_MAX,
    SEARCH_USER_DOCUMENTS_LIMIT_MIN,
    SEARCH_USER_DOCUMENTS_MAX_CHARS_MAX,
    SEARCH_USER_DOCUMENTS_MAX_CHARS_MIN,
    SEARCH_USER_DOCUMENTS_MAX_TOKENS_MAX,
    SEARCH_USER_DOCUMENTS_MAX_TOKENS_MIN,
)
from rag_core.documents import build_mistral_ocr_provider
from rag_core.documents.ocr_provider_names import (
    DEFAULT_MISTRAL_OCR_MODEL,
    MISTRAL_OCR_PROVIDER,
)
from rag_core.retrieval_defaults import DEFAULT_SEARCH_LIMIT
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


def test_rag_core_quickstart_runs_as_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rag_core.quickstart"],
        check=True,
        cwd=Path.cwd(),
        env=_module_env(),
        capture_output=True,
        text=True,
    )
    assert "Prompt-safe context text:" in result.stdout
    assert "Citations:" in result.stdout


def test_embedded_service_runs_as_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.embedded_service"],
        check=True,
        cwd=Path.cwd(),
        env=_module_env(),
        capture_output=True,
        text=True,
    )
    assert "card" in result.stdout.lower() or "ach" in result.stdout.lower()


def test_minimal_app_runs_as_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.minimal_app"],
        check=True,
        cwd=Path.cwd(),
        env=_module_env(),
        capture_output=True,
        text=True,
    )

    assert "Prompt-safe context text:" in result.stdout
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
    assert "PromptSourceReference" in source
    assert "PromptSourceLocator" in source
    assert "PromptSourcePreview" in source
    assert "isPromptSourceReference" in source
    assert "isPromptSourceLocator" in source
    assert "isPromptSourcePreview" in source
    assert "type SourceReference =" not in source
    assert "type SourceLocator =" not in source
    assert "isSourceReference" not in source
    assert "isSourceLocator" not in source
    assert "isSourcePreview" not in source
    assert "typeof payload.citation_summary" in source
    assert 'pattern: "\\\\S"' in source
    assert (
        "limit: { "
        f'type: "integer", minimum: {SEARCH_USER_DOCUMENTS_LIMIT_MIN}, '
        f"maximum: {SEARCH_USER_DOCUMENTS_LIMIT_MAX}, "
        f"default: {SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT} "
        "}" in source
    )
    assert (
        f'rerank: {{ type: "boolean", default: {str(SEARCH_USER_DOCUMENTS_DEFAULT_RERANK).lower()} }}'
        in source
    )
    assert "use_lexical_search: {" in source
    assert (
        f"default: {str(SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH).lower()},"
        in source
    )
    assert (
        "Controls configured lexical/exact-match expansion only; "
        "query-plan defaults remain provider capability-aware."
    ) in source
    assert (
        "max_chars: { "
        f'type: "integer", minimum: {SEARCH_USER_DOCUMENTS_MAX_CHARS_MIN}, '
        f"maximum: {SEARCH_USER_DOCUMENTS_MAX_CHARS_MAX}, "
        f"default: {SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS} "
        "}" in source
    )
    assert (
        "max_tokens: { "
        f'type: "integer", minimum: {SEARCH_USER_DOCUMENTS_MAX_TOKENS_MIN}, '
        f"maximum: {SEARCH_USER_DOCUMENTS_MAX_TOKENS_MAX} "
        "}" in source
    )
    assert "citation_id: string;" in source
    assert "source_id: string;" not in source
    assert "result_id: string;" not in source
    assert "source_hash: string | null;" not in source
    assert "line_start: number | null;" in source
    assert "line_end: number | null;" in source
    assert "inputSchema: searchUserDocumentsInputSchema" in source
    assert "parameters: searchUserDocumentsInputSchema" not in source
    assert "parameters: {" not in source
    assert 'part.type === "text-delta"' in source
    assert 'part.type === "text"' not in source
    assert "retrieval_metadata" in source
    assert "quality?:" in source
    assert 'hasExactKeys(value, ["quality", "rerank"])' in source
    assert "isSnippetQualityMetadata" in source
    assert "Number.isFinite" in source
    assert "value.bbox.every(isFiniteNumber)" in source
    assert "summarizeSearchResult" in source
    assert '"anthropic/claude-sonnet-4.6"' in source
    assert "claude-sonnet-4.5" not in source
    assert "toModelOutput: ({ output })" in source
    assert "formatSearchResultForModel(output)" in source
    assert 'type: "text"' in source
    assert "console.log(result.toolResults)" not in source
    assert "part.output" in source
    assert "part.result" not in source
    assert 'part.type === "tool-error"' in source
    assert 'part.type === "error"' in source
    assert "summarizeUnknownError(part.error)" in source

    docs = Path("docs/embed.md").read_text(encoding="utf-8")
    assert "stable AI SDK v6 tool contracts verified by" in docs
    assert "`./scripts/verify_vercel_ai_sdk_example.sh`" in docs
    assert "current `ai@^6.0.0`" in docs
    assert "TypeScript declarations" in docs
    assert "`text` tool output" in docs
    assert "`text-delta` parts on `fullStream`" in docs
    assert "`tool-error` and `error` stream parts" in docs
    assert "verify the current Vercel AI" in docs
    assert "Gateway model list before copying it into an application" in docs
    assert "not a v7 beta contract" in docs

    expectations = Path("docs/expectations.md").read_text(encoding="utf-8")
    assert "prompt-safe preview and" in expectations
    assert "locator projections in `to_prompt_payload()`" in expectations

    verify_script = Path("scripts/verify_vercel_ai_sdk_example.sh").read_text(
        encoding="utf-8"
    )
    assert "ai\": \"^6.0.0\"" in verify_script
    assert "@types/json-schema" in verify_script
    assert "ai@${pkg.version}" in verify_script
    assert "tsc --noEmit" in verify_script


def test_examples_use_prompt_safe_text() -> None:
    for path in (
        Path("docs/embed.md"),
        Path("examples/chatbot_context.py"),
        Path("examples/configured_retrieval.py"),
        Path("examples/embedded_service.py"),
        Path("examples/minimal_app.py"),
        Path("examples/source_ingest.py"),
        Path("src/rag_core/quickstart.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert ".as_prompt_text()" in source
        assert ".as_text()" not in source


def test_configured_retrieval_ci_wording_is_precise() -> None:
    example = Path("examples/configured_retrieval.py").read_text(encoding="utf-8")
    docs = Path("docs/embed.md").read_text(encoding="utf-8")

    for source in (example, docs):
        assert "not executed by default CI because it requires credentials" in source
        assert "source-checked and packaged" in source
        assert "Not run in CI" not in source
        assert "Not part of default CI" not in source


def test_prompt_context_examples_use_prompt_safe_citation_summary() -> None:
    for path in (
        Path("examples/chatbot_context.py"),
        Path("examples/configured_retrieval.py"),
        Path("examples/minimal_app.py"),
        Path("src/rag_core/quickstart.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert ".prompt_citation_summary" in source
        assert ".citations" not in source
        assert ".source_id" not in source
        assert ".result_id" not in source
        assert ".document_key" not in source


def test_python_examples_teach_public_root_config_import() -> None:
    for path in Path("examples").glob("*.py"):
        source = path.read_text(encoding="utf-8")

        assert "from rag_core.core_models import RAGCoreConfig" not in source
        assert "from rag_core.core import RAGCore" not in source
        assert "from rag_core.remote_document_keys import" not in source


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
    assert (
        payload["remote_document_key"]
        == "url:https://example.com/docs/remote-guide?redacted"
    )
    raw_context_text = str(payload["context_text"])
    context_text = raw_context_text.lower()
    assert "local files" in context_text
    assert "zip members" in context_text
    assert "remote guide" in context_text
    assert raw_context_text.count("**Path**: /") == 0
    assert "docs.zip!/" not in context_text
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


def test_corpus_lifecycle_search_helper_uses_shared_search_default() -> None:
    signature = inspect.signature(search_corpus)
    assert signature.parameters["limit"].default == DEFAULT_SEARCH_LIMIT


def test_pdf_example_reports_route_and_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_parse_bytes(
        *, file_bytes: bytes, filename: str, mime_type: str, path: str | None = None
    ) -> ParsedDocument:
        return ParsedDocument(
            filename=filename,
            mime_type=mime_type,
            markdown="",
            metadata={
                "parser": "local:pdf_inspector",
                "needs_ocr": True,
                "ocr_page_indices": [2, 0, 2],
            },
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
            ocr=OcrRoutingSignal(
                needed=False, page_indices=[], parser="local:pdf_inspector"
            ),
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
                "provider": MISTRAL_OCR_PROVIDER,
                "model": DEFAULT_MISTRAL_OCR_MODEL,
                "supports_page_selection": True,
            }
            assert isinstance(runtime["pdf_inspector"], dict)
        finally:
            await core.close()

    asyncio.run(go())


def test_chatbot_context_example_returns_context_pack() -> None:
    context = asyncio.run(run_chatbot_context_demo("How do I pay an invoice?"))

    assert context.snippets
    assert context.snippets[0].citation_id
    assert "billing" in context.as_prompt_text().lower()


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
    assert payload["run"]["search_profile"] == "balanced"
    assert payload["cases"][0]["case_label"] == "case-1"
    assert "case_id" not in payload["cases"][0]
    assert "query" not in payload["cases"][0]
    assert "expected_ids" not in payload["cases"][0]
    assert "expected_chunk_ids" not in payload["cases"][0]
    assert "How can a customer update payment details?" not in result.stdout
