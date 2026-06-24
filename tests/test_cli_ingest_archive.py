from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest

from rag_core.cli import main
from rag_core.core_models import CollectionManifestEntry, Config
from rag_core.ingest.local.models import LocalIngestResult, LocalIngestSuccess
from rag_core.manifest.persistence import write_entry
from rag_core.ingest.sources.archive_policy import archive_document_key


class _FakeEngine:
    _last_instance: "_FakeEngine | None" = None

    def __init__(self, config: Config, **kwargs: Any) -> None:
        self.config = config
        self.event_sink = kwargs.get("event_sink")
        self.ensure_ready_called = False
        self.closed = False
        self.ingest_archive_calls: list[dict[str, Any]] = []
        type(self)._last_instance = self

    async def ensure_ready(self) -> None:
        self.ensure_ready_called = True

    async def add_archive(self, archive_path: str, **kwargs: Any) -> LocalIngestResult:
        self.ingest_archive_calls.append({"archive_path": archive_path, **kwargs})
        return LocalIngestResult(
            namespace=kwargs["namespace"],
            collection=kwargs["collection"],
            records=(
                LocalIngestSuccess(
                    path=f"{Path(archive_path).name}!/docs/guide.md",
                    document_key=archive_document_key(
                        Path(archive_path),
                        "docs/guide.md",
                    ),
                    content_sha256="hash-guide",
                    document_id="doc-guide",
                    filename="guide.md",
                    chunk_count=2,
                    ingest_state="created",
                    replaced_existing=False,
                    manifest_status="missing",
                    manifest_reason="source_not_in_manifest",
                ),
            ),
        )

    async def close(self) -> None:
        self.closed = True


class _FakeOpenAIError(Exception):
    __module__ = "openai"


def test_ingest_archive_plan_json_reconciles_without_runtime_setup(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    _FakeEngine._last_instance = None
    monkeypatch.setattr(cli_module, "Engine", _FakeEngine)
    archive_path = tmp_path / "docs.zip"
    guide_body = b"# Guide\n\nAlpha"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("docs/guide.md", guide_body)
        archive.writestr("docs/reference.md", b"# Reference\n\nBeta")
        archive.writestr("assets/logo.png", b"\x89PNG\r\n\x1a\n")
    manifest_dir = tmp_path / "manifest"
    write_entry(
        manifest_dir,
        _manifest_entry(
            document_key=archive_document_key(archive_path, "docs/guide.md"),
            content_sha256=hashlib.sha256(guide_body).hexdigest(),
        ),
    )

    exit_code = main(
        [
            "add",
            str(archive_path),
            "--namespace",
            "acme",
            "--collection",
            "help",
            "--manifest-dir",
            str(manifest_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert _FakeEngine._last_instance is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_type"] == "archive"
    assert payload["archive_name"] == "docs.zip"
    assert payload["planned_count"] == 2
    assert [item["member_path"] for item in payload["items"]] == [
        "docs/guide.md",
        "docs/reference.md",
    ]
    assert [item["manifest_status"] for item in payload["items"]] == [
        "unchanged",
        "missing",
    ]
    assert str(tmp_path) not in repr(payload)
    assert "assets/logo.png" not in repr(payload)


def test_ingest_archive_json_sets_runtime_source_type_to_archive(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    monkeypatch.setattr(cli_module, "Engine", _FakeEngine)
    archive_path = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("docs/guide.md", "# Guide")

    exit_code = main(
        [
            "add",
            str(archive_path),
            "--namespace",
            "acme",
            "--collection",
            "help",
            "--qdrant-location",
            ":memory:",
            "--manifest-dir",
            str(tmp_path / "manifest"),
            "--metadata",
            "team=docs",
            "--force-reindex",
            "--max-concurrency",
            "2",
            "--archive-max-entries",
            "10",
            "--archive-max-entry-bytes",
            "100",
            "--archive-max-total-bytes",
            "200",
            "--trace-jsonl",
            str(tmp_path / "events.jsonl"),
            "--json",
        ]
    )

    assert exit_code == 0
    instance = cast(_FakeEngine, _FakeEngine._last_instance)
    assert instance.ensure_ready_called is True
    assert instance.closed is True
    assert instance.event_sink is not None
    assert instance.config.ingest.source_type == "archive"
    assert instance.config.ingest.manifest_directory == tmp_path / "manifest"
    assert len(instance.ingest_archive_calls) == 1
    call = instance.ingest_archive_calls[0]
    assert call["archive_path"] == str(archive_path)
    assert call["namespace"] == "acme"
    assert call["collection"] == "help"
    assert call["metadata"] == {"team": "docs"}
    assert call["force_reindex"] is True
    assert call["max_concurrency"] == 2
    assert call["manifest_dir"] == tmp_path / "manifest"
    assert call["archive_limits"].max_entries == 10
    assert call["archive_limits"].max_entry_bytes == 100
    assert call["archive_limits"].max_total_bytes == 200
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records == [
        {
            "chunk_count": 2,
            "document_id": "doc-guide",
            "filename": "guide.md",
            "ingest_state": "created",
            "manifest_reason": "source_not_in_manifest",
            "manifest_status": "missing",
            "ok": True,
            "path": "docs.zip!/docs/guide.md",
            "replaced_existing": False,
        }
    ]


def test_ingest_archive_text_output_prefers_member_path(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    monkeypatch.setattr(cli_module, "Engine", _FakeEngine)
    archive_path = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("docs/guide.md", "# Guide")

    exit_code = main(
        [
            "add",
            str(archive_path),
            "--namespace",
            "acme",
            "--collection",
            "help",
            "--qdrant-location",
            ":memory:",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "created: docs/guide.md -> doc-guide (2 chunks)" in output
    assert "created: guide.md -> doc-guide (2 chunks)" not in output


def test_ingest_archive_lazy_provider_bootstrap_error_is_actionable(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    class _ProviderFailingCore:
        closed = False

        async def ensure_ready(self) -> None:
            return None

        async def add_archive(self, *_args: Any, **_kwargs: Any) -> None:
            raise _FakeOpenAIError("raw api_key client option OPENAI_API_KEY")

        async def close(self) -> None:
            type(self).closed = True

    monkeypatch.setattr(cli_module, "Engine", lambda *_args, **_kwargs: _ProviderFailingCore())
    archive_path = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("docs/guide.md", "# Guide")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "add",
                str(archive_path),
                "--namespace",
                "acme",
                "--collection",
                "help",
                "--qdrant-location",
                ":memory:",
            ]
        )

    assert exc_info.value.code == 2
    assert _ProviderFailingCore.closed is True
    error = capsys.readouterr().err
    assert "provider failed during ingest" in error
    assert "provider setup failed before ingest" not in error
    assert "provider=openai" in error
    assert "raw api_key client option" not in error
    assert "OPENAI_API_KEY" not in error
    assert "Traceback" not in error


def _manifest_entry(
    *,
    document_key: str,
    content_sha256: str,
) -> CollectionManifestEntry:
    return CollectionManifestEntry(
        document_id="doc-existing",
        namespace="acme",
        collection="help",
        document_key=document_key,
        content_sha256=content_sha256,
        filename="guide.md",
        mime_type="text/markdown",
        chunk_count=1,
        parser="local:markdown",
        needs_ocr=False,
        metadata={"source_type": "archive"},
    )
