from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import rag_core.cli as cli_module
from rag_core.cli import async_main
from rag_core.core_models import CorpusManifestEntry
from rag_core.manifest_persistence import read_entries, write_entry


def _entry(
    *,
    document_id: str = "doc-1",
    chunk_count: int = 3,
) -> CorpusManifestEntry:
    return CorpusManifestEntry(
        document_id=document_id,
        namespace="acme",
        corpus_id="help",
        document_key=f"{document_id}.md",
        content_sha256=f"sha-{chunk_count}",
        filename=f"{document_id}.md",
        mime_type="text/markdown",
        chunk_count=chunk_count,
        parser="local:text",
        needs_ocr=False,
        metadata={},
    )


def test_manifest_compact_cli_rewrites_manifest_and_emits_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_entry(tmp_path, _entry(document_id="doc-1", chunk_count=1))
    write_entry(tmp_path, _entry(document_id="doc-2", chunk_count=2))
    write_entry(tmp_path, _entry(document_id="doc-1", chunk_count=5))

    exit_code = asyncio.run(
        async_main(
            [
                "manifest", "--compact",
                "--manifest-dir",
                str(tmp_path),
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--json",
            ]
        )
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "after_entry_count": 2,
        "before_entry_count": 3,
        "changed": True,
        "corpus_id": "help",
        "manifest_dir": str(tmp_path),
        "namespace": "acme",
        "removed_entry_count": 1,
    }
    entries = {
        entry.document_id: entry
        for entry in read_entries(tmp_path, namespace="acme", corpus_id="help")
    }
    assert entries["doc-1"].chunk_count == 5
    assert entries["doc-2"].chunk_count == 2


def test_manifest_compact_cli_missing_manifest_is_human_readable_noop(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = asyncio.run(
        async_main(
            [
                "manifest", "--compact",
                "--manifest-dir",
                str(tmp_path),
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
            ]
        )
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Entries Before: 0" in output
    assert "Entries After: 0" in output
    assert "Entries Removed: 0" in output
    assert "Changed: False" in output
    assert not (tmp_path / "acme").exists()


def test_manifest_compact_cli_does_not_construct_runtime(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_runtime_construction(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("manifest --compact must not construct RAGCore")

    monkeypatch.setattr(cli_module, "RAGCore", fail_runtime_construction)

    exit_code = asyncio.run(
        async_main(
            [
                "manifest", "--compact",
                "--manifest-dir",
                str(tmp_path),
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
            ]
        )
    )

    assert exit_code == 0
    assert "Entries Before: 0" in capsys.readouterr().out


def test_manifest_compact_cli_rejects_bad_scope(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(
            async_main(
                [
                    "manifest", "--compact",
                    "--manifest-dir",
                    str(tmp_path),
                    "--namespace",
                    "../escape",
                    "--corpus-id",
                    "help",
                ]
            )
        )

    assert exc_info.value.code == 2
    assert "single non-empty path segment" in capsys.readouterr().err


def test_manifest_compact_cli_rejects_file_manifest_dir_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_file = tmp_path / "manifest.jsonl"
    manifest_file.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(
            async_main(
                [
                    "manifest", "--compact",
                    "--manifest-dir",
                    str(manifest_file),
                    "--namespace",
                    "acme",
                    "--corpus-id",
                    "help",
                ]
            )
        )

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "manifest directory must be a directory" in error
    assert "Traceback" not in error
