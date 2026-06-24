from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag_core.cli import main
from rag_core.core_models import CollectionManifestEntry, IngestedDocument
from rag_core.cli.ingest_output import emit_ingested_document
from rag_core.fetch_security import validate_fetch_url
from rag_core.manifest.persistence import write_entry
from tests.support.cli_remote import (
    FakeOpenAIError,
    FakeRemoteEngine,
    install_fake_remote_core,
    remote_url_key,
    require_fake_remote_core,
)


def _manifest_entry(
    *,
    document_key: str,
    document_id: str = "doc-existing",
    content_sha256: str = "hash-existing",
) -> CollectionManifestEntry:
    return CollectionManifestEntry(
        document_id=document_id,
        namespace="acme",
        collection="help",
        document_key=document_key,
        content_sha256=content_sha256,
        filename="remote.txt",
        mime_type="text/plain",
        chunk_count=1,
        parser="remote:text",
        needs_ocr=False,
        metadata={"source_type": "url"},
    )


def test_ingest_urls_json_sets_runtime_source_type_to_url_and_redacted_records(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    install_fake_remote_core(monkeypatch)
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "\n".join(
            [
                "https://example.com/docs/guide?private=alpha",
                "https://example.com/docs/reference",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "add", "--url-list",
            str(url_file),
            "--namespace",
            "acme",
            "--collection",
            "help",
            "--qdrant-location",
            ":memory:",
            "--manifest-dir",
            str(tmp_path / "manifest"),
            "--metadata",
            "title=Docs",
            "--force-reindex",
            "--max-concurrency",
            "2",
            "--fetch-max-bytes",
            "4096",
            "--fetch-timeout-seconds",
            "3.5",
            "--fetch-max-redirects",
            "2",
            "--json",
        ]
    )

    assert exit_code == 0
    instance = require_fake_remote_core()
    assert instance.ensure_ready_called is True
    assert instance.closed is True
    assert instance.config.ingest.source_type == "url"
    assert instance.config.ingest.manifest_directory == tmp_path / "manifest"
    assert [
        {
            key: value
            for key, value in call.items()
            if key not in {"fetch_client", "fetch_policy", "fetch_limits"}
        }
        for call in instance.ingest_url_calls
    ] == [
        {
            "url": "https://example.com/docs/guide?private=alpha",
            "namespace": "acme",
            "collection": "help",
            "metadata": {"title": "Docs"},
            "force_reindex": True,
        },
        {
            "url": "https://example.com/docs/reference",
            "namespace": "acme",
            "collection": "help",
            "metadata": {"title": "Docs"},
            "force_reindex": True,
        },
    ]
    assert all(call.get("fetch_client") is None for call in instance.ingest_url_calls)
    assert all(
        call["fetch_policy"].allow_private_addresses is False
        for call in instance.ingest_url_calls
    )
    assert all(
        call["fetch_limits"].max_bytes == 4096 for call in instance.ingest_url_calls
    )
    assert all(
        call["fetch_limits"].timeout_seconds == 3.5
        for call in instance.ingest_url_calls
    )
    assert all(
        call["fetch_limits"].max_redirects == 2 for call in instance.ingest_url_calls
    )
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["ok"] for record in records] == [True, True]
    assert records[0]["source_url"] == "https://example.com/docs/guide?redacted"
    assert "private=alpha" not in repr(records)


def test_ingest_urls_json_returns_one_for_mixed_results_and_redacts_failure(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    install_fake_remote_core(monkeypatch)
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "\n".join(
            [
                "https://example.com/docs/guide?private=alpha",
                "https://example.com/docs/fail?private=beta",
                "https://example.com/docs/reference",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "add", "--url-list",
            str(url_file),
            "--namespace",
            "acme",
            "--collection",
            "help",
            "--qdrant-location",
            ":memory:",
            "--json",
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.out.splitlines()]
    assert [record["ok"] for record in records] == [True, False, True]
    assert records[1]["requested_url"] == "https://example.com/docs/fail?redacted"
    assert "private=alpha" not in repr(records)
    assert "private=beta" not in repr(records)
    assert "Traceback" not in captured.err


def test_ingest_urls_surfaces_provider_bootstrap_as_batch_setup_error(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    async def fail_add_url(
        self: FakeRemoteEngine,
        url: str,
        **kwargs: Any,
    ) -> IngestedDocument:
        self.ingest_url_calls.append({"url": url, **kwargs})
        raise FakeOpenAIError("raw api_key client option OPENAI_API_KEY private-token")

    install_fake_remote_core(monkeypatch)
    monkeypatch.setattr(FakeRemoteEngine, "add_url", fail_add_url)
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.com/docs/guide\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "add", "--url-list",
                str(url_file),
                "--namespace",
                "acme",
                "--collection",
                "help",
                "--qdrant-location",
                ":memory:",
                "--json",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert captured.out == ""
    assert "provider failed during ingest" in captured.err
    assert "provider setup failed before ingest" not in captured.err
    assert "provider=openai" in captured.err
    assert "OPENAI_API_KEY" not in captured.err
    assert "raw api_key" not in captured.err
    assert "private-token" not in captured.err
    assert "failed:" not in captured.out
    instance = require_fake_remote_core()
    assert instance.closed is True


def test_ingest_urls_plan_json_prints_redacted_plan_without_runtime_setup(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    install_fake_remote_core(monkeypatch)
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "\n".join(
            [
                "# discovered docs",
                "https://example.com/docs/guide?private=alpha",
                "https://example.com/docs/reference",
            ]
        ),
        encoding="utf-8",
    )
    manifest_dir = tmp_path / "manifest"
    write_entry(
        manifest_dir,
        _manifest_entry(
            document_key=remote_url_key(
                validate_fetch_url("https://example.com/docs/reference")
            )
        ),
    )

    exit_code = main(
        [
            "add", "--url-list",
            str(url_file),
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
    assert FakeRemoteEngine._last_instance is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_type"] == "url"
    assert payload["planned_count"] == 2
    assert payload["urls"][0]["url"] == "https://example.com/docs/guide?redacted"
    assert payload["urls"][0]["source_line"] == 2
    assert "query_sha256" not in payload["urls"][0]
    assert payload["urls"][0]["manifest_status"] == "unknown_until_fetch"
    assert payload["urls"][0]["manifest_reason"] == "canonical_url_unknown_until_fetch"
    assert payload["urls"][1]["manifest_status"] == "unchanged"
    assert payload["urls"][1]["manifest_reason"] == "present_without_hash_check"
    assert payload["reconciliation"]["summary"]["missing_count"] == 1
    assert "private=alpha" not in repr(payload)


def test_ingest_urls_rejects_invalid_url_before_runtime_setup(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    install_fake_remote_core(monkeypatch)
    url_file = tmp_path / "urls.txt"
    url_file.write_text("file:///tmp/private.md\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "add", "--url-list",
                str(url_file),
                "--namespace",
                "acme",
                "--collection",
                "help",
                "--qdrant-location",
                ":memory:",
            ]
        )

    assert exc_info.value.code == 2
    assert FakeRemoteEngine._last_instance is None
    error = capsys.readouterr().err
    assert "URL list line 1" in error
    assert "unsupported fetch URL scheme" in error
    assert "Traceback" not in error


def test_ingest_urls_rejects_duplicate_url_keys_before_runtime_setup(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    install_fake_remote_core(monkeypatch)
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "https://example.com/docs\nhttps://example.com/docs\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "add", "--url-list",
                str(url_file),
                "--namespace",
                "acme",
                "--collection",
                "help",
                "--qdrant-location",
                ":memory:",
            ]
        )

    assert exc_info.value.code == 2
    assert FakeRemoteEngine._last_instance is None
    error = capsys.readouterr().err
    assert "same document key" in error
    assert "lines 1 and 2" in error
    assert "Traceback" not in error


def test_ingest_urls_private_address_requires_explicit_fetch_opt_in(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    install_fake_remote_core(monkeypatch)
    url_file = tmp_path / "urls.txt"
    private_url = "http://127.0.0.1:8000/docs?private=alpha"
    url_file.write_text(private_url, encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "add", "--url-list",
                str(url_file),
                "--namespace",
                "acme",
                "--collection",
                "help",
                "--qdrant-location",
                ":memory:",
            ]
        )

    assert exc_info.value.code == 2
    assert FakeRemoteEngine._last_instance is None
    error = capsys.readouterr().err
    assert "URL list line 1" in error
    assert "HTTP requires explicit opt-in" in error
    assert "private=alpha" not in error

    exit_code = main(
        [
            "add", "--url-list",
            str(url_file),
            "--namespace",
            "acme",
            "--collection",
            "help",
            "--qdrant-location",
            ":memory:",
            "--fetch-allow-http",
            "--fetch-allow-private-addresses",
        ]
    )

    assert exit_code == 0
    instance = require_fake_remote_core()
    assert instance.ingest_url_calls[0]["url"] == private_url
    assert instance.ingest_url_calls[0]["fetch_policy"].allowed_schemes == (
        "https",
        "http",
    )
    assert instance.ingest_url_calls[0]["fetch_policy"].allow_private_addresses is True


def test_emit_ingested_document_json_sanitizes_url_bearing_metadata(
    capsys: Any,
) -> None:
    document = IngestedDocument(
        document_id="doc-1",
        collection="help",
        namespace="acme",
        chunk_count=1,
        filename="final.txt",
        mime_type="text/plain",
        document_key="url:https://example.com/docs/final?redacted",
        ingest_state="created",
        processing_version="pipeline-v-test",
        metadata={
            "source_type": "url",
            "source_url": "https://user:secret@example.com/docs/final?token=alpha#frag",
            "source_requested_url": "https://example.com/start?token=beta#frag",
            "source_canonical_url": "not a url?token=gamma",
            "title": "Guide",
        },
    )

    emit_ingested_document(document, as_json=True)

    payload = json.loads(capsys.readouterr().out)
    metadata = payload["metadata"]
    assert payload["pipeline_version"] == "pipeline-v-test"
    assert "processing_version" not in payload
    assert metadata["source_url"] == "https://example.com/docs/final?redacted"
    assert metadata["source_requested_url"] == "https://example.com/start?redacted"
    assert metadata["source_canonical_url"] == "<redacted-url>"
    assert metadata["title"] == "Guide"
    assert "secret" not in repr(payload)
    assert "token=alpha" not in repr(payload)
    assert "token=beta" not in repr(payload)
    assert "token=gamma" not in repr(payload)


def test_ingest_urls_rejects_bad_fetch_limits_before_runtime_setup(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    install_fake_remote_core(monkeypatch)
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.com/docs\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "add", "--url-list",
                str(url_file),
                "--namespace",
                "acme",
                "--collection",
                "help",
                "--qdrant-location",
                ":memory:",
                "--fetch-max-redirects",
                "-1",
            ]
        )

    assert exc_info.value.code == 2
    assert FakeRemoteEngine._last_instance is None
    error = capsys.readouterr().err
    assert "max_redirects" in error
    assert "Traceback" not in error
