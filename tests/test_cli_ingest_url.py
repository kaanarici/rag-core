from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag_core.cli import main
from rag_core.fetch_security import validate_fetch_url
from tests.support.cli_remote import (
    FakeOpenAIError,
    FakeRemoteRAGCore,
    install_fake_remote_core,
    remote_url_key,
    require_fake_remote_core,
)


def test_ingest_url_json_sets_runtime_source_type_to_url(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    install_fake_remote_core(monkeypatch)
    raw_url = "https://example.com/docs/guide?private=alpha"
    exit_code = main(
        [
            "ingest",
            raw_url,
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--document-id",
            "doc-explicit",
            "--qdrant-location",
            ":memory:",
            "--manifest-dir",
            str(tmp_path / "manifest"),
            "--metadata",
            "title=Remote Guide",
            "--force-reindex",
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
    [call] = instance.ingest_url_calls
    assert call["url"] == raw_url
    assert call["namespace"] == "acme"
    assert call["corpus_id"] == "help"
    assert call["document_id"] == "doc-explicit"
    assert call["metadata"] == {"title": "Remote Guide"}
    assert call["force_reindex"] is True
    assert call.get("fetch_client") is None
    assert call["fetch_policy"].allow_private_addresses is False
    assert call["fetch_limits"].max_bytes == 4096
    assert call["fetch_limits"].timeout_seconds == 3.5
    assert call["fetch_limits"].max_redirects == 2
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["document_key"] == remote_url_key(validate_fetch_url(raw_url))
    assert payload["metadata"]["source_url"] == "https://example.com/docs/guide?redacted"
    assert "private=alpha" not in output


def test_ingest_url_text_output_uses_redacted_source_url(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    install_fake_remote_core(monkeypatch)
    exit_code = main(
        [
            "ingest",
            "https://example.com/docs/guide?private=alpha",
            "--namespace",
            "acme",
            "--corpus-id",
            "help",
            "--qdrant-location",
            ":memory:",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "https://example.com/docs/guide?redacted" in output
    assert "private=alpha" not in output


def test_ingest_url_rejects_invalid_url_before_runtime_setup(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    install_fake_remote_core(monkeypatch)
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "ingest",
                "file:///tmp/guide.txt",
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--qdrant-location",
                ":memory:",
            ]
        )

    assert exc_info.value.code == 2
    assert FakeRemoteRAGCore._last_instance is None
    error = capsys.readouterr().err
    assert "unsupported fetch URL scheme" in error
    assert "Traceback" not in error


def test_ingest_url_provider_bootstrap_error_is_actionable(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    class _ProviderFailingCore:
        closed = False

        async def ensure_ready(self) -> None:
            raise FakeOpenAIError("raw api_key client option OPENAI_API_KEY")

        async def close(self) -> None:
            type(self).closed = True

    monkeypatch.setattr(cli_module, "RAGCore", lambda *_args, **_kwargs: _ProviderFailingCore())

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "ingest",
                "https://example.com/docs",
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--qdrant-location",
                ":memory:",
            ]
        )

    assert exc_info.value.code == 2
    assert _ProviderFailingCore.closed is True
    error = capsys.readouterr().err
    assert "provider setup failed before ingest" in error
    assert "provider=openai" in error
    assert "raw api_key client option" not in error
    assert "OPENAI_API_KEY" in error
    assert "Traceback" not in error


def test_ingest_url_lazy_provider_bootstrap_error_is_actionable(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    class _ProviderFailingCore:
        closed = False

        async def ensure_ready(self) -> None:
            return None

        async def ingest_url(self, *_args: Any, **_kwargs: Any) -> None:
            raise FakeOpenAIError("raw api_key client option OPENAI_API_KEY")

        async def close(self) -> None:
            type(self).closed = True

    monkeypatch.setattr(cli_module, "RAGCore", lambda *_args, **_kwargs: _ProviderFailingCore())

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "ingest",
                "https://example.com/docs",
                "--namespace",
                "acme",
                "--corpus-id",
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


def test_ingest_url_runtime_error_is_sanitized(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    from rag_core import cli as cli_module

    class _RuntimeFailingCore:
        closed = False

        async def ensure_ready(self) -> None:
            return None

        async def ingest_url(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("fetch failed: https://example.com/docs?token=secret")

        async def close(self) -> None:
            type(self).closed = True

    monkeypatch.setattr(cli_module, "RAGCore", lambda *_args, **_kwargs: _RuntimeFailingCore())

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "ingest",
                "https://example.com/docs?token=secret",
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--qdrant-location",
                ":memory:",
            ]
        )

    assert exc_info.value.code == 2
    assert _RuntimeFailingCore.closed is True
    error = capsys.readouterr().err
    assert "remote ingest failed" in error
    assert "token=secret" not in error
    assert "Traceback" not in error


def test_ingest_url_private_address_requires_explicit_fetch_opt_in(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    install_fake_remote_core(monkeypatch)
    private_url = "http://127.0.0.1:8000/docs?private=alpha"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "ingest",
                private_url,
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--qdrant-location",
                ":memory:",
            ]
        )

    assert exc_info.value.code == 2
    assert FakeRemoteRAGCore._last_instance is None
    error = capsys.readouterr().err
    assert "HTTP requires explicit opt-in" in error
    assert "private=alpha" not in error

    exit_code = main(
        [
            "ingest",
            private_url,
            "--namespace",
            "acme",
            "--corpus-id",
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


def test_ingest_url_rejects_bad_fetch_limits_before_runtime_setup(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    install_fake_remote_core(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "ingest",
                "https://example.com/docs",
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--qdrant-location",
                ":memory:",
                "--fetch-max-bytes",
                "0",
            ]
        )

    assert exc_info.value.code == 2
    assert FakeRemoteRAGCore._last_instance is None
    error = capsys.readouterr().err
    assert "max_bytes" in error
    assert "Traceback" not in error


def test_ingest_url_rejects_non_finite_fetch_timeout_before_runtime_setup(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    install_fake_remote_core(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "ingest",
                "https://example.com/docs",
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--qdrant-location",
                ":memory:",
                "--fetch-timeout-seconds",
                "nan",
            ]
        )

    assert exc_info.value.code == 2
    assert FakeRemoteRAGCore._last_instance is None
    error = capsys.readouterr().err
    assert "timeout_seconds" in error
    assert "Traceback" not in error


def test_ingest_url_rejects_malformed_fetch_env_before_runtime_setup(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    install_fake_remote_core(monkeypatch)
    monkeypatch.setenv("RAG_CORE_FETCH_MAX_BYTES", "not-an-int")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "ingest",
                "https://example.com/docs",
                "--namespace",
                "acme",
                "--corpus-id",
                "help",
                "--qdrant-location",
                ":memory:",
                "--manifest-dir",
                str(tmp_path / "manifest"),
            ]
        )

    assert exc_info.value.code == 2
    assert FakeRemoteRAGCore._last_instance is None
    error = capsys.readouterr().err
    assert "RAG_CORE_FETCH_MAX_BYTES must be an integer" in error
    assert "Traceback" not in error
