from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

import rag_core.documents.pdf_inspector_runtime as pdf_inspector_runtime
from tests.support import assert_caplog_omits_private, assert_log_record_contains

LOGGER_NAME = "rag_core.documents.pdf_inspector"
PRIVATE_BINARY_PATH = "/tmp/private/pdf-inspector-sk-test-secret/bin"
PRIVATE_STDERR = "private stderr body with api key sk-test-secret"
PRIVATE_STDOUT = "private stdout body with api key sk-test-secret"


def _assert_private_runtime_data_absent(caplog: pytest.LogCaptureFixture) -> None:
    assert_caplog_omits_private(
        caplog,
        PRIVATE_STDERR,
        PRIVATE_STDOUT,
        PRIVATE_BINARY_PATH,
        "PDF_INSPECTOR_BINARY_PATH",
    )


def test_nonzero_exit_log_omits_process_output(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(pdf_inspector_runtime, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(pdf_inspector_runtime, "_resolve_binary_path", lambda _: "detect-pdf")

    def complete_with_private_output(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=13,
            stdout=PRIVATE_STDOUT,
            stderr=PRIVATE_STDERR,
        )

    monkeypatch.setattr(
        pdf_inspector_runtime.subprocess,
        "run",
        complete_with_private_output,
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = pdf_inspector_runtime.run_pdf_inspector(
            ["detect-pdf", "--analyze", "--json"],
            b"%PDF-1.7",
    )

    assert result is None
    assert_log_record_contains(
        caplog,
        "pdf-inspector detect-pdf exited with code 13",
        logger_name=LOGGER_NAME,
    )
    _assert_private_runtime_data_absent(caplog)


@pytest.mark.parametrize(
    ("configured_path", "expected_configured"),
    [
        (None, "False"),
        (PRIVATE_BINARY_PATH, "True"),
    ],
)
def test_missing_binary_log_omits_configured_path(
    configured_path: str | None,
    expected_configured: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    if configured_path is None:
        monkeypatch.delenv("PDF_INSPECTOR_BINARY_PATH", raising=False)
    else:
        monkeypatch.setenv("PDF_INSPECTOR_BINARY_PATH", configured_path)
    monkeypatch.setattr(pdf_inspector_runtime, "_WARNED_BINARY_KEYS", set[str]())
    monkeypatch.setattr(pdf_inspector_runtime, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(pdf_inspector_runtime, "_resolve_binary_path", lambda _: None)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = pdf_inspector_runtime.run_pdf_inspector(
            ["detect-pdf", "--analyze", "--json"],
            b"%PDF-1.7",
        )

    assert result is None
    assert_log_record_contains(
        caplog,
        "pdf-inspector binary detect-pdf was not found",
        f"configured_path={expected_configured}",
        logger_name=LOGGER_NAME,
    )
    _assert_private_runtime_data_absent(caplog)


def test_pdf_inspector_subprocess_env_is_allowlisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/tmp/home")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.test:8080")
    monkeypatch.setenv("SSL_CERT_DIR", "/tmp/certs")
    monkeypatch.setenv("CURL_CA_BUNDLE", "/tmp/curl-ca.pem")
    monkeypatch.setenv("UNRELATED_SECRET", "sk-test-secret")
    monkeypatch.setattr(pdf_inspector_runtime, "pdf_inspector_enabled", lambda: True)
    monkeypatch.setattr(pdf_inspector_runtime, "_resolve_binary_path", lambda _: "detect-pdf")

    captured_env: dict[str, str] = {}

    def capture_run(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        env = kwargs.get("env")
        assert isinstance(env, dict)
        captured_env.update(env)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"pdf_type": "text"}),
            stderr="",
        )

    monkeypatch.setattr(pdf_inspector_runtime.subprocess, "run", capture_run)

    result = pdf_inspector_runtime.run_pdf_inspector(
        ["detect-pdf", "--analyze", "--json"],
        b"%PDF-1.7",
    )

    assert result == {"pdf_type": "text"}
    assert captured_env["PATH"] == "/usr/bin"
    assert captured_env["HOME"] == "/tmp/home"
    assert captured_env["HTTP_PROXY"] == "http://proxy.example.test:8080"
    assert captured_env["SSL_CERT_DIR"] == "/tmp/certs"
    assert captured_env["CURL_CA_BUNDLE"] == "/tmp/curl-ca.pem"
    assert "UNRELATED_SECRET" not in captured_env
