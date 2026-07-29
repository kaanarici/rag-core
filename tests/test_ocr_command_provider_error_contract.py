from __future__ import annotations

import asyncio
import json
import subprocess
from types import SimpleNamespace

import pytest

from rag_core.documents import ocr as ocr_module
from rag_core.documents import ocr_provider_runtime
from rag_core.documents.ocr_command_runtime import run_command_ocr

SECRET = "sk-test-secret"
PRIVATE_STDOUT = f"private stdout with api key {SECRET}"
PRIVATE_STDERR = f"private stderr with api key {SECRET}"
PRIVATE_COMMAND = f"/tmp/private/ocr-{SECRET}"


def _request() -> ocr_module.OcrRequest:
    return ocr_module.OcrRequest(
        file_bytes=b"%PDF-1.7",
        filename="document.pdf",
        mime_type="application/pdf",
    )


def _run_provider(command: list[str] | None = None) -> ocr_module.OcrResult:
    provider = ocr_module.CommandOcrProvider(
        command=command or ["ocr-command"],
        provider_name="test-ocr",
        model_name=None,
        supports_page_selection=True,
        timeout_seconds=1.25,
        extra_env={},
    )
    return asyncio.run(provider.extract_markdown(_request()))


def _assert_private_data_absent(message: str) -> None:
    assert PRIVATE_STDOUT not in message
    assert PRIVATE_STDERR not in message
    assert PRIVATE_COMMAND not in message
    assert SECRET not in message
    assert "Traceback" not in message


def test_command_provider_nonzero_exit_error_omits_process_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def completed_with_private_output(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=13,
            stdout=PRIVATE_STDOUT,
            stderr=PRIVATE_STDERR,
        )

    monkeypatch.setattr(ocr_provider_runtime.subprocess, "run", completed_with_private_output)

    with pytest.raises(RuntimeError) as exc_info:
        _run_provider(command=[PRIVATE_COMMAND])

    message = str(exc_info.value)
    assert message == "OCR provider test-ocr failed with code 13"
    _assert_private_data_absent(message)


def test_command_provider_timeout_error_omits_command_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout_with_private_output(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(
            cmd=[PRIVATE_COMMAND],
            timeout=1.25,
            output=PRIVATE_STDOUT,
            stderr=PRIVATE_STDERR,
        )

    monkeypatch.setattr(ocr_provider_runtime.subprocess, "run", timeout_with_private_output)

    with pytest.raises(RuntimeError) as exc_info:
        _run_provider(command=[PRIVATE_COMMAND])

    message = str(exc_info.value)
    assert message == "OCR provider test-ocr timed out after 1.25s"
    assert exc_info.value.__cause__ is None
    _assert_private_data_absent(message)


def test_command_provider_start_error_omits_os_error_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_private_os_error(*args: object, **kwargs: object) -> None:
        raise OSError(f"cannot start {PRIVATE_COMMAND}")

    monkeypatch.setattr(ocr_provider_runtime.subprocess, "run", raise_private_os_error)

    with pytest.raises(RuntimeError) as exc_info:
        _run_provider(command=[PRIVATE_COMMAND])

    message = str(exc_info.value)
    assert message == "OCR provider test-ocr failed to start (error_type=OSError)"
    assert exc_info.value.__cause__ is None
    _assert_private_data_absent(message)


def test_command_provider_invalid_markdown_error_omits_payload_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def completed_with_private_payload(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout='{"markdown": {"secret": "sk-test-secret"}}',
            stderr="",
        )

    monkeypatch.setattr(ocr_provider_runtime.subprocess, "run", completed_with_private_payload)

    with pytest.raises(RuntimeError) as exc_info:
        _run_provider()

    message = str(exc_info.value)
    assert message == "OCR provider test-ocr returned invalid markdown payload"
    _assert_private_data_absent(message)


def test_command_provider_subprocess_env_is_allowlisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/tmp/home")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example.test:8443")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.setenv("SSL_CERT_FILE", "/tmp/custom-ca.pem")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/tmp/requests-ca.pem")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-test-key")
    monkeypatch.setenv("UNRELATED_SECRET", "should-not-leak")

    captured_env: dict[str, str] = {}

    def capture_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        del command
        env = kwargs.get("env")
        assert isinstance(env, dict)
        captured_env.update(env)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"markdown": "# OCR"}),
            stderr="",
        )

    run_command_ocr(
        file_bytes=b"%PDF-1.7",
        filename="document.pdf",
        mime_type="application/pdf",
        page_indices=[],
        existing_markdown="",
        metadata={},
        command=["ocr-command"],
        provider_name="mistral",
        model_name="mistral-ocr-latest",
        supports_page_selection=True,
        timeout_seconds=1.0,
        extra_env={"OCR_PROVIDER_REGION": "us"},
        run_command=capture_run,
    )

    assert captured_env["PATH"] == "/usr/bin"
    assert captured_env["HOME"] == "/tmp/home"
    assert captured_env["HTTPS_PROXY"] == "https://proxy.example.test:8443"
    assert captured_env["NO_PROXY"] == "localhost,127.0.0.1"
    assert captured_env["SSL_CERT_FILE"] == "/tmp/custom-ca.pem"
    assert captured_env["REQUESTS_CA_BUNDLE"] == "/tmp/requests-ca.pem"
    assert captured_env["MISTRAL_API_KEY"] == "mistral-test-key"
    assert captured_env["OCR_PROVIDER_REGION"] == "us"
    assert "UNRELATED_SECRET" not in captured_env
