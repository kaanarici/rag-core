"""Subprocess runtime for Firecrawl's pdf-inspector CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Final

from rag_core.config.env_access import get_env_int, get_env_stripped

logger = logging.getLogger("rag_core.documents.pdf_inspector")

_DEFAULT_TIMEOUT_MS: Final[int] = 8_000
_DEFAULT_MAX_BYTES: Final[int] = 50 * 1024 * 1024
_WARNED_BINARY_KEYS: set[str] = set()
_RUNTIME_ENV_KEYS: Final[tuple[str, ...]] = (
    "PATH",
    "HOME",
    "SYSTEMROOT",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "NODE_OPTIONS",
)
_TRANSPORT_ENV_KEYS: Final[tuple[str, ...]] = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)


def pdf_inspector_enabled() -> bool:
    raw_mode = get_env_stripped("PDF_INSPECTOR_MODE", "")
    if raw_mode:
        return raw_mode.lower() not in {"disable", "disabled", "off", "false", "0"}
    return True


def describe_pdf_inspector_runtime() -> dict[str, object]:
    resolved_detect = _resolve_binary_path("detect-pdf")
    resolved_extract = _resolve_binary_path("pdf2md")
    configured_path = get_env_stripped("PDF_INSPECTOR_BINARY_PATH", "")
    return {
        "enabled": pdf_inspector_enabled(),
        "binary_path": "configured" if configured_path else None,
        "binary_path_configured": bool(configured_path),
        "detect_pdf_available": resolved_detect is not None,
        "pdf2md_available": resolved_extract is not None,
        "timeout_ms": max(1, get_env_int("PDF_INSPECTOR_TIMEOUT_MS", _DEFAULT_TIMEOUT_MS)),
        "max_bytes": max(1, get_env_int("PDF_INSPECTOR_MAX_BYTES", _DEFAULT_MAX_BYTES)),
    }


def run_pdf_inspector(command: list[str], file_bytes: bytes) -> dict[str, object] | None:
    if not pdf_inspector_enabled():
        return None

    max_bytes = max(1, get_env_int("PDF_INSPECTOR_MAX_BYTES", _DEFAULT_MAX_BYTES))
    if len(file_bytes) > max_bytes:
        logger.warning(
            "Skipping pdf-inspector for oversized PDF (%d bytes > %d byte limit)",
            len(file_bytes),
            max_bytes,
        )
        return None

    binary_name = command[0]
    binary_path = _resolve_binary_path(binary_name)
    if binary_path is None:
        _warn_missing_binary(binary_name)
        return None

    timeout_ms = max(1, get_env_int("PDF_INSPECTOR_TIMEOUT_MS", _DEFAULT_TIMEOUT_MS))
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_file.flush()
            temp_path = temp_file.name

        completed = subprocess.run(
            [binary_path, temp_path, *command[1:]],
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_ms / 1000,
            env=_subprocess_env(),
        )
    except FileNotFoundError:
        _warn_missing_binary(binary_name)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("pdf-inspector %s timed out after %dms", binary_name, timeout_ms)
        return None
    except OSError as exc:
        logger.warning(
            "pdf-inspector %s failed to start: %s",
            binary_name,
            _exception_type(exc),
        )
        return None
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)

    if completed.returncode != 0:
        logger.warning(
            "pdf-inspector %s exited with code %d",
            binary_name,
            completed.returncode,
        )
        return None

    stdout = completed.stdout.strip()
    if not stdout:
        logger.warning("pdf-inspector %s returned empty output", binary_name)
        return None

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        logger.warning(
            "pdf-inspector %s returned invalid JSON: %s",
            binary_name,
            _exception_type(exc),
        )
        return None

    if not isinstance(parsed, dict):
        logger.warning("pdf-inspector %s returned non-object JSON", binary_name)
        return None
    return parsed


def _resolve_binary_path(binary_name: str) -> str | None:
    configured_path = get_env_stripped("PDF_INSPECTOR_BINARY_PATH", "")
    if configured_path:
        configured = Path(configured_path)
        if configured.is_dir():
            candidate = configured / binary_name
        elif configured.name == binary_name:
            candidate = configured
        else:
            candidate = configured.parent / binary_name

        if candidate.is_file():
            return str(candidate)
        return None

    return shutil.which(binary_name)


def _warn_missing_binary(binary_name: str) -> None:
    configured_path = get_env_stripped("PDF_INSPECTOR_BINARY_PATH", "")
    warning_key = f"{binary_name}:{configured_path}"
    if warning_key in _WARNED_BINARY_KEYS:
        return
    _WARNED_BINARY_KEYS.add(warning_key)
    log_fn = logger.warning if configured_path else logger.info
    log_fn(
        "pdf-inspector binary %s was not found (configured_path=%s)",
        binary_name,
        bool(configured_path),
    )


def _exception_type(exc: Exception) -> str:
    return type(exc).__name__


def _subprocess_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in (*_RUNTIME_ENV_KEYS, *_TRANSPORT_ENV_KEYS):
        value = get_env_stripped(key, "")
        if value:
            env[key] = value
    return env
