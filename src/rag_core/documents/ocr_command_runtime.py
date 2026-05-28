"""Runtime helpers for command-backed OCR providers."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from rag_core.documents.exception_names import exception_type
from rag_core.documents.ocr_provider_names import (
    GEMINI_OCR_PROVIDER,
    MISTRAL_OCR_PROVIDER,
)
from rag_core.documents.page_indices import normalize_page_indices
from rag_core.documents.subprocess_env import (
    PYTHON_SUBPROCESS_ENV_KEYS,
    allowlisted_subprocess_env,
)
from rag_core.provider_api_keys import GEMINI_API_KEY_ENVS, MISTRAL_API_KEY_ENVS

_OCR_PROVIDER_ENV_KEYS: dict[str, tuple[str, ...]] = {
    MISTRAL_OCR_PROVIDER: MISTRAL_API_KEY_ENVS,
    GEMINI_OCR_PROVIDER: GEMINI_API_KEY_ENVS,
}


@dataclass(frozen=True)
class CommandOcrOutput:
    markdown: str
    merge_mode: str
    provider_name: str
    model_name: str | None
    pages_processed: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def run_command_ocr(
    *,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    page_indices: list[int],
    existing_markdown: str,
    metadata: dict[str, Any],
    command: list[str],
    provider_name: str,
    model_name: str | None,
    supports_page_selection: bool,
    timeout_seconds: float,
    extra_env: dict[str, str],
    run_command: Callable[..., Any] = subprocess.run,
) -> CommandOcrOutput:
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=_suffix_for_filename(filename),
            delete=False,
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_file.flush()
            temp_path = temp_file.name

        payload = {
            "file_path": temp_path,
            "filename": filename,
            "mime_type": mime_type,
            "page_indices": page_indices if supports_page_selection else [],
            "requested_page_indices": page_indices,
            "existing_markdown": existing_markdown,
            "metadata": metadata,
        }
        try:
            completed = run_command(
                command,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
                env=_subprocess_env(
                    provider_name=provider_name,
                    extra_env=extra_env,
                ),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"OCR provider {provider_name} timed out after {timeout_seconds:g}s"
            ) from None
        except OSError as exc:
            raise RuntimeError(
                f"OCR provider {provider_name} failed to start "
                f"(error_type={exception_type(exc)})"
            ) from None
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)

    if completed.returncode != 0:
        raise RuntimeError(
            f"OCR provider {provider_name} failed with code {_safe_returncode(completed.returncode)}"
        )

    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OCR provider {provider_name} returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"OCR provider {provider_name} returned a non-object payload")

    raw_markdown = parsed.get("markdown")
    if not isinstance(raw_markdown, str):
        raise RuntimeError(f"OCR provider {provider_name} returned invalid markdown payload")

    raw_metadata = parsed.get("metadata", {})
    return CommandOcrOutput(
        markdown=raw_markdown,
        merge_mode=_resolve_merge_mode(
            raw_mode=parsed.get("merge_mode"),
            supports_page_selection=supports_page_selection,
            requested_page_indices=page_indices,
        ),
        provider_name=str(parsed.get("provider_name") or provider_name),
        model_name=_optional_str(parsed.get("model_name")) or model_name,
        pages_processed=normalize_ocr_page_indices(
            parsed.get("pages_processed", page_indices)
        ),
        metadata=dict(raw_metadata) if isinstance(raw_metadata, dict) else {},
    )


def normalize_ocr_page_indices(raw_indices: object) -> list[int]:
    return normalize_page_indices(raw_indices, sort=True)


def _subprocess_env(
    *,
    provider_name: str,
    extra_env: dict[str, str],
) -> dict[str, str]:
    return allowlisted_subprocess_env(
        runtime_env_keys=PYTHON_SUBPROCESS_ENV_KEYS,
        provider_env_keys=_OCR_PROVIDER_ENV_KEYS.get(provider_name.lower(), ()),
        extra_env=extra_env,
    )


def _suffix_for_filename(filename: str) -> str:
    suffix = Path(filename).suffix.strip()
    return suffix if suffix else ".bin"


def _safe_returncode(value: object) -> int | str:
    if isinstance(value, bool) or not isinstance(value, int):
        return "unknown"
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _resolve_merge_mode(
    *,
    raw_mode: object,
    supports_page_selection: bool,
    requested_page_indices: list[int],
) -> str:
    if raw_mode == "replace":
        return "replace"
    if raw_mode == "append":
        return "append"
    if not supports_page_selection:
        return "replace"
    if requested_page_indices:
        return "append"
    return "replace"
