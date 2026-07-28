"""Provider-level helpers for command-backed OCR adapters."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from rag_core.documents.ocr_command_runtime import (
    CommandOcrOutput,
    normalize_ocr_page_indices,
    run_command_ocr,
)


def build_mistral_ocr_command(
    *,
    model_name: str,
    python_executable: str | None,
) -> list[str]:
    return _build_ocr_module_command(
        module="rag_core.documents.ocr_commands.mistral",
        model_name=model_name,
        python_executable=python_executable,
    )


def build_gemini_ocr_command(
    *,
    model_name: str,
    python_executable: str | None,
) -> list[str]:
    return _build_ocr_module_command(
        module="rag_core.documents.ocr_commands.gemini",
        model_name=model_name,
        python_executable=python_executable,
    )


def run_command_provider_ocr(
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
) -> CommandOcrOutput:
    return run_command_ocr(
        file_bytes=file_bytes,
        filename=filename,
        mime_type=mime_type,
        page_indices=normalize_ocr_page_indices(page_indices),
        existing_markdown=existing_markdown,
        metadata=metadata,
        command=command,
        provider_name=provider_name,
        model_name=model_name,
        supports_page_selection=supports_page_selection,
        timeout_seconds=timeout_seconds,
        extra_env=extra_env,
        run_command=subprocess.run,
    )


def _build_ocr_module_command(
    *,
    module: str,
    model_name: str,
    python_executable: str | None,
) -> list[str]:
    return [
        python_executable or sys.executable,
        "-m",
        module,
        "--model",
        model_name,
    ]
