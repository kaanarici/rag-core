"""Runtime helpers for command-backed OCR providers."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any, BinaryIO

from rag_core.documents.exception_names import exception_type
from rag_core.documents.page_indices import normalize_page_indices
from rag_core.documents.subprocess_env import (
    PYTHON_SUBPROCESS_ENV_KEYS,
    allowlisted_subprocess_env,
)

_OCR_PROVIDER_ENV_KEYS: dict[str, tuple[str, ...]] = {}


@dataclass(frozen=True)
class CommandOcrOutput:
    markdown: str
    merge_mode: str
    provider_name: str
    model_name: str | None
    pages_processed: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


async def run_command_ocr(
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
            with tempfile.TemporaryFile() as output_file:
                stdout = await _run_ocr_process(
                    command=command,
                    payload=json.dumps(payload).encode(),
                    output_file=output_file,
                    provider_name=provider_name,
                    timeout_seconds=timeout_seconds,
                    extra_env=extra_env,
                )
        except OSError as exc:
            raise RuntimeError(
                f"OCR provider {provider_name} failed to start "
                f"(error_type={exception_type(exc)})"
            ) from None
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"OCR provider {provider_name} returned invalid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"OCR provider {provider_name} returned a non-object payload"
        )

    raw_markdown = parsed.get("markdown")
    if not isinstance(raw_markdown, str):
        raise RuntimeError(
            f"OCR provider {provider_name} returned invalid markdown payload"
        )

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


async def _run_ocr_process(
    *,
    command: list[str],
    payload: bytes,
    output_file: BinaryIO,
    provider_name: str,
    timeout_seconds: float,
    extra_env: dict[str, str],
) -> bytes:
    # File-backed output avoids inherited-pipe waits after the command exits.
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=output_file,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=os.name == "posix",
        env=_subprocess_env(provider_name=provider_name, extra_env=extra_env),
    )
    try:
        await asyncio.wait_for(process.communicate(payload), timeout=timeout_seconds)
    except BaseException as exc:
        failure = exc
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.returncode is None:
                process.kill()
        except ProcessLookupError:
            pass
        cleanup = asyncio.create_task(asyncio.wait_for(process.wait(), timeout=0.5))
        while True:
            try:
                await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError as cancelled:
                failure = cancelled
                continue
            except TimeoutError:
                break
        if isinstance(failure, TimeoutError):
            raise RuntimeError(
                f"OCR provider {provider_name} timed out after {timeout_seconds:g}s"
            ) from None
        raise failure

    if process.returncode != 0:
        raise RuntimeError(
            f"OCR provider {provider_name} failed with code {_safe_returncode(process.returncode)}"
        )
    output_file.seek(0)
    return output_file.read()


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
