from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from rag_core.documents.ocr_command_runtime import (
    run_command_ocr as run_command_provider_ocr,
)

COMMAND_OCR_PROVIDER = "command"


@dataclass(frozen=True)
class OcrRequest:
    file_bytes: bytes
    filename: str
    mime_type: str
    page_indices: list[int] = field(default_factory=list)
    existing_markdown: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OcrResult:
    markdown: str
    merge_mode: str = "append"
    provider_name: str | None = None
    model_name: str | None = None
    pages_processed: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class OcrProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str | None: ...

    @property
    def supports_page_selection(self) -> bool: ...

    async def extract_markdown(self, request: OcrRequest) -> OcrResult: ...


class CommandOcrProvider:
    def __init__(
        self,
        *,
        command: list[str],
        provider_name: str = COMMAND_OCR_PROVIDER,
        model_name: str | None = None,
        supports_page_selection: bool = True,
        timeout_seconds: float = 120.0,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("CommandOcrProvider requires a non-empty command")
        self._command = list(command)
        self._provider_name = provider_name
        self._model_name = model_name
        self._supports_page_selection = supports_page_selection
        self._timeout_seconds = timeout_seconds
        self._extra_env = dict(extra_env or {})

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def supports_page_selection(self) -> bool:
        return self._supports_page_selection

    async def extract_markdown(self, request: OcrRequest) -> OcrResult:
        output = await run_command_provider_ocr(
            file_bytes=request.file_bytes,
            filename=request.filename,
            mime_type=request.mime_type,
            page_indices=request.page_indices,
            existing_markdown=request.existing_markdown,
            metadata=request.metadata,
            command=self._command,
            provider_name=self._provider_name,
            model_name=self._model_name,
            supports_page_selection=self._supports_page_selection,
            timeout_seconds=self._timeout_seconds,
            extra_env=self._extra_env,
        )
        return OcrResult(
            markdown=output.markdown,
            merge_mode=output.merge_mode,
            provider_name=output.provider_name,
            model_name=output.model_name,
            pages_processed=output.pages_processed,
            metadata=output.metadata,
        )
