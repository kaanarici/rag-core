"""Anthropic contextualizer runtime helpers."""

from __future__ import annotations

import importlib
import html
from typing import Any

from .contextualizer import ChunkContextRequest

_ANTHROPIC_CONTEXT_PROMPT_VERSION = "context-v1"
_ANTHROPIC_PROMPT = (
    "Please give a short succinct context to situate this chunk within the "
    "overall document for the purposes of improving search retrieval of the "
    "chunk. Answer only with the succinct context and nothing else."
)
_DOCUMENT_CHAR_BUDGET = 200_000


def create_anthropic_client(*, api_key: str | None) -> Any:
    try:
        anthropic_module = importlib.import_module("anthropic")
    except ImportError as exc:
        raise ImportError(
            "anthropic package is required for AnthropicChunkContextualizer. "
            "Install it with: pip install anthropic"
        ) from exc
    async_client_class = getattr(anthropic_module, "AsyncAnthropic", None)
    if async_client_class is None:
        raise ImportError(
            "anthropic package with AsyncAnthropic is required for "
            "AnthropicChunkContextualizer."
        )
    if api_key is not None:
        return async_client_class(api_key=api_key)
    return async_client_class()


def build_context_messages(request: ChunkContextRequest) -> list[dict[str, object]] | None:
    document = (request.document_markdown or "").strip()
    if not document:
        return None
    if len(document) > _DOCUMENT_CHAR_BUDGET:
        document = document[:_DOCUMENT_CHAR_BUDGET]
    filename = html.escape(request.document_filename, quote=True)
    document = html.escape(document, quote=False)
    chunk_text = html.escape(request.chunk_text, quote=False)
    document_block = {
        "type": "text",
        "text": (
            f"<document filename=\"{filename}\">\n"
            f"{document}\n</document>"
        ),
        "cache_control": {"type": "ephemeral"},
    }
    chunk_block = {
        "type": "text",
        "text": (
            f"<chunk index=\"{request.chunk_index}\" "
            f"total=\"{request.total_chunks}\">\n"
            f"{chunk_text}\n</chunk>\n\n{_ANTHROPIC_PROMPT}"
        ),
    }
    return [{"role": "user", "content": [document_block, chunk_block]}]


def extract_anthropic_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)
