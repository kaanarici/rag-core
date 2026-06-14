"""JSON converter that pretty-prints JSON in a markdown code fence."""

from __future__ import annotations

import asyncio
import json
import logging

from .base import BaseConverter, ConversionResult, safe_decode, score_text_quality
from .converter_keys import JSON_CONVERTER_KEY

logger = logging.getLogger(__name__)


class JsonConverter(BaseConverter):
    """Converts JSON files to pretty-printed markdown code fences."""

    format_name = JSON_CONVERTER_KEY

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert JSON to pretty-printed code fence."""

        def _convert() -> ConversionResult:
            try:
                text = safe_decode(file_bytes)
            except ValueError as exc:
                return ConversionResult(
                    metadata={"parser": "local:json", "error": str(exc)},
                )

            if not text.strip():
                return ConversionResult(
                    metadata={"parser": "local:json"},
                    quality=score_text_quality(""),
                )

            if _is_jsonl(filename=filename, mime_type=mime_type):
                return _convert_jsonl_text(text, format_name=self.format_name)

            try:
                data = json.loads(text)
                formatted = json.dumps(data, indent=2, ensure_ascii=False)
                content = "```json\n%s\n```" % formatted
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Invalid %s content; falling back to raw code fence "
                    "(error_type=%s)",
                    self.format_name,
                    type(exc).__name__,
                )
                content = "```json\n%s\n```" % text
                return ConversionResult(
                    content=content,
                    metadata={"parser": "local:json", "parse_error": str(exc)},
                    quality=score_text_quality(content),
                )

            quality = score_text_quality(content)

            return ConversionResult(
                content=content,
                metadata={"parser": "local:json", "needs_ocr": False},
                quality=quality,
            )

        return await asyncio.to_thread(_convert)


def _is_jsonl(*, filename: str, mime_type: str) -> bool:
    lowered_name = filename.lower()
    lowered_mime = mime_type.lower().strip()
    return (
        lowered_name.endswith((".jsonl", ".ndjson"))
        or lowered_mime in _JSONL_MIME_TYPES
    )


def _convert_jsonl_text(text: str, *, format_name: str) -> ConversionResult:
    records: list[object] = []
    try:
        for raw_line in text.splitlines():
            if not raw_line.strip():
                continue
            records.append(json.loads(raw_line))
    except json.JSONDecodeError as exc:
        logger.warning(
            "Invalid %s content; falling back to raw code fence (error_type=%s)",
            format_name,
            type(exc).__name__,
        )
        content = "```jsonl\n%s\n```" % text
        return ConversionResult(
            content=content,
            metadata={
                "parser": "local:json",
                "format": "jsonl",
                "parse_error": str(exc),
            },
            quality=score_text_quality(content),
        )

    formatted = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records
    )
    content = "```jsonl\n%s\n```" % formatted
    return ConversionResult(
        content=content,
        metadata={
            "parser": "local:json",
            "format": "jsonl",
            "record_count": len(records),
            "needs_ocr": False,
        },
        quality=score_text_quality(content),
    )


_JSONL_MIME_TYPES = frozenset(
    {
        "application/jsonl",
        "application/jsonlines",
        "application/ldjson",
        "application/x-ldjson",
        "application/ndjson",
        "application/x-ndjson",
    }
)
