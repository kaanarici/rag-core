"""Text file converter with encoding detection.

Handles TXT, MD, YAML, TOML, and other plain text formats.
"""

from __future__ import annotations

import asyncio
import logging

from .base import (
    BaseConverter,
    ConversionResult,
    QualityVerdict,
    safe_decode,
    score_text_quality,
)
from .converter_keys import TEXT_CONVERTER_KEY

logger = logging.getLogger(__name__)


class TextConverter(BaseConverter):
    """Converts plain text files (TXT, MD, YAML, TOML, etc.) with encoding detection."""

    format_name = TEXT_CONVERTER_KEY

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert text file with encoding detection."""

        def _convert() -> ConversionResult:
            try:
                content = safe_decode(file_bytes)
            except ValueError:
                return ConversionResult(
                    metadata={
                        "parser": "local:text",
                        "error": "binary content detected",
                    },
                )

            quality = score_text_quality(content)
            if quality.verdict == QualityVerdict.POOR and (
                quality.mojibake_ratio > 0.1 or quality.meaningful_ratio < 0.2
            ):
                return ConversionResult(
                    metadata={
                        "parser": "local:text",
                        "error": "binary content detected",
                    },
                )

            return ConversionResult(
                content=content,
                metadata={"parser": "local:text", "needs_ocr": False},
                quality=quality,
            )

        return await asyncio.to_thread(_convert)
