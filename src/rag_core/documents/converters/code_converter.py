"""Code file converter with language detection metadata."""

from __future__ import annotations

import asyncio
import logging
import os

from .base import BaseConverter, ConversionResult, score_text_quality
from .converter_keys import CODE_CONVERTER_KEY

logger = logging.getLogger(__name__)

LANGUAGE_MAP: dict[str, str] = {
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".vue": "vue",
    ".svelte": "svelte",
    ".py": "python",
    ".rb": "ruby",
    ".php": "php",
    ".java": "java",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".m": "objectivec",
    ".d": "d",
    ".jl": "julia",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".clj": "clojure",
    ".groovy": "groovy",
    ".dart": "dart",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".fs": "fsharp",
    ".nim": "nim",
    ".cr": "crystal",
    ".zig": "zig",
    ".lua": "lua",
    ".pl": "perl",
    ".r": "r",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".ps1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    ".tf": "terraform",
    ".hcl": "hcl",
    ".sql": "sql",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".proto": "protobuf",
    ".gradle": "gradle",
    ".cmake": "cmake",
    ".make": "makefile",
    ".mak": "makefile",
}


def detect_language(filename: str) -> str:
    """Detect programming language from filename extension.

    Returns the language identifier for code fences, or 'text' if unknown.
    """
    _, ext = os.path.splitext(filename.lower())
    return LANGUAGE_MAP.get(ext, "text")


class CodeConverter(BaseConverter):
    """Converts code files to text with language metadata for chunking."""

    format_name = CODE_CONVERTER_KEY

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
    ) -> ConversionResult:
        """Convert code file with language detection."""

        def _convert() -> ConversionResult:
            if not file_bytes:
                return ConversionResult(
                    metadata={"parser": "local:code"},
                    quality=score_text_quality(""),
                )

            try:
                code = file_bytes.decode("utf-8")
                if "\ufffd" in code:
                    raise UnicodeDecodeError("utf-8", b"", 0, 1, "replacement chars")
            except UnicodeDecodeError:
                code = file_bytes.decode("utf-8", errors="replace")
                replacement_count = code.count("\ufffd")
                if replacement_count > 0:
                    logger.warning(
                        "%s content has %d replacement chars, may be binary",
                        self.format_name,
                        replacement_count,
                    )
                    return ConversionResult(
                        metadata={
                            "parser": "local:code",
                            "error": "binary content detected",
                        },
                    )

            language = detect_language(filename)
            quality = score_text_quality(code)

            metadata: dict[str, str | bool] = {
                "parser": "local:code",
                "language": language,
                "needs_ocr": False,
            }

            return ConversionResult(
                content=code,
                metadata=metadata,
                quality=quality,
            )

        return await asyncio.to_thread(_convert)
