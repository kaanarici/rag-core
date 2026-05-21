"""Code-aware chunking with AST-first and regex-fallback strategies."""

from __future__ import annotations

import logging
from typing import Optional

from rag_core.config.env_access import get_env_bool
from rag_core.core_models import PreparedChunk

from .code_language_support import (
    FALLBACK_PATTERNS,
    LANGUAGE_PATTERNS,
)
from .code_runtime import (
    ast_boundaries_for_language,
    detect_language_with_magika,
    tree_sitter_backend_available,
)
from .code_segments import (
    assemble_code_chunks,
    build_code_chunk_metadata,
    mask_non_code_regions,
    segments_from_boundaries,
)
from .protocol import ChunkConfig

logger = logging.getLogger(__name__)


class CodeChunker:
    """Chunks source code by preferring AST boundaries and falling back to regex."""

    def __init__(
        self,
        language: Optional[str] = None,
        *,
        skip_unsupported_language: Optional[bool] = None,
        enable_magika_detection: Optional[bool] = None,
    ) -> None:
        self._language = language.lower() if language else None
        self._skip_unsupported_language = (
            get_env_bool("CHUNKING_SKIP_UNSUPPORTED_CODE", False)
            if skip_unsupported_language is None
            else skip_unsupported_language
        )
        self._enable_magika_detection = (
            get_env_bool("CHUNKING_ENABLE_MAGIKA_LANGUAGE_DETECTION", True)
            if enable_magika_detection is None
            else enable_magika_detection
        )
        self._magika: object | None = None

    def _detect_language_with_magika(self, text: str) -> Optional[str]:
        if not self._enable_magika_detection:
            return None

        detection = detect_language_with_magika(text, detector=self._magika)
        self._magika = detection.detector
        return detection.language

    def _resolve_language(self, text: str) -> Optional[str]:
        if self._language:
            return self._language
        return self._detect_language_with_magika(text)

    def _regex_boundaries(self, text: str, language: Optional[str]) -> list[int]:
        patterns = LANGUAGE_PATTERNS.get(language or "", FALLBACK_PATTERNS)
        masked = mask_non_code_regions(text)
        boundaries = {0}

        for pattern in patterns:
            for match in pattern.finditer(masked):
                boundaries.add(match.start())

        return sorted(boundaries)

    def chunk(self, text: str, config: ChunkConfig) -> list[PreparedChunk]:
        if not text:
            return []

        resolved_language = self._resolve_language(text)
        ast_boundaries = ast_boundaries_for_language(text, resolved_language)

        if (
            ast_boundaries is None
            and resolved_language
            and self._skip_unsupported_language
            and tree_sitter_backend_available()
        ):
            logger.info(
                "Skipping code chunking for unsupported tree-sitter language '%s'",
                resolved_language,
            )
            return []

        chunking_engine = "ast" if ast_boundaries else "regex"
        boundaries = ast_boundaries or self._regex_boundaries(text, resolved_language)
        segments = segments_from_boundaries(text, boundaries)
        metadata = build_code_chunk_metadata(
            chunking_engine=chunking_engine,
            resolved_language=resolved_language,
        )
        return assemble_code_chunks(
            text=text,
            segments=segments,
            config=config,
            metadata=metadata,
        )
