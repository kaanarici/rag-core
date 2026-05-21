"""Optional runtime helpers for code chunking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

from .code_language_support import (
    MAGIKA_TO_INTERNAL_LANGUAGE,
    TREE_SITTER_LANGUAGE_CANDIDATES,
)


class TreeSitterParser(Protocol):
    def parse(self, source: bytes) -> Any: ...


class MagikaDetector(Protocol):
    def identify_bytes(self, data: bytes) -> Any: ...


@dataclass(frozen=True)
class MagikaDetection:
    language: str | None
    detector: MagikaDetector | None


def tree_sitter_backend_available() -> bool:
    try:
        from tree_sitter_language_pack import get_parser  # noqa: F401
    except ImportError:
        return False
    return True


def ast_boundaries_for_language(text: str, language: str | None) -> list[int] | None:
    if not language:
        return None

    parser = _get_tree_sitter_parser(language)
    if parser is None:
        return None

    try:
        tree = parser.parse(text.encode("utf-8", errors="ignore"))
        root = tree.root_node
    except Exception:
        return None

    boundaries = {0}
    named_children = getattr(root, "named_children", None)
    children = named_children if named_children else getattr(root, "children", [])

    for child in children:
        start = int(getattr(child, "start_byte", 0))
        end = int(getattr(child, "end_byte", 0))
        if end - start < 8:
            continue
        child_type = str(getattr(child, "type", ""))
        if child_type == "comment":
            continue
        boundaries.add(start)

    result = sorted(boundaries)
    return result if len(result) > 1 else None


def detect_language_with_magika(
    text: str,
    *,
    detector: object | None,
) -> MagikaDetection:
    active_detector = detector
    if active_detector is None:
        try:
            from magika import Magika
        except ImportError:
            return MagikaDetection(language=None, detector=None)
        active_detector = cast(MagikaDetector, Magika())

    typed_detector = cast(MagikaDetector, active_detector)
    try:
        result = typed_detector.identify_bytes(text.encode("utf-8", errors="ignore"))
        label = str(result.output.label).lower()
    except Exception:
        return MagikaDetection(language=None, detector=typed_detector)

    return MagikaDetection(
        language=MAGIKA_TO_INTERNAL_LANGUAGE.get(label, label),
        detector=typed_detector,
    )


def _get_tree_sitter_parser(language: str) -> TreeSitterParser | None:
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        return None

    for candidate in _language_candidates(language):
        try:
            return cast(TreeSitterParser, get_parser(candidate))
        except Exception:
            continue
    return None


def _language_candidates(language: str) -> tuple[str, ...]:
    candidates = TREE_SITTER_LANGUAGE_CANDIDATES.get(language)
    return tuple(candidates) if candidates is not None else (language,)
