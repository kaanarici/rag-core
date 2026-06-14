from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum


class QualityVerdict(str, Enum):
    """Whether extracted text is good enough or OCR is needed."""

    GOOD = "good"
    POOR = "poor"
    EMPTY = "empty"


@dataclass
class QualityScore:
    """Multi-signal quality assessment of extracted text."""

    char_count: int = 0
    meaningful_ratio: float = 0.0
    mojibake_ratio: float = 0.0
    text_to_page_ratio: float = 0.0
    page_count: int = 0
    verdict: QualityVerdict = QualityVerdict.EMPTY
    details: str = ""


def score_text_quality(
    text: str,
    *,
    page_count: int = 1,
    min_chars: int = 50,
    min_meaningful_ratio: float = 0.3,
    max_mojibake_ratio: float = 0.1,
    min_chars_per_page: float = 20.0,
) -> QualityScore:
    """Score extracted text quality using content, encoding, and page signals."""
    if not text or not text.strip():
        return QualityScore(verdict=QualityVerdict.EMPTY, details="no text")

    stripped = text.strip()
    char_count = len(stripped)
    meaningful = sum(1 for c in stripped if c.isalnum() or c in " \t\n.,;:!?-")
    meaningful_ratio = meaningful / char_count if char_count > 0 else 0.0
    mojibake_count = _mojibake_character_count(stripped)
    mojibake_ratio = mojibake_count / char_count if char_count > 0 else 0.0
    pages = max(1, page_count)
    text_to_page_ratio = char_count / pages

    score = QualityScore(
        char_count=char_count,
        meaningful_ratio=meaningful_ratio,
        mojibake_ratio=mojibake_ratio,
        text_to_page_ratio=text_to_page_ratio,
        page_count=page_count,
    )
    _apply_quality_verdict(
        score,
        min_chars=min_chars,
        min_meaningful_ratio=min_meaningful_ratio,
        max_mojibake_ratio=max_mojibake_ratio,
        min_chars_per_page=min_chars_per_page,
    )
    return score


def _mojibake_character_count(text: str) -> int:
    count = 0
    for c in text:
        if c == "\ufffd":
            count += 1
        elif _is_corrupt_control_character(c):
            count += 1
        elif unicodedata.category(c) in ("Co", "Cn"):
            count += 1
    return count


def _is_corrupt_control_character(char: str) -> bool:
    if char in "\n\r\t\f":
        return False
    return unicodedata.category(char) == "Cc"


def _apply_quality_verdict(
    score: QualityScore,
    *,
    min_chars: int,
    min_meaningful_ratio: float,
    max_mojibake_ratio: float,
    min_chars_per_page: float,
) -> None:
    if score.char_count < min_chars:
        score.verdict = QualityVerdict.POOR
        score.details = "below minimum char count (%d < %d)" % (
            score.char_count,
            min_chars,
        )
    elif score.meaningful_ratio < min_meaningful_ratio:
        score.verdict = QualityVerdict.POOR
        score.details = "low meaningful ratio (%.2f < %.2f)" % (
            score.meaningful_ratio,
            min_meaningful_ratio,
        )
    elif score.mojibake_ratio > max_mojibake_ratio:
        score.verdict = QualityVerdict.POOR
        score.details = "high mojibake/control ratio (%.2f > %.2f)" % (
            score.mojibake_ratio,
            max_mojibake_ratio,
        )
    elif score.page_count > 1 and score.text_to_page_ratio < min_chars_per_page:
        score.verdict = QualityVerdict.POOR
        score.details = "low chars per page (%.1f < %.1f)" % (
            score.text_to_page_ratio,
            min_chars_per_page,
        )
    else:
        score.verdict = QualityVerdict.GOOD
        score.details = "quality OK"


def is_char_count_only_quality_failure(score: QualityScore) -> bool:
    return score.verdict == QualityVerdict.POOR and score.details.startswith(
        "below minimum char count"
    )


__all__ = [
    "QualityScore",
    "QualityVerdict",
    "is_char_count_only_quality_failure",
    "score_text_quality",
]
