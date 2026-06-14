from __future__ import annotations

from typing import Any


def quality_score_to_metadata(quality: Any) -> dict[str, Any]:
    verdict = getattr(quality, "verdict", "")
    return {
        "verdict": getattr(verdict, "value", verdict) or "",
        "details": getattr(quality, "details", "") or "",
        "char_count": getattr(quality, "char_count", 0) or 0,
        "meaningful_ratio": getattr(quality, "meaningful_ratio", 0.0) or 0.0,
        "mojibake_ratio": getattr(quality, "mojibake_ratio", 0.0) or 0.0,
        "text_to_page_ratio": getattr(quality, "text_to_page_ratio", 0.0) or 0.0,
        "page_count": getattr(quality, "page_count", 0) or 0,
    }


__all__ = ["quality_score_to_metadata"]
