from __future__ import annotations

import codecs
import re
from typing import List


_BOM_ENCODINGS = (
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF16_BE, "utf-16"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF8, "utf-8-sig"),
)


def text_to_markdown(text: str) -> str:
    """Convert extracted plain text to basic Markdown with heuristics.

    Detects headings, bullet points, and numbered lists.
    """
    if not text:
        return ""

    lines = text.split("\n")
    result: List[str] = []
    prev_blank = True

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if not prev_blank:
                result.append("")
                prev_blank = True
            continue

        prev_blank = False

        if len(stripped) >= 2 and stripped[0] in "\u2022\u00b7\u25e6" and stripped[1] == " ":
            result.append("- %s" % stripped[2:].strip())
            continue

        if len(stripped) >= 2 and stripped[0] in "*-" and stripped[1] == " ":
            result.append("- %s" % stripped[2:].strip())
            continue

        if re.match(r"^\d+[.)]\s", stripped):
            result.append(stripped)
            continue

        is_short = len(stripped) < 80
        is_uppercase = stripped.isupper() and len(stripped) > 3
        is_titlecase = stripped.istitle() and len(stripped) < 60
        ends_with_punct = stripped.endswith((".", ",", ";", ":", "?", "!"))

        if is_short and is_uppercase and not ends_with_punct:
            result.append("## %s" % stripped.title())
        elif is_short and is_titlecase and not ends_with_punct:
            result.append("## %s" % stripped)
        else:
            result.append(stripped)

    return "\n".join(result)


def render_markdown_table(rows: List[List[str]]) -> str:
    """Render rows as a markdown table with header separator."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    padded = [
        [_escape_markdown_table_cell(cell) for cell in r + [""] * (width - len(r))]
        for r in rows
    ]

    header = padded[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in padded[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _escape_markdown_table_cell(cell: str) -> str:
    return cell.replace("|", r"\|")


def detect_encoding(raw_bytes: bytes, sample_size: int = 100_000) -> str:
    """Detect encoding of raw bytes using declared runtime dependencies.

    Returns the detected encoding name, defaulting to 'utf-8'.
    """
    for bom, encoding in _BOM_ENCODINGS:
        if raw_bytes.startswith(bom):
            return encoding

    unicode_encoding = _detect_bomless_unicode(raw_bytes[:sample_size])
    if unicode_encoding is not None:
        return unicode_encoding

    try:
        raw_bytes.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    sample = raw_bytes[:sample_size]
    if _looks_like_text(sample, encoding="cp1252"):
        return "cp1252"

    return "utf-8"


def _detect_bomless_unicode(sample: bytes) -> str | None:
    if len(sample) < 2:
        return None
    if len(sample) >= 16:
        if _utf32_zero_ratio(sample, zero_positions=(0, 1, 2)) > 0.75:
            return "utf-32-be"
        if _utf32_zero_ratio(sample, zero_positions=(1, 2, 3)) > 0.75:
            return "utf-32-le"
    sample = sample[: len(sample) - (len(sample) % 2)]
    even_nuls = sample[0::2].count(0)
    odd_nuls = sample[1::2].count(0)
    pair_count = max(1, len(sample) // 2)
    if (
        odd_nuls / pair_count > 0.35
        and even_nuls / pair_count < 0.05
        and _looks_like_text(sample, encoding="utf-16-le")
    ):
        return "utf-16-le"
    if (
        even_nuls / pair_count > 0.35
        and odd_nuls / pair_count < 0.05
        and _looks_like_text(sample, encoding="utf-16-be")
    ):
        return "utf-16-be"
    return None


def _utf32_zero_ratio(sample: bytes, *, zero_positions: tuple[int, int, int]) -> float:
    groups = [sample[index : index + 4] for index in range(0, len(sample) - 3, 4)]
    if not groups:
        return 0.0
    matches = sum(
        1
        for group in groups
        if all(group[position] == 0 for position in zero_positions)
    )
    return matches / len(groups)


def _looks_like_text(raw_bytes: bytes, *, encoding: str) -> bool:
    try:
        text = raw_bytes.decode(encoding)
    except UnicodeDecodeError:
        return False
    if not text:
        return True
    control_count = sum(1 for char in text if _is_unexpected_control(char))
    return control_count / len(text) <= 0.02


def _is_unexpected_control(char: str) -> bool:
    if char in "\n\r\t\f":
        return False
    codepoint = ord(char)
    return codepoint < 32 or 0x7F <= codepoint <= 0x9F


def _is_probably_binary_payload(raw_bytes: bytes, *, encoding: str) -> bool:
    if not raw_bytes:
        return False

    if encoding.startswith(("utf-16", "utf-32")):
        return False

    sample = raw_bytes[:100_000]
    if b"\x00" in sample:
        return True

    if len(sample) < 32:
        return False

    suspicious_control_count = sum(
        1
        for value in sample
        if value not in (9, 10, 12, 13) and (value < 32 or value == 127)
    )
    return suspicious_control_count / len(sample) > 0.05


def safe_decode(raw_bytes: bytes, max_replacement_ratio: float = 0.25) -> str:
    """Decode bytes to string with encoding detection and corruption check."""
    if not raw_bytes:
        return ""

    encoding = detect_encoding(raw_bytes)
    if _is_probably_binary_payload(raw_bytes, encoding=encoding):
        raise ValueError("binary content detected")
    try:
        text = raw_bytes.decode(encoding)
        if "\ufffd" not in text:
            return text
    except (UnicodeDecodeError, LookupError):
        pass

    text = raw_bytes.decode("utf-8", errors="replace")
    replacement_count = text.count("\ufffd")

    if replacement_count > 0:
        ratio = replacement_count / len(text) if len(text) > 0 else 0
        if ratio > max_replacement_ratio or replacement_count > 5000:
            raise ValueError(
                "Content appears binary/corrupted: %d replacement chars (%.1f%%)"
                % (replacement_count, ratio * 100)
            )

    return text
