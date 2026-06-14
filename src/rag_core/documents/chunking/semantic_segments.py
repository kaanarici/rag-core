"""Sentence and boundary helpers for semantic chunking."""

from __future__ import annotations

import math
import re

_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = _SENTENCE_PATTERN.split(text)
    result: list[str] = []
    for sentence in sentences:
        parts = sentence.split("\n\n")
        result.extend(part.strip() for part in parts if part.strip())
    return result


def segments_from_semantic_boundaries(
    sentences: list[str],
    embeddings: list[list[float]],
    *,
    similarity_threshold: float,
) -> list[str]:
    boundaries: list[int] = [0]
    for idx in range(1, len(embeddings)):
        similarity = _cosine_similarity(embeddings[idx - 1], embeddings[idx])
        if similarity < similarity_threshold:
            boundaries.append(idx)

    segments: list[str] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
        chunk_text = " ".join(sentences[start:end]).strip()
        if chunk_text:
            segments.append(chunk_text)
    return segments


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
