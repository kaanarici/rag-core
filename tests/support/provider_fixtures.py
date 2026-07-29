"""Provider replay helpers: fixture loading, redaction, and VCR record mode."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "providers"
_CASSETTE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "providers" / "vcr_cassettes"

_REDACT_PATTERNS = (
    re.compile(r"(Authorization:\s*)([^\r\n]+)", re.IGNORECASE),
    re.compile(r'("api_key"\s*:\s*)"[^"]+"', re.IGNORECASE),
    re.compile(r"(api-key:\s*)([^\r\n]+)", re.IGNORECASE),
)


def provider_fixture_path(*parts: str) -> Path:
    return _FIXTURE_ROOT.joinpath(*parts)


def load_provider_fixture(*parts: str) -> dict[str, Any]:
    path = provider_fixture_path(*parts)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"provider fixture must be a JSON object: {path}")
    return cast(dict[str, Any], payload)


def scrub_secrets(text: str) -> str:
    redacted = text
    for pattern in _REDACT_PATTERNS:
        redacted = pattern.sub(r"\1***REDACTED***", redacted)
    return redacted


def record_mode() -> str:
    return os.environ.get("PYTEST_RECORD_MODE", "none")


def embeddings_response_from_fixture(payload: dict[str, Any]) -> SimpleNamespace:
    rows = [
        SimpleNamespace(index=row["index"], embedding=row["embedding"])
        for row in payload["data"]
    ]
    return SimpleNamespace(data=rows)


def rerank_response_from_fixture(payload: dict[str, Any]) -> SimpleNamespace:
    rows = [
        SimpleNamespace(index=row["index"], relevance_score=row["relevance_score"])
        for row in payload["results"]
    ]
    return SimpleNamespace(results=rows)


def voyage_embed_response_from_fixture(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(embeddings=payload["embeddings"])


def cohere_embed_response_from_fixture(payload: dict[str, Any]) -> SimpleNamespace:
    embeddings = payload["embeddings"]
    return SimpleNamespace(
        embeddings=SimpleNamespace(float_=embeddings["float"]),
    )


def zeroentropy_embed_response_from_fixture(payload: dict[str, Any]) -> SimpleNamespace:
    rows = [
        SimpleNamespace(embedding=row["embedding"])
        for row in payload["results"]
    ]
    return SimpleNamespace(results=rows)
