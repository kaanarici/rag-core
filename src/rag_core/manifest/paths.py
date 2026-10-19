"""Encode scope identifiers without changing their public identity."""

from __future__ import annotations
import hashlib
import re


def manifest_identity(label: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(
            f"{label} must be a non-empty scope identifier without surrounding whitespace"
        )
    return value


def manifest_segment(label: str, value: str) -> str:
    value = manifest_identity(label, value)
    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", value) and value not in {".", ".."}:
        return value
    return "~" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def manifest_scope_segments(tenant_id: str, collection: str) -> tuple[str, str]:
    return manifest_segment("tenant_id", tenant_id), manifest_segment(
        "collection", collection
    )
