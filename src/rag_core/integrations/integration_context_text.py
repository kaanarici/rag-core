"""Shared model-facing context text helpers for optional integrations."""

from __future__ import annotations

from typing import Protocol


class ContextPackTextLike(Protocol):
    def as_text(self) -> str: ...


def context_pack_model_text(pack: ContextPackTextLike) -> str:
    """Return model-safe context text when available, else ``as_text()``."""

    as_model_text = getattr(pack, "as_model_text", None)
    if callable(as_model_text):
        value = as_model_text()
        if not isinstance(value, str):
            raise ValueError("context pack model text must be a string")
        return value
    return pack.as_text()


__all__ = ["ContextPackTextLike", "context_pack_model_text"]
