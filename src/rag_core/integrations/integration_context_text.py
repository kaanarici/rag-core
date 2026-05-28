"""Prompt-facing context text helpers for optional integrations."""

from __future__ import annotations

from rag_core.contracts import SupportsContextPackPromptPayload


def context_pack_prompt_text(pack: SupportsContextPackPromptPayload) -> str:
    """Return prompt-safe context text for agent tool responses."""
    return pack.as_prompt_text()


__all__ = ["context_pack_prompt_text"]
