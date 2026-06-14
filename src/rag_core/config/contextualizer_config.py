from __future__ import annotations

from dataclasses import dataclass, field


def validate_contextualizer_chunk_cap(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("contextualizer_chunk_cap must be a positive integer")
    return value


def _default_contextualizer_provider() -> str:
    from rag_core.documents.contextualizer_provider_names import (
        ANTHROPIC_CONTEXTUALIZER_ID,
    )

    return ANTHROPIC_CONTEXTUALIZER_ID


@dataclass(frozen=True)
class ContextualizerConfig:
    provider: str = field(default_factory=_default_contextualizer_provider)
    model: str | None = None
    enabled: bool = False
    contextualizer_chunk_cap: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("ContextualizerConfig.provider must be a non-empty string")
        if not isinstance(self.enabled, bool):
            raise ValueError("ContextualizerConfig.enabled must be a boolean")
        if self.model is not None:
            if not isinstance(self.model, str):
                raise ValueError("ContextualizerConfig.model must be a string")
            model = self.model.strip()
            object.__setattr__(self, "model", model or None)
        object.__setattr__(self, "provider", self.provider.strip().lower())
        object.__setattr__(
            self,
            "contextualizer_chunk_cap",
            validate_contextualizer_chunk_cap(self.contextualizer_chunk_cap),
        )
