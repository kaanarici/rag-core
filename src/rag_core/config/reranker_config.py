from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RerankerConfig:
    provider: str = "none"
    model: str | None = None
    api_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("RerankerConfig.provider must be a non-empty string")
        object.__setattr__(self, "provider", self.provider.strip().lower())
