"""Built-in OCR provider names and defaults."""

from __future__ import annotations

MISTRAL_OCR_PROVIDER = "mistral"
GEMINI_OCR_PROVIDER = "gemini"
COMMAND_OCR_PROVIDER = "command"
OCR_PROVIDER_ORDER = (MISTRAL_OCR_PROVIDER, GEMINI_OCR_PROVIDER)
DEFAULT_MISTRAL_OCR_MODEL = "mistral-ocr-latest"
DEFAULT_GEMINI_OCR_MODEL = "gemini-2.5-flash"

__all__ = [
    "COMMAND_OCR_PROVIDER",
    "DEFAULT_GEMINI_OCR_MODEL",
    "DEFAULT_MISTRAL_OCR_MODEL",
    "GEMINI_OCR_PROVIDER",
    "MISTRAL_OCR_PROVIDER",
    "OCR_PROVIDER_ORDER",
]
