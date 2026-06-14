from __future__ import annotations

from typing import Final


TEXT_CONVERTER_KEY: Final[str] = "text"
CODE_CONVERTER_KEY: Final[str] = "code"
HTML_CONVERTER_KEY: Final[str] = "html"
CSV_CONVERTER_KEY: Final[str] = "csv"
JSON_CONVERTER_KEY: Final[str] = "json"
XML_CONVERTER_KEY: Final[str] = "xml"
PDF_CONVERTER_KEY: Final[str] = "pdf"
DOCX_CONVERTER_KEY: Final[str] = "docx"
PPTX_CONVERTER_KEY: Final[str] = "pptx"
XLSX_CONVERTER_KEY: Final[str] = "xlsx"
IMAGE_CONVERTER_KEY: Final[str] = "image"

CONVERTER_KEYS: tuple[str, ...] = (
    TEXT_CONVERTER_KEY,
    CODE_CONVERTER_KEY,
    HTML_CONVERTER_KEY,
    CSV_CONVERTER_KEY,
    JSON_CONVERTER_KEY,
    XML_CONVERTER_KEY,
    PDF_CONVERTER_KEY,
    DOCX_CONVERTER_KEY,
    PPTX_CONVERTER_KEY,
    XLSX_CONVERTER_KEY,
    IMAGE_CONVERTER_KEY,
)
