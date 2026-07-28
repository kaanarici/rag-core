from __future__ import annotations

from dataclasses import dataclass

from .converter_keys import (
    CODE_CONVERTER_KEY,
    CSV_CONVERTER_KEY,
    DOCX_CONVERTER_KEY,
    HTML_CONVERTER_KEY,
    IMAGE_CONVERTER_KEY,
    JSON_CONVERTER_KEY,
    PDF_CONVERTER_KEY,
    PPTX_CONVERTER_KEY,
    TEXT_CONVERTER_KEY,
    XLSX_CONVERTER_KEY,
    XML_CONVERTER_KEY,
)


@dataclass(frozen=True)
class ConverterSpec:
    key: str
    module_name: str
    class_name: str
    required: bool = False


CONVERTER_SPECS: tuple[ConverterSpec, ...] = (
    ConverterSpec(
        TEXT_CONVERTER_KEY, ".text_converter", "TextConverter", required=True
    ),
    ConverterSpec(
        CODE_CONVERTER_KEY, ".code_converter", "CodeConverter", required=True
    ),
    ConverterSpec(
        HTML_CONVERTER_KEY, ".html_converter", "HtmlConverter", required=True
    ),
    ConverterSpec(CSV_CONVERTER_KEY, ".csv_converter", "CsvConverter", required=True),
    ConverterSpec(
        JSON_CONVERTER_KEY, ".json_converter", "JsonConverter", required=True
    ),
    ConverterSpec(XML_CONVERTER_KEY, ".xml_converter", "XmlConverter"),
    ConverterSpec(PDF_CONVERTER_KEY, ".pdf_converter", "PdfConverter"),
    ConverterSpec(DOCX_CONVERTER_KEY, ".docx_converter", "DocxConverter"),
    ConverterSpec(PPTX_CONVERTER_KEY, ".pptx_converter", "PptxConverter"),
    ConverterSpec(XLSX_CONVERTER_KEY, ".xlsx_converter", "XlsxConverter"),
    ConverterSpec(IMAGE_CONVERTER_KEY, ".image_converter", "ImageConverter"),
)

CONVERTER_SPEC_BY_KEY = {spec.key: spec for spec in CONVERTER_SPECS}
REQUIRED_CONVERTER_KEYS = tuple(spec.key for spec in CONVERTER_SPECS if spec.required)
