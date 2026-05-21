from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConverterSpec:
    key: str
    module_name: str
    class_name: str
    required: bool = False


CONVERTER_SPECS: tuple[ConverterSpec, ...] = (
    ConverterSpec("text", ".text_converter", "TextConverter", required=True),
    ConverterSpec("code", ".code_converter", "CodeConverter", required=True),
    ConverterSpec("html", ".html_converter", "HtmlConverter", required=True),
    ConverterSpec("csv", ".csv_converter", "CsvConverter", required=True),
    ConverterSpec("json", ".json_converter", "JsonConverter", required=True),
    ConverterSpec("xml", ".xml_converter", "XmlConverter"),
    ConverterSpec("pdf", ".pdf_converter", "PdfConverter"),
    ConverterSpec("docx", ".docx_converter", "DocxConverter"),
    ConverterSpec("pptx", ".pptx_converter", "PptxConverter"),
    ConverterSpec("xlsx", ".xlsx_converter", "XlsxConverter"),
    ConverterSpec("image", ".image_converter", "ImageConverter"),
)

CONVERTER_SPEC_BY_KEY = {spec.key: spec for spec in CONVERTER_SPECS}
REQUIRED_CONVERTER_KEYS = tuple(spec.key for spec in CONVERTER_SPECS if spec.required)
