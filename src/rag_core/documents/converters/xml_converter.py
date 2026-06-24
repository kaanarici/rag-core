"""XML converter that pretty-prints XML in a markdown code fence."""

from __future__ import annotations

import logging
from defusedxml import minidom

from .base import ConversionResult, TextLikeConverter
from .converter_keys import XML_CONVERTER_KEY

logger = logging.getLogger(__name__)


class XmlConverter(TextLikeConverter):
    """Converts XML files to pretty-printed markdown code fences."""

    format_name = XML_CONVERTER_KEY
    parser_label = "local:xml"

    def _render(
        self,
        text: str,
        filename: str,
        mime_type: str,
    ) -> tuple[str, dict[str, object]] | ConversionResult:
        try:
            dom = minidom.parseString(text)
            formatted = dom.toprettyxml(indent="  ")
            content = "```xml\n%s\n```" % formatted
        except Exception as exc:
            logger.debug(
                "%s parsing failed; falling back to raw code fence (error_type=%s)",
                self.format_name.upper(),
                type(exc).__name__,
            )
            content = "```xml\n%s\n```" % text

        return content, {"parser": "local:xml", "needs_ocr": False}
