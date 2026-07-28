"""HTML converter with content extraction, cleanup, and fallback strategies."""

from __future__ import annotations

import re

from .base import ConversionResult, TextLikeConverter
from .converter_keys import HTML_CONVERTER_KEY


def _strip_non_content_html(html: str) -> str:
    """Remove scripts, styles, nav, footer, and other non-content elements."""
    html = re.sub(
        r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
    )
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(
        r"<noscript[^>]*>.*?</noscript>", "", html, flags=re.DOTALL | re.IGNORECASE
    )

    for tag in ("nav", "footer", "aside"):
        html = re.sub(
            r"<%s[^>]*>.*?</%s>" % (tag, tag), "", html, flags=re.DOTALL | re.IGNORECASE
        )

    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    return html


def _try_markdownify(html: str) -> str | None:
    """Convert HTML to markdown using markdownify (preserves structure)."""
    try:
        from markdownify import markdownify

        result = markdownify(html, heading_style="ATX", strip=["img", "iframe"])
        return result.strip() if result else None
    except Exception:
        # Malformed HTML or converter-specific failures should not abort the fallback chain.
        return None


def _try_html_to_markdown(html: str) -> str | None:
    """Convert HTML to markdown using html-to-markdown (Rust-powered)."""
    try:
        from html_to_markdown import convert  # type: ignore[import-not-found]

        result = convert(html)
        return result.strip() if result else None
    except ImportError:
        # html-to-markdown is optional; the converter should degrade to the next strategy.
        return None
    except Exception:
        # Conversion quality may vary by input, so fall through to the next extractor.
        return None


def _regex_fallback(html: str) -> str:
    """Strip HTML tags via regex (last resort)."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class HtmlConverter(TextLikeConverter):
    """Converts HTML to markdown with content extraction.

    Strips non-content elements (scripts, styles, nav, footer) before
    conversion. Uses a layered fallback strategy:
    1. markdownify (CORE dep; best structure preservation)
    2. html-to-markdown (``html`` extra; exception-path fallback)
    3. Regex strip (last resort)
    """

    format_name = HTML_CONVERTER_KEY
    parser_label = "local:html"

    def _render(
        self,
        text: str,
        filename: str,
        mime_type: str,
    ) -> tuple[str, dict[str, object]] | ConversionResult:
        cleaned = _strip_non_content_html(text)

        parser_name = "local:html"
        content = None

        result = _try_markdownify(cleaned)
        if result:
            content = result
            parser_name = "local:markdownify"

        if not content:
            result = _try_html_to_markdown(cleaned)
            if result:
                content = result
                parser_name = "local:html-to-markdown"

        if not content:
            content = _regex_fallback(cleaned)
            parser_name = "local:html-regex"

        return content or "", {"parser": parser_name, "needs_ocr": False}
