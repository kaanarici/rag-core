from __future__ import annotations

import re
import xml.etree.ElementTree as ET

LLMS_LINK_RE = re.compile(
    r"^\s*[-*+]\s+\[(?P<title>[^\]]+)\]\((?P<url>[^)\s]+)\)(?P<after>.*)$"
)


def decode_text(content: str | bytes) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8")
    return content


def decode_xml_text(content: str | bytes) -> str:
    if isinstance(content, str):
        return content
    for encoding in ("utf-8-sig", "utf-32", "utf-16"):
        try:
            return content.decode(encoding)
        except UnicodeError:
            continue
    raise ValueError("sitemap XML bytes must use a supported Unicode encoding")


def reject_xml_entities(text: str) -> None:
    lowered = text.casefold()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise ValueError("sitemap XML must not contain DTD or entity declarations")


def children_named(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if local_name(child.tag) == name]


def child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if local_name(child.tag) == name:
            return (child.text or "").strip()
    return ""


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].strip()


def llms_notes(after: str) -> str:
    stripped = after.strip()
    if not stripped.startswith(":"):
        return ""
    return stripped[1:].strip()


def validate_max_urls(max_urls: int) -> None:
    if max_urls <= 0:
        raise ValueError("max_urls must be positive")
