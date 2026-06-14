from __future__ import annotations

from collections.abc import Mapping

from rag_core.search.stored_payload import SECTION_PAYLOAD_KEYS


def build_section_lookup(
    mappings: list[dict[str, object]] | None,
) -> dict[int, dict[str, object]]:
    if not mappings:
        return {}

    section_lookup: dict[int, dict[str, object]] = {}
    for mapping in mappings:
        raw_index = mapping.get("chunk_index")
        if not isinstance(raw_index, int):
            continue
        section_lookup[raw_index] = with_section_title(mapping)
    return section_lookup


def resolve_section_info(
    *,
    chunk_metadata: Mapping[str, object],
    mapping: dict[str, object] | None,
) -> dict[str, object] | None:
    section_info = chunk_locator_metadata(chunk_metadata)
    if mapping:
        section_info.update(mapping)
    if not section_info:
        return None
    return with_section_title(section_info)


def chunk_locator_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    keys = ("section_id", "section_path", "section_title", *SECTION_PAYLOAD_KEYS)
    return {key: metadata[key] for key in keys if key in metadata}


def with_section_title(mapping: Mapping[str, object]) -> dict[str, object]:
    section_path = mapping.get("section_path")
    section_title = mapping.get("section_title")
    if section_title is None and isinstance(section_path, str) and section_path.strip():
        section_title = section_path.split(">")[-1].strip()
    return {
        **dict(mapping),
        "section_title": section_title,
    }
