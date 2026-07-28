from __future__ import annotations

from rag_core.search.planning import (
    describe_query_plan_presets,
    describe_search_profiles,
)


def search_profile_help(*, prefix: str, suffix: str) -> str:
    return _catalog_help(
        prefix=prefix,
        label="Profiles",
        catalog=describe_search_profiles(),
        suffix=suffix,
    )


def query_plan_preset_help(*, prefix: str, suffix: str) -> str:
    return _catalog_help(
        prefix=prefix,
        label="Presets",
        catalog=describe_query_plan_presets(),
        suffix=suffix,
    )


def _catalog_help(
    *,
    prefix: str,
    label: str,
    catalog: dict[str, dict[str, object]],
    suffix: str,
) -> str:
    choices = "; ".join(
        f"{name}={item.get('summary')}" for name, item in catalog.items()
    )
    return f"{prefix} {label}: {choices}. {suffix}"


__all__ = ["query_plan_preset_help", "search_profile_help"]
