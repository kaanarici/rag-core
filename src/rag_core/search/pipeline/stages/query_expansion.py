"""Experimental query expansion transforms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import importlib
import json
import logging
from typing import Any, Protocol, runtime_checkable

from rag_core.events.emit import emit_event
from rag_core.events.trace_payload_fields import QUERY_TRANSFORM_SEARCH_STAGE
from rag_core.events.types import SearchStageCompleted
from rag_core.provider_api_keys import ANTHROPIC_API_KEY_ENVS, first_configured_api_key
from rag_core.search.pipeline.types import PipelineContext, PipelineQuery

DEFAULT_ANTHROPIC_QUERY_EXPANSION_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_QUERY_VARIANT_COUNT = 3
MAX_QUERY_VARIANT_COUNT = 8
_PROMPT_VERSION = "query-expansion-v1"
_LOGGER = logging.getLogger("rag_core.search.pipeline.stages.query_expansion")


@runtime_checkable
class QueryVariantGenerator(Protocol):
    async def generate_variants(self, query: str, *, count: int) -> Sequence[str]: ...

    async def generate_hypothetical_passage(self, query: str) -> str: ...


@dataclass
class MultiQueryTransform:
    generator: QueryVariantGenerator
    variant_count: int = DEFAULT_QUERY_VARIANT_COUNT

    def __post_init__(self) -> None:
        self.variant_count = min(max(1, self.variant_count), MAX_QUERY_VARIANT_COUNT)

    async def transform(
        self, query: PipelineQuery, ctx: PipelineContext
    ) -> PipelineQuery:
        try:
            generated = await self.generator.generate_variants(
                query.query,
                count=self.variant_count,
            )
        except Exception as exc:
            _record_generator_fallback(ctx, type(self).__name__, exc)
            return query
        variants = _normalized_variants(
            [*query.query_variants, *generated],
            original=query.query,
        )
        return replace(query, query_variants=variants)


@dataclass(frozen=True)
class HydeTransform:
    generator: QueryVariantGenerator

    async def transform(
        self, query: PipelineQuery, ctx: PipelineContext
    ) -> PipelineQuery:
        try:
            passage = (
                await self.generator.generate_hypothetical_passage(query.query)
            ).strip()
            if not passage:
                raise ValueError("generator returned an empty hypothetical passage")
        except Exception as exc:
            _record_generator_fallback(ctx, type(self).__name__, exc)
            return query
        return replace(query, dense_query_text=passage, query_vector=None)


class AnthropicQueryVariantGenerator:
    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_QUERY_EXPANSION_MODEL,
        *,
        max_tokens: int = 512,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        if client is not None:
            self._client = client
            return
        resolved_key = first_configured_api_key(
            ANTHROPIC_API_KEY_ENVS,
            explicit_key=api_key,
        )
        self._client = _create_anthropic_client(api_key=resolved_key or None)

    async def generate_variants(self, query: str, *, count: int) -> Sequence[str]:
        payload = await self._ask(_variant_prompt(query, count=count))
        variants = payload.get("queries")
        if not isinstance(variants, list):
            raise ValueError("variant response did not contain queries")
        return [item for item in variants if isinstance(item, str)]

    async def generate_hypothetical_passage(self, query: str) -> str:
        payload = await self._ask(_hyde_prompt(query))
        passage = payload.get("passage")
        if not isinstance(passage, str):
            raise ValueError("HyDE response did not contain a passage")
        return passage

    async def _ask(self, prompt: str) -> dict[str, object]:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0,
            system=(
                "You generate retrieval queries. Return only the requested JSON "
                "object, with no prose."
            ),
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        )
        return _json_object_from_text(_extract_anthropic_text(response))


def _normalized_variants(
    variants: Sequence[str],
    *,
    original: str,
) -> tuple[str, ...]:
    seen = {original.strip()}
    normalized: list[str] = []
    for variant in variants:
        stripped = variant.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        normalized.append(stripped)
        if len(normalized) >= MAX_QUERY_VARIANT_COUNT:
            break
    return tuple(normalized)


def _record_generator_fallback(
    ctx: PipelineContext,
    stage_name: str,
    exc: Exception,
) -> None:
    _LOGGER.warning(
        "%s fell back to the original query after %s",
        stage_name,
        type(exc).__name__,
    )
    emit_event(
        ctx.event_sink,
        SearchStageCompleted(
            stage=QUERY_TRANSFORM_SEARCH_STAGE,
            stage_name=f"{stage_name}.fallback",
            dropped_count=1,
        ),
    )


def _create_anthropic_client(*, api_key: str | None) -> Any:
    try:
        anthropic_module = importlib.import_module("anthropic")
    except ImportError as exc:
        raise ImportError(
            "anthropic package is required for AnthropicQueryVariantGenerator. "
            "Install it with: pip install 'rag-core[anthropic]'"
        ) from exc
    async_client_class = getattr(anthropic_module, "AsyncAnthropic", None)
    if async_client_class is None:
        raise ImportError(
            "anthropic package with AsyncAnthropic is required for "
            "AnthropicQueryVariantGenerator."
        )
    if api_key is not None:
        return async_client_class(api_key=api_key)
    return async_client_class()


def _variant_prompt(query: str, *, count: int) -> str:
    return (
        f"Prompt version: {_PROMPT_VERSION}\n"
        f"Write {count} alternative search queries for this user query.\n\n"
        f"Query:\n{query}\n\n"
        'Return exactly JSON: {"queries": ["..."]}'
    )


def _hyde_prompt(query: str) -> str:
    return (
        f"Prompt version: {_PROMPT_VERSION}\n"
        "Write a concise hypothetical answer passage that would be useful to "
        "embed for semantic retrieval. Do not answer with caveats.\n\n"
        f"Query:\n{query}\n\n"
        'Return exactly JSON: {"passage": "..."}'
    )


def _json_object_from_text(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_code_fence(stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("generator response was not a JSON object") from None
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("generator response was not a JSON object")
    return payload


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _extract_anthropic_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


__all__ = [
    "AnthropicQueryVariantGenerator",
    "HydeTransform",
    "MultiQueryTransform",
    "QueryVariantGenerator",
]
