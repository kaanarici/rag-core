from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import math
import re
from typing import Protocol, TypedDict

from .tool_contract_requests import (
    SearchUserDocumentsRequest,
    normalize_static_content_types,
    normalize_static_retrieval_scope,
    parse_search_user_documents_request,
    scope_document_ids,
    validate_bound_namespace,
    validate_search_user_documents_bounds,
)
from .tool_contract_schemas import (
    SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
    SEARCH_USER_DOCUMENTS_DEFAULT_RERANK,
    SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH,
    SEARCH_USER_DOCUMENTS_INPUT_SCHEMA,
    SEARCH_USER_DOCUMENTS_LIMIT_MAX,
    SEARCH_USER_DOCUMENTS_LIMIT_MIN,
    SEARCH_USER_DOCUMENTS_MAX_CHARS_MAX,
    SEARCH_USER_DOCUMENTS_MAX_CHARS_MIN,
    SEARCH_USER_DOCUMENTS_MAX_TOKENS_MAX,
    SEARCH_USER_DOCUMENTS_MAX_TOKENS_MIN,
    SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA,
    SEARCH_USER_DOCUMENTS_TOOL_NAME,
    JsonObject,
)

__all__ = (
    "SEARCH_USER_DOCUMENTS_INPUT_SCHEMA",
    "SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT",
    "SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS",
    "SEARCH_USER_DOCUMENTS_DEFAULT_RERANK",
    "SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH",
    "SEARCH_USER_DOCUMENTS_LIMIT_MAX",
    "SEARCH_USER_DOCUMENTS_LIMIT_MIN",
    "SEARCH_USER_DOCUMENTS_MAX_CHARS_MAX",
    "SEARCH_USER_DOCUMENTS_MAX_CHARS_MIN",
    "SEARCH_USER_DOCUMENTS_MAX_TOKENS_MAX",
    "SEARCH_USER_DOCUMENTS_MAX_TOKENS_MIN",
    "SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA",
    "SEARCH_USER_DOCUMENTS_TOOL_NAME",
    "SearchUserDocumentsRequest",
    "SupportsContextPackPromptPayload",
    "ToolContract",
    "normalize_static_content_types",
    "normalize_static_retrieval_scope",
    "parse_search_user_documents_request",
    "scope_document_ids",
    "search_user_documents_tool_contract",
    "search_user_documents_tool_result",
    "validate_bound_namespace",
    "validate_search_user_documents_bounds",
)


class ToolContract(TypedDict):
    tool_name: str
    input_schema: JsonObject
    output_schema: JsonObject


class SupportsContextPackPromptPayload(Protocol):
    def as_prompt_text(self) -> str: ...

    def to_prompt_payload(self) -> dict[str, object]: ...


def search_user_documents_tool_contract() -> ToolContract:
    """Return a copy-safe contract payload for external tool integrations."""
    return {
        "tool_name": SEARCH_USER_DOCUMENTS_TOOL_NAME,
        "input_schema": deepcopy(SEARCH_USER_DOCUMENTS_INPUT_SCHEMA),
        "output_schema": deepcopy(SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA),
    }


def search_user_documents_tool_result(
    pack: SupportsContextPackPromptPayload,
) -> dict[str, object]:
    """Return the canonical JSON tool output for a ``ContextPack``."""
    payload = {
        **pack.to_prompt_payload(),
        "ok": True,
        "context_text": pack.as_prompt_text(),
    }
    _validate_schema_value(payload, SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA, path="payload")
    return payload


def _validate_schema_value(
    value: object,
    schema: Mapping[str, object],
    *,
    path: str,
) -> None:
    ref = schema.get("$ref")
    if isinstance(ref, str):
        definitions = SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA.get("definitions", {})
        if not isinstance(definitions, Mapping):
            raise ValueError(f"{path} references an invalid tool schema definition")
        definition = definitions.get(ref.rsplit("/", 1)[-1])
        if isinstance(definition, Mapping):
            _validate_schema_value(value, definition, path=path)
            return
        raise ValueError(f"{path} references an unknown tool schema definition")

    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        last_error: ValueError | None = None
        for option in one_of:
            if not isinstance(option, Mapping):
                continue
            try:
                _validate_schema_value(value, option, path=path)
            except ValueError as exc:
                last_error = exc
                continue
            return
        if last_error is not None:
            raise last_error
        raise ValueError(f"{path} does not match any allowed tool schema shape")

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        for option_type in schema_type:
            if not isinstance(option_type, str):
                continue
            option_schema = dict(schema)
            option_schema["type"] = option_type
            try:
                _validate_schema_value(value, option_schema, path=path)
            except ValueError:
                continue
            return
        raise ValueError(f"{path} has invalid tool field type")

    if schema_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise ValueError(f"{path} must contain at least {min_items} items")
        if isinstance(max_items, int) and len(value) > max_items:
            raise ValueError(f"{path} must contain at most {max_items} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, path=f"{path}[{index}]")
        return

    if schema_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ValueError(f"{path} has invalid tool schema properties")
        required = schema.get("required", [])
        if isinstance(required, list):
            missing = sorted(
                key for key in required if isinstance(key, str) and key not in value
            )
            if missing:
                raise ValueError(
                    f"{path} is missing required tool fields: " + ", ".join(missing)
                )
        extra = sorted(set(value).difference(properties))
        if schema.get("additionalProperties") is False and extra:
            raise ValueError(
                f"{path} contains unsupported tool fields: " + ", ".join(extra)
            )
        for key, item in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, Mapping):
                _validate_schema_value(item, child_schema, path=f"{path}.{key}")
        return

    if schema_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            raise ValueError(f"{path} must contain at least {min_length} characters")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            raise ValueError(f"{path} does not match the required string pattern")
        return

    if schema_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} must be an integer")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, int | float) and value < minimum:
            raise ValueError(f"{path} must be at least {minimum}")
        if isinstance(maximum, int | float) and value > maximum:
            raise ValueError(f"{path} must be at most {maximum}")
        return

    if schema_type == "number":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{path} must be a number")
        if not _is_finite_number(value):
            raise ValueError(f"{path} must be a finite number")
        return

    if schema_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be a boolean")
        const = schema.get("const")
        if const is not None and value != const:
            raise ValueError(f"{path} must be {const!r}")
        return

    if schema_type == "null":
        if value is not None:
            raise ValueError(f"{path} must be null")
        return


def _is_finite_number(value: int | float) -> bool:
    if isinstance(value, int):
        return True
    return math.isfinite(value)
