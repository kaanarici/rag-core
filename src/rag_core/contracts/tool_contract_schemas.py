from __future__ import annotations

from typing import Any, Final

from rag_core.retrieval_defaults import DEFAULT_RERANK, DEFAULT_USE_LEXICAL_SEARCH

JsonObject = dict[str, Any]

SEARCH_USER_DOCUMENTS_TOOL_NAME: Final[str] = "search_user_documents"
SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT: Final[int] = 5
SEARCH_USER_DOCUMENTS_DEFAULT_RERANK: Final[bool] = DEFAULT_RERANK
SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH: Final[bool] = DEFAULT_USE_LEXICAL_SEARCH
SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS: Final[int] = 3000
SEARCH_USER_DOCUMENTS_LIMIT_MIN: Final[int] = 1
SEARCH_USER_DOCUMENTS_LIMIT_MAX: Final[int] = 20
SEARCH_USER_DOCUMENTS_MAX_CHARS_MIN: Final[int] = 256
SEARCH_USER_DOCUMENTS_MAX_CHARS_MAX: Final[int] = 12000
SEARCH_USER_DOCUMENTS_MAX_TOKENS_MIN: Final[int] = 64
SEARCH_USER_DOCUMENTS_MAX_TOKENS_MAX: Final[int] = 4000

SEARCH_USER_DOCUMENTS_INPUT_SCHEMA: Final[JsonObject] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "search_user_documents.input",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "pattern": r"\S",
            "description": "Natural-language user query to run against app-owned documents.",
        },
        "limit": {
            "type": "integer",
            "minimum": SEARCH_USER_DOCUMENTS_LIMIT_MIN,
            "maximum": SEARCH_USER_DOCUMENTS_LIMIT_MAX,
            "default": SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
            "description": "Maximum number of context snippets to return.",
        },
        "document_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "pattern": r"\S"},
            "description": "Optional narrowing filter inside the app-bound document scope.",
        },
        "rerank": {
            "type": "boolean",
            "default": SEARCH_USER_DOCUMENTS_DEFAULT_RERANK,
            "description": "Whether the endpoint should apply reranking.",
        },
        "use_lexical_search": {
            "type": "boolean",
            "default": SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH,
            "description": (
                "Controls configured lexical/exact-match expansion only; "
                "query-plan defaults remain provider capability-aware."
            ),
        },
        "max_chars": {
            "type": "integer",
            "minimum": SEARCH_USER_DOCUMENTS_MAX_CHARS_MIN,
            "maximum": SEARCH_USER_DOCUMENTS_MAX_CHARS_MAX,
            "default": SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
            "description": "Approximate character budget for the returned snippets.",
        },
        "max_tokens": {
            "type": "integer",
            "minimum": SEARCH_USER_DOCUMENTS_MAX_TOKENS_MIN,
            "maximum": SEARCH_USER_DOCUMENTS_MAX_TOKENS_MAX,
            "description": "Approximate token budget for the returned snippets.",
        },
    },
    "required": ["query"],
}
_SEARCH_USER_DOCUMENTS_INPUT_FIELDS: Final[frozenset[str]] = frozenset(
    SEARCH_USER_DOCUMENTS_INPUT_SCHEMA["properties"]
)

_PROMPT_SOURCE_REFERENCE_SCHEMA: Final[JsonObject] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "citation_id": {"type": "string"},
        "title": {"type": "string"},
        "section_title": {"type": "string"},
        "section_path": {"type": "string"},
        "chunk_index": {"type": "integer"},
        "source_type": {"type": "string"},
        "result_type": {"type": "string"},
    },
    "required": ["citation_id"],
}

_PROMPT_SOURCE_LOCATOR_SCHEMA: Final[JsonObject] = {
    "type": "object",
    "description": (
        "Prompt-safe source locator projection; app-private source hashes are omitted."
    ),
    "additionalProperties": False,
    "properties": {
        "chunk_index": {"type": ["integer", "null"]},
        "section_path": {"type": ["string", "null"]},
        "page_number": {"type": ["integer", "null"]},
        "page_index": {"type": ["integer", "null"]},
        "slide_number": {"type": ["integer", "null"]},
        "sheet_name": {"type": ["string", "null"]},
        "row_range": {"type": ["string", "null"]},
        "line_start": {"type": ["integer", "null"]},
        "line_end": {"type": ["integer", "null"]},
        "start_offset": {"type": ["integer", "null"]},
        "end_offset": {"type": ["integer", "null"]},
        "bbox": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
            ]
        },
        "figure_id": {"type": ["string", "null"]},
        "figure_caption": {"type": ["string", "null"]},
        "figure_thumbnail_url": {"type": ["string", "null"]},
    },
    "required": [
        "chunk_index",
        "section_path",
        "page_number",
        "page_index",
        "slide_number",
        "sheet_name",
        "row_range",
        "line_start",
        "line_end",
        "start_offset",
        "end_offset",
        "bbox",
        "figure_id",
        "figure_caption",
        "figure_thumbnail_url",
    ],
}

_PROMPT_SOURCE_PREVIEW_SCHEMA: Final[JsonObject] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "citation_id": {"type": "string"},
        "title": {"type": "string"},
        "locator_label": {"type": ["string", "null"]},
        "source_type": {"type": ["string", "null"]},
        "result_type": {"type": ["string", "null"]},
        "truncated": {"type": "boolean"},
    },
    "required": [
        "citation_id",
        "title",
        "locator_label",
        "source_type",
        "result_type",
        "truncated",
    ],
}

_QUALITY_METADATA_SCHEMA: Final[JsonObject] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string"},
        "details": {"type": "string"},
        "char_count": {"type": "integer"},
        "page_count": {"type": "integer"},
        "meaningful_ratio": {"type": "number"},
        "mojibake_ratio": {"type": "number"},
        "text_to_page_ratio": {"type": "number"},
    },
}

_CONTEXT_SNIPPET_SCHEMA: Final[JsonObject] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "citation_id": {"type": "string"},
        "rank": {"type": "integer", "minimum": 1},
        "text": {"type": "string"},
        "score": {"type": "number"},
        "source": {"$ref": "#/definitions/prompt_source_reference"},
        "locator": {"$ref": "#/definitions/prompt_source_locator"},
        "token_estimate": {"type": "integer", "minimum": 0},
        "char_count": {"type": "integer", "minimum": 0},
        "retrieval_metadata": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "quality": _QUALITY_METADATA_SCHEMA,
                "rerank": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "provider": {"type": "string"},
                        "model": {"type": "string"},
                        "provider_score": {"type": "number"},
                        "search_score": {"type": "number"},
                        "original_rank": {"type": "integer"},
                        "rerank_rank": {"type": "integer"},
                        "rank_delta": {"type": "integer"},
                    },
                },
            },
        },
        "truncated": {"type": "boolean"},
    },
    "required": [
        "citation_id",
        "rank",
        "text",
        "score",
        "source",
        "locator",
        "token_estimate",
        "char_count",
        "truncated",
    ],
}

SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA: Final[JsonObject] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "search_user_documents.output",
    "type": "object",
    "additionalProperties": False,
    "definitions": {
        "prompt_source_reference": _PROMPT_SOURCE_REFERENCE_SCHEMA,
        "prompt_source_locator": _PROMPT_SOURCE_LOCATOR_SCHEMA,
        "prompt_source_preview": _PROMPT_SOURCE_PREVIEW_SCHEMA,
        "context_snippet": _CONTEXT_SNIPPET_SCHEMA,
    },
    "properties": {
        "ok": {"type": "boolean", "const": True},
        "query": {"type": "string"},
        "context_text": {"type": "string"},
        "snippets": {
            "type": "array",
            "items": {"$ref": "#/definitions/context_snippet"},
        },
        "citations": {
            "type": "array",
            "items": {"$ref": "#/definitions/prompt_source_reference"},
        },
        "source_previews": {
            "type": "array",
            "items": {"$ref": "#/definitions/prompt_source_preview"},
        },
        "citation_summary": {"type": "string"},
        "dropped_count": {"type": "integer", "minimum": 0},
        "max_snippets": {"type": "integer", "minimum": 1},
        "max_chars": {"type": ["integer", "null"]},
        "max_tokens": {"type": ["integer", "null"]},
        "token_estimate": {"type": "integer", "minimum": 0},
        "char_count": {"type": "integer", "minimum": 0},
        "truncated": {"type": "boolean"},
    },
    "required": [
        "ok",
        "query",
        "context_text",
        "snippets",
        "citations",
        "source_previews",
        "citation_summary",
        "dropped_count",
        "max_snippets",
        "max_chars",
        "max_tokens",
        "token_estimate",
        "char_count",
        "truncated",
    ],
}
