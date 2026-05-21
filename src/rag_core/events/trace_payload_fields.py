from __future__ import annotations

import math
import re
from typing import Literal, Mapping, cast

SearchStageName = Literal[
    "query_transform",
    "retrieve",
    "fuse",
    "rerank",
    "postprocess",
    "context_pack",
]

_SEARCH_STAGE_NAMES = frozenset(
    {
        "query_transform",
        "retrieve",
        "fuse",
        "rerank",
        "postprocess",
        "context_pack",
    }
)

_SAFE_TRACE_LABEL_RE = re.compile(r"[A-Za-z0-9_.:,-]{1,80}")
_SAFE_TRACE_STAGE_LABEL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:]{0,79}")
_SENSITIVE_TRACE_LABEL_FRAGMENTS = (
    "secret",
    "token",
    "password",
    "api_key",
    "apikey",
    "bearer",
)
_SENSITIVE_TRACE_LABEL_PATTERNS = (
    re.compile(r"(?<![A-Z0-9])(?:A3T[A-Z0-9]|AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    re.compile(r"gh[opsur]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)(?:^|[/:])sk-(?!ant-)[A-Za-z0-9][A-Za-z0-9_-]{10,}"),
    re.compile(r"(?i)(?:^|[/:])sk-ant-[A-Za-z0-9][A-Za-z0-9_-]{6,}"),
    re.compile(r"(?i)(?:^|[/:])(?:xox[a-z]{1,3}|xapp)-[A-Za-z0-9-]{6,}"),
    re.compile(r"[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
)


def stage_field(payload: Mapping[str, object], key: str) -> SearchStageName:
    value = str_field(payload, key)
    if value not in _SEARCH_STAGE_NAMES:
        raise ValueError(f"trace field {key} must be a supported search stage")
    return cast(SearchStageName, value)


def str_field(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"trace field {key} must be a string")
    return value


def safe_label_field(payload: Mapping[str, object], key: str) -> str:
    value = str_field(payload, key)
    return safe_trace_label(value, stage=False)


def safe_stage_label_field(payload: Mapping[str, object], key: str) -> str:
    value = str_field(payload, key)
    return safe_trace_label(value, stage=True)


def safe_optional_label_field(payload: Mapping[str, object], key: str) -> str:
    value = str_field(payload, key)
    if not value:
        return ""
    return safe_trace_label(value, stage=False)


def search_id_field(payload: Mapping[str, object], key: str) -> str:
    value = str_field(payload, key)
    if not value:
        return ""
    if safe_trace_label(value, stage=False) != value:
        raise ValueError(f"trace field {key} must be a safe search identifier")
    return value


def int_field(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key, 0)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"trace field {key} must be an integer")
    return value


def float_field(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key, 0.0)
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"trace field {key} must be a number")
    return _finite_float_field(value, key)


def optional_float_field(payload: Mapping[str, object], key: str) -> float | None:
    if key not in payload or payload.get(key) is None:
        return None
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"trace field {key} must be a number")
    return _finite_float_field(value, key)


def bool_field(
    payload: Mapping[str, object],
    key: str,
    *,
    default: bool = False,
) -> bool:
    value = payload.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"trace field {key} must be a boolean")
    return value


def str_tuple_field(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key, ())
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"trace field {key} must be a string array")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"trace field {key} must be a string array")
    return tuple(value)


def safe_label_tuple_field(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    return tuple(
        safe_trace_label(item, stage=False)
        for item in str_tuple_field(payload, key)
    )


def safe_stage_label_tuple_field(
    payload: Mapping[str, object],
    key: str,
) -> tuple[str, ...]:
    return tuple(
        safe_trace_label(item, stage=True)
        for item in str_tuple_field(payload, key)
    )


def int_tuple_field(payload: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = payload.get(key, ())
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"trace field {key} must be an integer array")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ValueError(f"trace field {key} must be an integer array")
    return tuple(value)


def safe_trace_label(value: object, *, stage: bool) -> str:
    if not isinstance(value, str):
        return "unknown"
    if value == "":
        return ""
    normalized = value.lower()
    if (
        any(fragment in normalized for fragment in _SENSITIVE_TRACE_LABEL_FRAGMENTS)
        or any(
            pattern.search(value) is not None
            for pattern in _SENSITIVE_TRACE_LABEL_PATTERNS
        )
    ):
        return "unknown"
    pattern = _SAFE_TRACE_STAGE_LABEL_RE if stage else _SAFE_TRACE_LABEL_RE
    if pattern.fullmatch(value):
        return value
    return "unknown"


def safe_trace_label_sequence(value: object, *, stage: bool) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [safe_trace_label(item, stage=stage) for item in value]


def _finite_float_field(value: int | float, key: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"trace field {key} must be a finite number")
    return number
