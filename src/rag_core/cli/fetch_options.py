from __future__ import annotations

import argparse

from rag_core.config.env_access import (
    get_env_bool_strict,
    get_env_float_strict,
    get_env_int_strict,
)
from rag_core.fetch_security import (
    DEFAULT_FETCH_MAX_BYTES,
    DEFAULT_FETCH_MAX_REDIRECTS,
    DEFAULT_FETCH_TIMEOUT_SECONDS,
    FETCH_ALLOW_HTTP_ENV,
    FETCH_ALLOW_PRIVATE_ADDRESSES_ENV,
    FETCH_MAX_BYTES_ENV,
    FETCH_MAX_REDIRECTS_ENV,
    FETCH_TIMEOUT_SECONDS_ENV,
    FetchLimits,
    FetchScheme,
    FetchSecurityPolicy,
)


def fetch_policy_from_args(args: argparse.Namespace) -> FetchSecurityPolicy:
    allow_http = _arg_or_env_bool(
        args,
        "fetch_allow_http",
        env_name=FETCH_ALLOW_HTTP_ENV,
        default=False,
    )
    allowed_schemes: tuple[FetchScheme, ...] = (
        ("https", "http") if allow_http else ("https",)
    )
    return FetchSecurityPolicy(
        allowed_schemes=allowed_schemes,
        allow_private_addresses=_arg_or_env_bool(
            args,
            "fetch_allow_private_addresses",
            env_name=FETCH_ALLOW_PRIVATE_ADDRESSES_ENV,
            default=False,
        ),
    )


def fetch_limits_from_args(args: argparse.Namespace) -> FetchLimits:
    return FetchLimits(
        max_bytes=_arg_or_env_int(
            args,
            "fetch_max_bytes",
            env_name=FETCH_MAX_BYTES_ENV,
            default=DEFAULT_FETCH_MAX_BYTES,
        ),
        timeout_seconds=_arg_or_env_float(
            args,
            "fetch_timeout_seconds",
            env_name=FETCH_TIMEOUT_SECONDS_ENV,
            default=DEFAULT_FETCH_TIMEOUT_SECONDS,
        ),
        max_redirects=_arg_or_env_int(
            args,
            "fetch_max_redirects",
            env_name=FETCH_MAX_REDIRECTS_ENV,
            default=DEFAULT_FETCH_MAX_REDIRECTS,
        ),
    )


def _arg_or_env_bool(
    args: argparse.Namespace,
    name: str,
    *,
    env_name: str,
    default: bool,
) -> bool:
    value = getattr(args, name)
    if value is not None:
        return bool(value)
    return get_env_bool_strict(env_name, default)


def _arg_or_env_int(
    args: argparse.Namespace,
    name: str,
    *,
    env_name: str,
    default: int,
) -> int:
    value = getattr(args, name)
    if value is not None:
        return int(value)
    return get_env_int_strict(env_name, default)


def _arg_or_env_float(
    args: argparse.Namespace,
    name: str,
    *,
    env_name: str,
    default: float,
) -> float:
    value = getattr(args, name)
    if value is not None:
        return float(value)
    return get_env_float_strict(env_name, default)
