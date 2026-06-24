from __future__ import annotations

import argparse

from rag_core.fetch_security import (
    DEFAULT_FETCH_MAX_BYTES,
    DEFAULT_FETCH_MAX_REDIRECTS,
    DEFAULT_FETCH_TIMEOUT_SECONDS,
)


def add_fetch_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fetch-max-bytes",
        type=int,
        default=None,
        help=f"Maximum bytes to download per fetched URL. Default: {DEFAULT_FETCH_MAX_BYTES}.",
    )
    parser.add_argument(
        "--fetch-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Socket timeout for each fetched URL in seconds. "
            f"Default: {DEFAULT_FETCH_TIMEOUT_SECONDS}."
        ),
    )
    parser.add_argument(
        "--fetch-max-redirects",
        type=int,
        default=None,
        help=f"Maximum redirects to follow per fetched URL. Default: {DEFAULT_FETCH_MAX_REDIRECTS}.",
    )
    parser.add_argument(
        "--fetch-allow-http",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow explicit URL fetches over plaintext HTTP. Default: false.",
    )
    parser.add_argument(
        "--fetch-allow-private-addresses",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Allow explicit URL fetches to local, private, or link-local addresses. "
            "Default: false."
        ),
    )


__all__ = ["add_fetch_flags"]
