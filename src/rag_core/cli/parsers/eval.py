from __future__ import annotations

import argparse
from typing import Literal

from rag_core.cli.parsers.events import add_events_jsonl_flag
from rag_core.cli.help_examples import apply_command_examples
from rag_core.cli.profile_help import search_profile_help
from rag_core.local_search.models import DEFAULT_LOCAL_MAX_FILES
from rag_core.search.planning import DEFAULT_SEARCH_PROFILE, SEARCH_PROFILES


def add_local_eval_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    command_name: Literal["eval", "local-eval"] = "eval",
    deprecated_alias_for: str | None = None,
) -> argparse.ArgumentParser:
    local_eval = subparsers.add_parser(
        command_name,
        help=(
            argparse.SUPPRESS
            if deprecated_alias_for is not None
            else "Index a local file/folder and run JSONL retrieval eval cases without API keys."
        ),
    )
    if deprecated_alias_for is not None:
        local_eval.set_defaults(
            deprecated_command=command_name,
            canonical_command=deprecated_alias_for,
        )
    local_eval.add_argument("path", help="Local file or folder to index.")
    local_eval.add_argument("cases", help="JSONL eval cases to run.")
    local_eval.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_LOCAL_MAX_FILES,
        help="Maximum supported files to index from a folder.",
    )
    local_eval.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum eval cases to search concurrently. Default: 1.",
    )
    local_eval.add_argument(
        "--search-profile",
        choices=SEARCH_PROFILES,
        default=DEFAULT_SEARCH_PROFILE,
        help=search_profile_help(
            prefix="Common search profile.",
            suffix="Defaults to the normal balanced retrieval profile.",
        ),
    )
    add_events_jsonl_flag(local_eval)
    local_eval.add_argument("--json", action="store_true", help="Emit JSON output.")
    apply_command_examples(local_eval, command_name)
    return local_eval


__all__ = ["add_local_eval_command"]
