from __future__ import annotations

import argparse

from rag_core.cli_events_parser import add_events_jsonl_flag
from rag_core.cli_help_examples import apply_command_examples
from rag_core.cli_profile_help import search_profile_help
from rag_core.local_search_models import DEFAULT_LOCAL_MAX_FILES
from rag_core.search.planning import DEFAULT_SEARCH_PROFILE, SEARCH_PROFILES


def add_local_eval_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    local_eval = subparsers.add_parser(
        "local-eval",
        help="Index a local file/folder and run JSONL retrieval eval cases without API keys.",
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
    apply_command_examples(local_eval, "local-eval")
    return local_eval


__all__ = ["add_local_eval_command"]
