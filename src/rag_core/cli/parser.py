from __future__ import annotations

import argparse

from rag_core.cli.parsers.config import env_or_default
from rag_core.cli.parsers.demo import add_demo_command
from rag_core.cli.parsers.doctor import add_doctor_command
from rag_core.cli.parsers.ingest import add_ingest_command
from rag_core.cli.parsers.eval import add_local_eval_command
from rag_core.cli.parsers.local_search import add_local_search_command
from rag_core.cli.parsers.manifest import add_manifest_command
from rag_core.cli.parsers.mcp import add_mcp_command
from rag_core.cli.parsers.search import add_context_command, add_search_command
from rag_core.cli.parsers.serve import add_serve_command
from rag_core.config.ingest_config import (
    CLI_MANIFEST_DIR_ENV,
    DEFAULT_CLI_MANIFEST_DIRECTORY,
)


class _TopLevelHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _format_action(self, action: argparse.Action) -> str:
        if isinstance(action, argparse._SubParsersAction):
            return ""
        return super()._format_action(action)


_TOP_LEVEL_HELP = """\
Turn a folder into cited context:
  rag-core context "<question>" <folder>

Advanced configured commands:
  add <source>|--url-list <file.txt>
                                Add files, .zip archives, URLs, or URL lists.
  search "<query>" [path]      Search a local path or configured store.
  context "<query>" [path]     Emit prompt-safe context output.
  manifest [--compact] [file]  Preview a manifest entry or compact a collection manifest.
  eval                         Run a local retrieval eval.
  doctor                       Inspect provider configuration.
  demo                         Print deterministic demo output.
  serve                        Run the thin HTTP runtime.
  mcp                          Run the MCP server.

Deprecated aliases:
  ingest -> add
  local-search -> search
  local-eval -> eval
  search --context -> context
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-core",
        description="One no-setup path first; configured commands are advanced.",
        epilog=_TOP_LEVEL_HELP,
        formatter_class=_TopLevelHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    add_doctor_command(subparsers)

    add_demo_command(subparsers)

    add_local_search_command(
        subparsers,
        command_name="local-search",
        deprecated_alias_for="search",
    )

    add_local_eval_command(subparsers)
    add_local_eval_command(
        subparsers,
        command_name="local-eval",
        deprecated_alias_for="eval",
    )

    add_manifest_command(
        subparsers,
        manifest_dir_default=env_or_default(
            CLI_MANIFEST_DIR_ENV, DEFAULT_CLI_MANIFEST_DIRECTORY
        ),
        manifest_dir_help_default=DEFAULT_CLI_MANIFEST_DIRECTORY,
    )

    add_ingest_command(
        subparsers,
        manifest_dir_default=env_or_default(
            CLI_MANIFEST_DIR_ENV, DEFAULT_CLI_MANIFEST_DIRECTORY
        ),
        manifest_dir_help_default=DEFAULT_CLI_MANIFEST_DIRECTORY,
    )
    add_ingest_command(
        subparsers,
        manifest_dir_default=env_or_default(
            CLI_MANIFEST_DIR_ENV, DEFAULT_CLI_MANIFEST_DIRECTORY
        ),
        manifest_dir_help_default=DEFAULT_CLI_MANIFEST_DIRECTORY,
        command_name="ingest",
        deprecated_alias_for="add",
    )

    add_search_command(subparsers)
    add_context_command(subparsers)

    add_serve_command(subparsers)

    add_mcp_command(subparsers)

    return parser
