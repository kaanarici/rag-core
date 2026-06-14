from __future__ import annotations

import argparse

from rag_core.cli_config_parser import env_or_default
from rag_core.cli_demo_parser import add_demo_command
from rag_core.cli_doctor_parser import add_doctor_command
from rag_core.cli_ingest_parser import add_ingest_command
from rag_core.cli_local_eval_parser import add_local_eval_command
from rag_core.cli_local_search_parser import add_local_search_command
from rag_core.cli_manifest_parser import add_manifest_command
from rag_core.cli_mcp_parser import add_mcp_command
from rag_core.cli_search_parser import add_search_command
from rag_core.cli_serve_parser import add_serve_command
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
  rag-core local-search <folder> "<question>"

Advanced configured commands:
  ingest <source>|--url-list <file.txt>
                                Ingest files, .zip archives, URLs, or URL lists.
  search [--context] "<query>"  Search, or emit a prompt-safe context-pack JSON.
  manifest [--compact] [file]  Preview a manifest entry or compact a corpus manifest.
  local-eval                   Run a local retrieval eval.
  doctor                       Inspect provider configuration.
  demo                         Print deterministic demo output.
  serve                        Run the thin HTTP runtime.
  mcp                          Run the MCP server.
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

    add_local_search_command(subparsers)

    add_local_eval_command(subparsers)

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

    add_search_command(subparsers)

    add_serve_command(subparsers)

    add_mcp_command(subparsers)

    return parser
