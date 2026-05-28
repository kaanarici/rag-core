from __future__ import annotations

import argparse

from rag_core.cli_archive_parser import add_ingest_archive_command
from rag_core.cli_config_parser import env_or_default
from rag_core.cli_demo_parser import add_demo_command
from rag_core.cli_doctor_parser import add_doctor_command
from rag_core.cli_ingest_parser import add_ingest_command
from rag_core.cli_local_eval_parser import add_local_eval_command
from rag_core.cli_local_search_parser import add_local_search_command
from rag_core.cli_manifest_parser import add_manifest_commands
from rag_core.cli_search_parser import add_search_command
from rag_core.cli_remote_parser import add_remote_commands
from rag_core.cli_serve_parser import add_serve_command
from rag_core.config.ingest_config import (
    CLI_MANIFEST_DIR_ENV,
    DEFAULT_CLI_MANIFEST_DIRECTORY,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-core")
    subparsers = parser.add_subparsers(dest="command")

    add_doctor_command(subparsers)

    add_demo_command(subparsers)

    add_local_search_command(subparsers)

    add_local_eval_command(subparsers)

    add_manifest_commands(
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

    add_ingest_archive_command(
        subparsers,
        manifest_dir_default=env_or_default(
            CLI_MANIFEST_DIR_ENV, DEFAULT_CLI_MANIFEST_DIRECTORY
        ),
        manifest_dir_help_default=DEFAULT_CLI_MANIFEST_DIRECTORY,
    )

    add_remote_commands(
        subparsers,
        manifest_dir_default=env_or_default(
            CLI_MANIFEST_DIR_ENV, DEFAULT_CLI_MANIFEST_DIRECTORY
        ),
        manifest_dir_help_default=DEFAULT_CLI_MANIFEST_DIRECTORY,
    )

    add_search_command(subparsers)

    add_serve_command(subparsers)

    return parser
