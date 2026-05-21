from __future__ import annotations

import argparse

from rag_core.cli_archive_parser import add_ingest_archive_command
from rag_core.cli_config_parser import env_or_default
from rag_core.cli_demo_parser import add_demo_command
from rag_core.cli_doctor_parser import add_doctor_command
from rag_core.cli_eval_parser import add_eval_command
from rag_core.cli_ingest_parser import add_ingest_command
from rag_core.cli_local_search_parser import add_local_search_command
from rag_core.cli_manifest_parser import add_manifest_commands
from rag_core.cli_query_parser import add_query_command
from rag_core.cli_remote_parser import add_remote_commands
from rag_core.cli_trace_parser import add_trace_summary_command

DEFAULT_MANIFEST_DIRECTORY = ".rag-core/manifest"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-core")
    subparsers = parser.add_subparsers(dest="command")

    add_doctor_command(subparsers)

    add_demo_command(subparsers)

    add_local_search_command(subparsers)

    add_manifest_commands(
        subparsers,
        manifest_dir_default=env_or_default(
            "RAG_CORE_MANIFEST_DIR", DEFAULT_MANIFEST_DIRECTORY
        ),
        manifest_dir_help_default=DEFAULT_MANIFEST_DIRECTORY,
    )

    add_ingest_command(
        subparsers,
        manifest_dir_default=env_or_default(
            "RAG_CORE_MANIFEST_DIR", DEFAULT_MANIFEST_DIRECTORY
        ),
        manifest_dir_help_default=DEFAULT_MANIFEST_DIRECTORY,
    )

    add_ingest_archive_command(
        subparsers,
        manifest_dir_default=env_or_default(
            "RAG_CORE_MANIFEST_DIR", DEFAULT_MANIFEST_DIRECTORY
        ),
        manifest_dir_help_default=DEFAULT_MANIFEST_DIRECTORY,
    )

    add_remote_commands(
        subparsers,
        manifest_dir_default=env_or_default(
            "RAG_CORE_MANIFEST_DIR", DEFAULT_MANIFEST_DIRECTORY
        ),
        manifest_dir_help_default=DEFAULT_MANIFEST_DIRECTORY,
    )

    add_query_command(subparsers)

    add_trace_summary_command(subparsers)

    add_eval_command(subparsers)

    return parser
