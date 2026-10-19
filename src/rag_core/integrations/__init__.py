"""Optional standard-protocol adapters. Frameworks call RAGCore directly."""

from __future__ import annotations


def build_mcp_server(*args, **kwargs):
    from .mcp_server import build_mcp_server as build

    return build(*args, **kwargs)


__all__ = ["build_mcp_server"]
