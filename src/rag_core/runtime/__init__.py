"""Optional self-hostable HTTP runtime (install ``rag-core[runtime]``)."""

from rag_core.runtime.app import create_app

__all__ = ("create_app",)
