"""Shared structural dispatch for translating the engine Filter AST.

Each vector-store adapter renders the same ``Filter`` AST (Term/In/Range/Geo/
And/Or/Not) into its own backend dialect. ``FilterTranslator`` owns the node
dispatch, recursion, and exhaustiveness so an adapter supplies only the leaf and
combine builders. The node hooks are abstract: a new Filter node type forces
every adapter to handle it instead of silently mis-scoping one backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, NoReturn, TypeVar

from rag_core.search.filters import And, Filter, Geo, In, Not, Or, Range, Term

T = TypeVar("T")


class FilterTranslator(ABC, Generic[T]):
    """Walks a ``Filter`` AST, delegating each node type to an adapter hook."""

    def translate(self, node: Filter) -> T:
        if isinstance(node, Term):
            return self.term(node)
        if isinstance(node, In):
            return self.in_(node)
        if isinstance(node, Range):
            return self.range_(node)
        if isinstance(node, Geo):
            return self.geo(node)
        if isinstance(node, And):
            return self.and_([self.translate(child) for child in node.filters])
        if isinstance(node, Or):
            return self.or_([self.translate(child) for child in node.filters])
        if isinstance(node, Not):
            return self.not_(self.translate(node.filter))
        self.unsupported(node)

    @abstractmethod
    def term(self, node: Term) -> T: ...

    @abstractmethod
    def in_(self, node: In) -> T: ...

    @abstractmethod
    def range_(self, node: Range) -> T: ...

    @abstractmethod
    def geo(self, node: Geo) -> T: ...

    @abstractmethod
    def and_(self, children: list[T]) -> T: ...

    @abstractmethod
    def or_(self, children: list[T]) -> T: ...

    @abstractmethod
    def not_(self, child: T) -> T: ...

    @abstractmethod
    def unsupported(self, node: Filter) -> NoReturn:
        """Reject a node type this backend cannot translate."""
