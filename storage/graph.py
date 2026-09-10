"""Property-graph contracts, independent of memory entities and evidence rules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphNode:
    id: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    kind: str
    properties: dict[str, Any] = field(default_factory=dict)


class GraphStore(ABC):
    @abstractmethod
    def put_node(self, node: GraphNode): ...

    @abstractmethod
    def put_edge(self, edge: GraphEdge): ...

    @abstractmethod
    def nodes(self, *, where=None) -> list[GraphNode]: ...

    @abstractmethod
    def edges(self, *, where=None) -> list[GraphEdge]: ...

    @abstractmethod
    def delete_node(self, node_id): ...

    @abstractmethod
    def delete_edge(self, source, target, kind): ...

    def close(self):
        pass
