"""Entity/evidence semantics of layered memory over a generic property graph."""

from storage.graph import GraphNode, GraphEdge, GraphStore
from .V2.Store.GraphStore import Entity, Relation
from .repositories import _json_metadata


class MemoryGraphRepository:
    def __init__(self, store):
        self.store = store

    def add_entity(self, entity_id, name, entity_type, properties):
        self.store.put_node(GraphNode(entity_id, _json_metadata({"name": name, "entity_type": entity_type, "frequency": 1, **properties})))
        return True

    def add_relation(self, from_entity, to_entity, relation_type, properties):
        self.store.put_edge(GraphEdge(from_entity, to_entity, relation_type, _json_metadata({"frequency": 1, **properties})))
        return True

    @staticmethod
    def _entity(node):
        data = dict(node.properties)
        entity = Entity(node.id, data.get("entity_type", ""), data.get("name", ""), data.get("description", ""), data)
        entity.frequency = data.get("frequency", 1)
        return entity

    @staticmethod
    def _relation(edge):
        data = dict(edge.properties)
        relation = Relation(edge.source, edge.target, edge.kind, data.get("strength", 0.5), data.get("evidence", ""), data)
        relation.frequency = data.get("frequency", 1)
        return relation

    def get_entity(self, entity_id):
        return next((self._entity(node) for node in self.store.nodes() if node.id == entity_id), None)

    def get_relation(self, from_entity, to_entity, relation_type):
        return next((self._relation(edge) for edge in self.store.edges() if (edge.source, edge.target, edge.kind) == (from_entity, to_entity, relation_type)), None)

    def update_entity(self, entity):
        self.store.put_node(GraphNode(entity.entity_id, _json_metadata({**entity.properties, "name": entity.name,
            "entity_type": entity.entity_type, "description": entity.description, "frequency": entity.frequency})))
        return True

    def delete_entity(self, entity_id):
        return self.store.delete_node(entity_id)

    def delete_relation(self, from_entity, to_entity, relation_type):
        return self.store.delete_edge(from_entity, to_entity, relation_type)

    def delete_entity_by_memoryid(self, memory_id):
        removed = False
        for node in self.store.nodes(where={"memory_id": memory_id}):
            removed = self.store.delete_node(node.id) or removed
        return removed

    def delete_relation_by_memoryid(self, memory_id):
        removed = False
        for edge in self.store.edges(where={"memory_id": memory_id}):
            removed = self.store.delete_edge(edge.source, edge.target, edge.kind) or removed
        return removed

    def get_all_entities(self):
        return [self._entity(node) for node in self.store.nodes()]

    def get_all_relations(self):
        return [self._relation(edge) for edge in self.store.edges()]

    def _reachable(self, entity_name, user_id, rel_type, max_depth):
        nodes = {node.id: node for node in self.store.nodes(where={"user_id": user_id} if user_id else None)}
        seeds = {node.id for node in nodes.values() if node.properties.get("name") == entity_name}
        visited = set(seeds)
        selected = {}
        edges = self.store.edges(where={"user_id": user_id} if user_id else None)
        for _ in range(max(0, max_depth)):
            frontier = set()
            for edge in edges:
                if rel_type and edge.kind != rel_type:
                    continue
                if edge.source not in nodes or edge.target not in nodes:
                    continue
                if edge.source in visited or edge.target in visited:
                    selected[(edge.source, edge.target, edge.kind)] = edge
                    frontier.update((edge.source, edge.target))
            if frontier <= visited:
                break
            visited.update(frontier)
        return [nodes[node_id] for node_id in visited - seeds], list(selected.values())

    def get_related_entities(self, entity_name, limit=10, user_id=None, rel_type=None, max_depth=2):
        nodes, _ = self._reachable(entity_name, user_id, rel_type, max_depth)
        return [self._entity(node) for node in nodes[:limit]]

    def get_entity_relations(self, entity_name, user_id=None, rel_type=None, max_depth=2, limit=10):
        _, edges = self._reachable(entity_name, user_id, rel_type, max_depth)
        return [self._relation(edge) for edge in edges[:limit]]

    def clear(self):
        for node in self.store.nodes():
            self.store.delete_node(node.id)
        return True

    def get_stats(self):
        return {"entities": len(self.store.nodes()), "relations": len(self.store.edges())}


def graph_repository(store):
    return MemoryGraphRepository(store) if isinstance(store, GraphStore) else store
