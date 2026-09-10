"""Namespaced Neo4j property graph with fixed, parameterized schema."""

import json
from .graph import GraphStore, GraphNode, GraphEdge
from .records import matches, validate_filter


class Neo4jGraphStore(GraphStore):
    def __init__(self, uri=None, username=None, password=None, *, database="neo4j", namespace="default", driver=None):
        self._owns_driver = driver is None
        if driver is None:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver = driver
        self.database = database
        self.namespace = namespace

    def _run(self, query, **parameters):
        with self.driver.session(database=self.database) as session:
            return [record.data() for record in session.run(query, namespace=self.namespace, **parameters)]

    def put_node(self, node):
        payload = json.dumps(node.properties, ensure_ascii=False, allow_nan=False)
        self._run("MERGE (n:EasyAgentNode {namespace:$namespace,id:$id}) SET n.payload=$payload", id=node.id, payload=payload)

    def put_edge(self, edge):
        payload = json.dumps(edge.properties, ensure_ascii=False, allow_nan=False)
        rows = self._run("MATCH (a:EasyAgentNode {namespace:$namespace,id:$source}), (b:EasyAgentNode {namespace:$namespace,id:$target}) MERGE (a)-[r:EASYAGENT_EDGE {kind:$kind}]->(b) SET r.payload=$payload RETURN r.kind AS kind", source=edge.source, target=edge.target, kind=edge.kind, payload=payload)
        if not rows:
            raise ValueError("Both graph edge endpoints must exist")

    def nodes(self, *, where=None):
        validate_filter(where)
        return [GraphNode(row["id"], properties) for row in self._run("MATCH (n:EasyAgentNode {namespace:$namespace}) RETURN n.id AS id,n.payload AS payload")
                if matches(properties := json.loads(row["payload"]), where)]

    def edges(self, *, where=None):
        validate_filter(where)
        return [GraphEdge(row["source"], row["target"], row["kind"], properties)
                for row in self._run("MATCH (a:EasyAgentNode {namespace:$namespace})-[r:EASYAGENT_EDGE]->(b:EasyAgentNode {namespace:$namespace}) RETURN a.id AS source,b.id AS target,r.kind AS kind,r.payload AS payload")
                if matches(properties := json.loads(row["payload"]), where)]

    def delete_node(self, node_id):
        rows = self._run("MATCH (n:EasyAgentNode {namespace:$namespace,id:$id}) DETACH DELETE n RETURN count(*) AS removed", id=node_id)
        return bool(rows and rows[0]["removed"])

    def delete_edge(self, source, target, kind):
        rows = self._run("MATCH (a:EasyAgentNode {namespace:$namespace,id:$source})-[r:EASYAGENT_EDGE {kind:$kind}]->(b:EasyAgentNode {namespace:$namespace,id:$target}) DELETE r RETURN count(*) AS removed", source=source, target=target, kind=kind)
        return bool(rows and rows[0]["removed"])

    def close(self):
        if self._owns_driver:
            self.driver.close()
