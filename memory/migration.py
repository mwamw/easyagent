"""Explicit migration of the built-in layered scheme; never part of Agent startup."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3

def migrate_memory_sqlite(source, target, *, dry_run=True):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("Migration target must differ from the source")
    if not source.is_file():
        raise FileNotFoundError(source)
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        records = list(conn.execute("SELECT * FROM memories"))
    from storage import DocumentRecord
    converted = [DocumentRecord(row["id"], row["content"], {
        "user_id": row["user_id"], "memory_type": row["memory_type"], "timestamp": row["timestamp"],
        "importance": row["importance"], "properties": properties,
        "session_id": properties.get("session_id"),
    }) for row in records for properties in [json.loads(row["properties"] or "{}") or {}]]
    if not dry_run:
        if target.exists():
            raise FileExistsError("Use a new target database")
        from storage import SQLiteDocumentStore
        store = SQLiteDocumentStore(target, namespace="layered-memory")
        for record in converted:
            store.put(record)
    return {"dry_run": dry_run, "count": len(converted), "source_preserved": True, "target": str(target)}


def _legacy_json(value):
    if isinstance(value, dict):
        return {key: _legacy_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_legacy_json(item) for item in value]
    for method in ("isoformat", "iso_format"):
        renderer = getattr(value, method, None)
        if callable(renderer):
            return renderer()
    return value


def migrate_legacy_graph(source_driver, target, *, database="neo4j", dry_run=True):
    """Copy legacy entity_id nodes to an empty generic graph namespace.

    The supplied source driver is borrowed. No source writes or deletions occur.
    Cross-server graph writes are not a distributed transaction; on failure keep
    the old graph and inspect the newly created namespace before retrying.
    """
    from storage.graph import GraphNode, GraphEdge
    with source_driver.session(database=database) as session:
        nodes = [GraphNode(str(row["id"]), _legacy_json(dict(row["properties"]))) for row in session.run(
            "MATCH (n) WHERE n.entity_id IS NOT NULL RETURN n.entity_id AS id,properties(n) AS properties")]
        edges = [GraphEdge(str(row["source"]), str(row["target"]), row["kind"], _legacy_json(dict(row["properties"]))) for row in session.run(
            "MATCH (a)-[r]->(b) WHERE a.entity_id IS NOT NULL AND b.entity_id IS NOT NULL RETURN a.entity_id AS source,b.entity_id AS target,type(r) AS kind,properties(r) AS properties")]
    if len({node.id for node in nodes}) != len(nodes):
        raise ValueError("Legacy graph contains duplicate entity IDs; reconcile before migrating")
    json.dumps([node.properties for node in nodes] + [edge.properties for edge in edges], allow_nan=False)
    if not dry_run:
        if target.nodes() or target.edges():
            raise ValueError("Graph migration requires an empty target namespace")
        for node in nodes:
            target.put_node(node)
        for edge in edges:
            target.put_edge(edge)
    return {"dry_run": dry_run, "nodes": len(nodes), "edges": len(edges), "source_preserved": True}
