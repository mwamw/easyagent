"""Domain-neutral Qdrant backend. SDK imports occur only on use."""

from uuid import UUID, uuid5, NAMESPACE_URL

from .vector import VectorStore
from .records import VectorRecord, VectorMatch, json_copy, validate_vector, validate_filter, validate_limit


class QdrantVectorStore(VectorStore):
    def __init__(self, collection_name="easyagent", *, dimension=384, client=None, **client_options):
        from qdrant_client import QdrantClient, models
        self.client = client or QdrantClient(**(client_options or {"location": ":memory:"}))
        self._owns_client = client is None
        self.collection_name = collection_name
        self.dimension = dimension
        existing = {item.name for item in self.client.get_collections().collections}
        if collection_name not in existing:
            self.client.create_collection(collection_name, vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE))
        else:
            config = self.client.get_collection(collection_name).config.params.vectors
            if getattr(config, "size", None) != dimension or getattr(config, "distance", None) != models.Distance.COSINE:
                raise ValueError("Qdrant collection must match configured dimension and cosine metric")

    @staticmethod
    def _id(value):
        try:
            return str(UUID(str(value)))
        except ValueError:
            return str(uuid5(NAMESPACE_URL, str(value)))

    @staticmethod
    def _filter(where):
        from qdrant_client import models
        validate_filter(where)
        conditions = []
        for key, value in (where or {}).items():
            operations = value if isinstance(value, dict) else {"$eq": value}
            for op, expected in operations.items():
                if op == "$eq":
                    if isinstance(expected, float):
                        condition = models.FieldCondition(key=key, range=models.Range(gte=expected, lte=expected))
                    else:
                        condition = models.FieldCondition(key=key, match=models.MatchValue(value=expected))
                elif op == "$in":
                    condition = models.FieldCondition(key=key, match=models.MatchAny(any=expected))
                else:
                    condition = models.FieldCondition(key=key, range=models.Range(**{op[1:]: expected}))
                conditions.append(condition)
        return models.Filter(must=conditions) if conditions else None

    def upsert(self, records):
        from qdrant_client import models
        points = []
        for record in records:
            validate_vector(record.vector, self.dimension)
            if not record.id or any(key.startswith("_ea_") for key in record.metadata):
                raise ValueError("Record IDs must be non-empty; _ea_ metadata keys are reserved")
            payload = json_copy(record.metadata)
            payload.update(_ea_id=record.id, _ea_content=record.content)
            points.append(models.PointStruct(id=self._id(record.id), vector=record.vector, payload=payload))
        if points:
            self.client.upsert(self.collection_name, points=points, wait=True)

    @staticmethod
    def _payload(point):
        metadata = dict(point.payload or {})
        return str(metadata.pop("_ea_id", point.id)), metadata.pop("_ea_content", ""), metadata

    def search(self, vector, *, where=None, limit=5):
        validate_vector(vector, self.dimension)
        validate_limit(limit)
        filter_ = self._filter(where)
        if not limit:
            return []
        points = self.client.query_points(self.collection_name, query=vector, query_filter=filter_, limit=limit, with_payload=True).points
        result = []
        for point in points:
            record_id, content, metadata = self._payload(point)
            result.append(VectorMatch(record_id, float(point.score), metadata, content))
        return result

    def scan(self, *, where=None):
        filter_ = self._filter(where)
        offset = None
        result = []
        while True:
            points, offset = self.client.scroll(self.collection_name, scroll_filter=filter_, offset=offset, limit=1000, with_payload=True, with_vectors=True)
            for point in points:
                record_id, content, metadata = self._payload(point)
                result.append(VectorRecord(record_id, list(point.vector), metadata, content))
            if offset is None:
                return result

    def delete(self, ids):
        from qdrant_client import models
        if ids:
            self.client.delete(self.collection_name, points_selector=models.PointIdsList(points=[self._id(i) for i in ids]), wait=True)

    def close(self):
        if self._owns_client:
            self.client.close()
