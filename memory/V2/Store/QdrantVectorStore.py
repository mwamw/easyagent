"""Layered-memory index over the common Qdrant backend."""
from storage.qdrant import QdrantVectorStore as QdrantBackend
from memory.repositories import MemoryVectorIndex


class QdrantVectorStore(MemoryVectorIndex):
    def __init__(self, way="memory", collection_name="layered-memory", host=None, port=None,
                 api_key=None, vector_size=384, distance=None, timeout=10):
        options = {"timeout": timeout}
        if way == "memory":
            options["location"] = ":memory:"
        elif way == "cloud":
            options.update(url=host, api_key=api_key)
        elif way == "local":
            options.update(host=host, port=port)
        else:
            raise ValueError("way must be memory, local, or cloud")
        if distance is not None and str(getattr(distance, "value", distance)).lower() != "cosine":
            raise ValueError("The shared memory vector index uses cosine distance")
        super().__init__(QdrantBackend(collection_name, dimension=vector_size, **options))

    def close(self):
        self.store.close()
