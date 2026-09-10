"""Vector records and cosine search, usable without an Agent or Memory."""

from abc import ABC, abstractmethod
import asyncio
from copy import deepcopy
import math
from threading import RLock

from .records import VectorRecord, VectorMatch, json_copy, matches, validate_vector, validate_filter, validate_limit


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None: ...

    @abstractmethod
    def search(self, vector, *, where=None, limit=5) -> list[VectorMatch]: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    def scan(self, *, where=None) -> list[VectorRecord]: ...

    def count(self):
        return len(self.scan())

    def clear(self, *, where=None):
        self.delete([record.id for record in self.scan(where=where)])

    async def aupsert(self, records):
        return await asyncio.to_thread(self.upsert, records)

    async def asearch(self, vector, **kwargs):
        return await asyncio.to_thread(self.search, vector, **kwargs)

    async def adelete(self, ids):
        return await asyncio.to_thread(self.delete, ids)

    def close(self):
        pass

    async def aclose(self):
        await asyncio.to_thread(self.close)


class InMemoryVectorStore(VectorStore):
    def __init__(self, *, dimension=None):
        self.dimension = dimension
        self._records = {}
        self._lock = RLock()

    def upsert(self, records):
        records = deepcopy(list(records))
        with self._lock:
            dimension = self.dimension
            for record in records:
                if not record.id:
                    raise ValueError("Record ID must not be empty")
                validate_vector(record.vector, dimension)
                dimension = dimension or len(record.vector)
                record.metadata = json_copy(record.metadata)
            self.dimension = dimension
            self._records.update({record.id: record for record in records})

    def scan(self, *, where=None):
        validate_filter(where)
        with self._lock:
            return deepcopy([record for record in self._records.values() if matches(record.metadata, where)])

    def search(self, vector, *, where=None, limit=5):
        validate_vector(vector, self.dimension)
        validate_limit(limit)
        records = self.scan(where=where)
        norm = math.sqrt(sum(x*x for x in vector))
        if not norm:
            return []
        results = []
        for record in records:
            other = math.sqrt(sum(x*x for x in record.vector))
            if other:
                score = sum(x*y for x,y in zip(vector, record.vector)) / (norm * other)
                results.append(VectorMatch(record.id, score, record.metadata, record.content))
        return sorted(results, key=lambda item: item.score, reverse=True)[:limit]

    def delete(self, ids):
        with self._lock:
            for record_id in ids:
                self._records.pop(record_id, None)
