"""Shared query/document embedding contract with explicit async support."""

from abc import ABC, abstractmethod
import asyncio


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, texts): ...

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    @property
    @abstractmethod
    def dimension(self): ...

    async def aembed_documents(self, texts):
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text):
        return await asyncio.to_thread(self.embed_query, text)

    def close(self):
        pass

    async def aclose(self):
        await asyncio.to_thread(self.close)
