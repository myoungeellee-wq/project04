from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chromadb

from embeddings import BgeM3EmbeddingFunction
from rag_config import RagConfig


@dataclass(frozen=True)
class KnowledgeResources:
    """Cached access point for embedding and Chroma resources."""

    config: RagConfig

    @property
    def embedding_function(self) -> BgeM3EmbeddingFunction:
        return get_embedding_function(self.config.embedding_model)

    @property
    def chroma_client(self):
        return get_chroma_client(self.config.chroma_dir)

    def get_collection(self):
        return self.chroma_client.get_collection(
            name=self.config.collection_name,
            embedding_function=self.embedding_function,
        )

    def get_or_create_collection(self):
        return self.chroma_client.get_or_create_collection(
            name=self.config.collection_name,
            embedding_function=self.embedding_function,
        )


@lru_cache(maxsize=8)
def get_chroma_client(chroma_dir: str):
    return chromadb.PersistentClient(path=chroma_dir)


@lru_cache(maxsize=8)
def get_embedding_function(model_name: str) -> BgeM3EmbeddingFunction:
    return BgeM3EmbeddingFunction(model_name)


def get_knowledge_resources(config: RagConfig) -> KnowledgeResources:
    return KnowledgeResources(config=config)
