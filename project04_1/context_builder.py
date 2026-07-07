from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from knowledge_resources import get_knowledge_resources
from rag_config import RagConfig, config_for_embedding_model


@dataclass(frozen=True)
class ContextBundle:
    question: str
    contexts: list[dict[str, Any]]
    context_text: str
    retrieve_seconds: float


class ContextBuilder:
    """Builds RAG context from the primary and optional auxiliary Chroma collections."""

    def __init__(self, config: RagConfig):
        self.config = config

    def retrieve_one_model(self, config: RagConfig, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        resources = get_knowledge_resources(config)
        embedding = resources.embedding_function
        collection = resources.get_collection()
        result = collection.query(query_embeddings=[embedding.embed_query(question)], n_results=top_k)

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        items: list[dict[str, Any]] = []
        for idx, document in enumerate(documents):
            items.append(
                {
                    "id": ids[idx] if idx < len(ids) else "",
                    "document": document,
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "distance": distances[idx] if idx < len(distances) else None,
                    "model": config.embedding_model,
                    "collection": config.collection_name,
                }
            )
        return items

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        auxiliary_models: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        model_names = [self.config.embedding_model]
        if auxiliary_models:
            model_names.extend(auxiliary_models)
        model_names = list(dict.fromkeys(model_names))

        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for model_name in model_names:
            model_config = config_for_embedding_model(self.config, model_name)
            try:
                items = self.retrieve_one_model(model_config, question, top_k=top_k)
            except Exception:
                continue
            for item in items:
                key = (item.get("id", ""), item.get("model", ""))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        return merged

    def build(
        self,
        question: str,
        top_k: int = 5,
        auxiliary_models: list[str] | None = None,
    ) -> ContextBundle:
        started_at = time.perf_counter()
        contexts = self.retrieve(question, top_k=top_k, auxiliary_models=auxiliary_models)
        context_text = "\n\n---\n\n".join(
            f"[Search result {idx + 1}]\n{item['document']}" for idx, item in enumerate(contexts)
        )
        return ContextBundle(
            question=question,
            contexts=contexts,
            context_text=context_text,
            retrieve_seconds=time.perf_counter() - started_at,
        )
