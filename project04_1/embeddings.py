from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class BgeM3EmbeddingFunction:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = load_embedding_model(model_name)

    def name(self) -> str:
        return f"sentence-transformers-{self.model_name}"

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def embed_query(self, input: str | list[str]):
        if isinstance(input, list):
            return self._encode(input, query=True)
        return self._encode([input], query=True)[0]

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._encode(input, query=False)

    def _prefix(self, text: str, query: bool) -> str:
        if "multilingual-e5" in self.model_name.lower():
            return f"{'query' if query else 'passage'}: {text}"
        return text

    def _encode(self, input: list[str], query: bool) -> list[list[float]]:
        texts = [self._prefix(str(text), query=query) for text in input]
        embeddings = self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
