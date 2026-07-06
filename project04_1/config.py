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
            return self.__call__(input)
        return self.__call__([input])[0]

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            input,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
