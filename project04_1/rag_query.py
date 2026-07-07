from __future__ import annotations

import time
from typing import Any

from embeddings import BgeM3EmbeddingFunction
from rag_config import RagConfig, config_for_embedding_model


def get_existing_collection(config: RagConfig):
    import chromadb

    client = chromadb.PersistentClient(path=config.chroma_dir)
    return client.get_collection(
        name=config.collection_name,
        embedding_function=BgeM3EmbeddingFunction(config.embedding_model),
    )


def retrieve(config: RagConfig, question: str, top_k: int = 5) -> list[dict[str, Any]]:
    embedding_function = BgeM3EmbeddingFunction(config.embedding_model)
    collection = get_existing_collection(config)
    result = collection.query(query_embeddings=[embedding_function.embed_query(question)], n_results=top_k)

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


def retrieve_with_auxiliary_models(
    config: RagConfig,
    question: str,
    top_k: int = 5,
    auxiliary_models: list[str] | None = None,
) -> list[dict[str, Any]]:
    model_names = [config.embedding_model]
    if auxiliary_models:
        model_names.extend(auxiliary_models)
    model_names = list(dict.fromkeys(model_names))

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for model_name in model_names:
        model_config = config_for_embedding_model(config, model_name)
        try:
            items = retrieve(model_config, question, top_k=top_k)
        except Exception:
            continue
        for item in items:
            key = (item.get("id", ""), item.get("model", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def answer_question(
    config: RagConfig,
    question: str,
    top_k: int = 5,
    auxiliary_models: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, float]]:
    from report_runner import ReportRunner

    return ReportRunner(config).answer(
        question=question,
        top_k=top_k,
        auxiliary_models=auxiliary_models,
    )

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama

    started_at = time.perf_counter()
    retrieve_started_at = time.perf_counter()
    contexts = retrieve_with_auxiliary_models(
        config,
        question,
        top_k=top_k,
        auxiliary_models=auxiliary_models,
    )
    retrieve_seconds = time.perf_counter() - retrieve_started_at
    context_text = "\n\n---\n\n".join(
        f"[검색결과 {idx + 1}]\n{item['document']}" for idx, item in enumerate(contexts)
    )

    system_prompt = (
        "너는 서울시 부동산 실거래가 CSV를 기반으로 답하는 RAG 어시스턴트다. "
        "반드시 제공된 검색결과 안의 사실만 사용하고, 모르는 내용은 모른다고 말한다. "
        "금액은 만원과 억원 단위를 함께 설명하고, 면적당금액이 있으면 만원/㎡와 만원/평 기준을 함께 활용한다. "
        "비교/요약 질문에는 근거 거래를 간단히 인용한다."
    )
    user_prompt = f"질문:\n{question}\n\n검색결과:\n{context_text}\n\n답변:"

    generation_started_at = time.perf_counter()
    llm = ChatOllama(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        temperature=config.ollama_temperature,
    )
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    generation_seconds = time.perf_counter() - generation_started_at
    total_seconds = time.perf_counter() - started_at
    timings = {
        "retrieve_seconds": retrieve_seconds,
        "generation_seconds": generation_seconds,
        "total_seconds": total_seconds,
    }
    return response.content, contexts, timings
