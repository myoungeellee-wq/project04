from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import streamlit as st
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class RagConfig:
    chroma_dir: str
    collection_name: str
    embedding_model: str
    ollama_model: str


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class BgeM3EmbeddingFunction:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = load_embedding_model(model_name)

    def name(self) -> str:
        return f"sentence-transformers-{self.model_name}"

    def embed_documents(self, texts: list[str] | None = None, **kwargs) -> list[list[float]]:
        values = texts if texts is not None else kwargs.get("input", [])
        return self._encode(self._as_text_list(values))

    def embed_query(self, text: str | list[str] | None = None, **kwargs):
        value = text if text is not None else kwargs.get("input", "")
        values = self._as_text_list(value)
        if not values:
            values = [""]
        embeddings = self._encode(values)
        return embeddings if isinstance(value, (list, tuple)) else embeddings[0]

    def __call__(self, input: list[str] | None = None, **kwargs) -> list[list[float]]:
        values = input if input is not None else kwargs.get("texts", [])
        return self._encode(self._as_text_list(values))

    def _as_text_list(self, values: str | list[str] | tuple[str, ...]) -> list[str]:
        if isinstance(values, str):
            return [values]
        return [str(value) for value in values]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


def get_existing_collection(config: RagConfig):
    import chromadb

    client = chromadb.PersistentClient(path=config.chroma_dir)
    embedding_function = BgeM3EmbeddingFunction(config.embedding_model)
    return client.get_collection(
        name=config.collection_name,
        embedding_function=embedding_function,
    )


def retrieve(config: RagConfig, question: str, top_k: int) -> list[dict[str, Any]]:
    collection = get_existing_collection(config)
    result = collection.query(query_texts=[question], n_results=top_k)

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
            }
        )
    return items


def answer_question(config: RagConfig, question: str, top_k: int) -> tuple[str, list[dict[str, Any]]]:
    import ollama

    contexts = retrieve(config, question, top_k)
    context_text = "\n\n---\n\n".join(
        f"[검색결과 {idx + 1}]\n{item['document']}" for idx, item in enumerate(contexts)
    )

    system_prompt = (
        "너는 서울시 부동산 실거래가 CSV를 기반으로 답하는 RAG 어시스턴트다. "
        "반드시 제공된 검색결과 안의 사실만 사용하고, 모르는 내용은 모른다고 말한다. "
        "금액은 만원과 억원 단위를 함께 설명하고, 비교/요약 질문에는 근거 거래를 간단히 인용한다."
    )
    user_prompt = f"질문:\n{question}\n\n검색결과:\n{context_text}\n\n답변:"

    response = ollama.chat(
        model=config.ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"], contexts


def build_config() -> RagConfig:
    return RagConfig(
        chroma_dir=st.session_state.chroma_dir.strip() or "./chroma_db",
        collection_name=st.session_state.collection_name.strip() or "seoul_real_estate_202606",
        embedding_model=st.session_state.embedding_model.strip() or "BAAI/bge-m3",
        ollama_model=st.session_state.ollama_model.strip() or "qwen2.5:7b",
    )


def show_source(idx: int, source: dict[str, Any]) -> None:
    metadata = source["metadata"] or {}
    title = (
        f"{idx}. {metadata.get('contract_date', '날짜 미상')} | "
        f"{metadata.get('gu', '')} {metadata.get('dong', '')} | "
        f"{metadata.get('building') or metadata.get('building_use') or source.get('id') or '검색 결과'}"
    )
    with st.expander(title):
        distance = source.get("distance")
        if distance is not None:
            st.caption(f"distance: {distance:.4f}")
        st.json(metadata)
        st.text(source["document"])


def app() -> None:
    st.set_page_config(page_title="서울시 부동산 RAG 질문", layout="wide")

    st.title("서울시 부동산 실거래가 RAG 질문")
    st.write("이미 인덱싱된 ChromaDB 컬렉션을 사용해 질문 결과만 가져옵니다.")

    with st.sidebar:
        st.header("기존 인덱스 설정")
        st.text_input("ChromaDB 저장 폴더", value=os.getenv("CHROMA_DIR", "./chroma_db"), key="chroma_dir")
        st.text_input(
            "컬렉션 이름",
            value=os.getenv("COLLECTION_NAME", "seoul_real_estate_202606"),
            key="collection_name",
        )
        st.text_input("임베딩 모델", value=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"), key="embedding_model")
        st.text_input("Ollama 모델", value=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"), key="ollama_model")
        top_k = st.slider("검색 문서 수", min_value=1, max_value=15, value=5)

        if st.button("컬렉션 확인", use_container_width=True):
            try:
                collection = get_existing_collection(build_config())
                st.success(f"문서 수: {collection.count():,}건")
            except Exception as exc:
                st.error(f"컬렉션 확인 실패: {exc}")

    question = st.text_area(
        "질문 입력",
        placeholder="예: 2026년 6월 서초구 오피스텔 거래 중 3억원대 사례를 알려줘",
        height=140,
    )

    col_answer, col_clear = st.columns([1, 1])
    with col_answer:
        ask_clicked = st.button("질문하기", type="primary", use_container_width=True)
    with col_clear:
        if st.button("결과 지우기", use_container_width=True):
            st.session_state.pop("last_answer", None)
            st.session_state.pop("last_sources", None)

    if ask_clicked:
        if not question.strip():
            st.warning("질문을 입력해주세요.")
            return

        config = build_config()
        try:
            with st.spinner("기존 ChromaDB에서 검색하고 Ollama로 답변을 생성하는 중입니다..."):
                answer, sources = answer_question(config, question.strip(), int(top_k))
            st.session_state.last_answer = answer
            st.session_state.last_sources = sources
        except Exception as exc:
            st.error(f"답변 생성 실패: {exc}")
            st.info("ChromaDB 폴더와 컬렉션 이름이 기존 인덱스와 같은지 확인해주세요.")

    if "last_answer" in st.session_state:
        st.subheader("답변")
        st.write(st.session_state.last_answer)

    if "last_sources" in st.session_state:
        st.subheader("검색 근거")
        for idx, source in enumerate(st.session_state.last_sources, start=1):
            show_source(idx, source)


app()
