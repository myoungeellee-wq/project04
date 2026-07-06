from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import ollama
import pandas as pd
from chromadb.api.types import EmbeddingFunction
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


load_dotenv()


DEFAULT_CSV_PATH = r"D:\AI_학습\AI_test\dataset\서울시 부동산 실거래가 정보_202606.csv"


@dataclass(frozen=True)
class RagConfig:
    csv_path: str = os.getenv("CSV_PATH", DEFAULT_CSV_PATH)
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "seoul_real_estate_202606")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")



class BgeM3EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            input,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


def get_collection(config: RagConfig):
    client = chromadb.PersistentClient(path=config.chroma_dir)
    embedding_function = BgeM3EmbeddingFunction(config.embedding_model)
    return client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

def _clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _number(value: Any) -> float | None:
    text = _clean(value).replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _format_date(value: Any) -> str:
    text = _clean(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _format_price(amount_manwon: float | None) -> str:
    if amount_manwon is None:
        return "미상"
    eok = amount_manwon / 10000
    return f"{amount_manwon:,.0f}만원 ({eok:,.2f}억원)"


def row_to_document(row: pd.Series) -> tuple[str, dict[str, str | int | float]]:
    amount = _number(row.get("물건금액(만원)"))
    building_area = _number(row.get("건물면적(㎡)"))
    land_area = _number(row.get("토지면적(㎡)"))
    contract_date = _format_date(row.get("계약일"))

    gu = _clean(row.get("자치구명"))
    dong = _clean(row.get("법정동명"))
    main_no = _clean(row.get("본번")).lstrip("0")
    sub_no = _clean(row.get("부번")).lstrip("0")
    lot_no = "-".join(part for part in [main_no, sub_no] if part)
    building = _clean(row.get("건물명"))
    building_use = _clean(row.get("건물용도"))

    location_parts = [part for part in [gu, dong, lot_no, building] if part]
    location = " ".join(location_parts) if location_parts else "위치 미상"

    text = "\n".join(
        [
            "서울시 부동산 실거래가 기록",
            f"계약일: {contract_date}",
            f"위치: {location}",
            f"자치구: {gu}",
            f"법정동: {dong}",
            f"건물명: {building or '미상'}",
            f"건물용도: {building_use or '미상'}",
            f"거래금액: {_format_price(amount)}",
            f"건물면적: {building_area if building_area is not None else '미상'}㎡",
            f"토지면적: {land_area if land_area is not None else '미상'}㎡",
            f"층: {_clean(row.get('층')) or '미상'}",
            f"건축년도: {_clean(row.get('건축년도')) or '미상'}",
            f"거래유형: {_clean(row.get('신고구분')) or '미상'}",
            f"중개사 소재지: {_clean(row.get('신고한 개업공인중개사 시군구명')) or '미상'}",
        ]
    )

    metadata: dict[str, str | int | float] = {
        "contract_date": contract_date,
        "gu": gu,
        "dong": dong,
        "building": building,
        "building_use": building_use,
        "deal_type": _clean(row.get("신고구분")),
        "floor": _clean(row.get("층")),
        "built_year": _clean(row.get("건축년도")),
    }
    if amount is not None:
        metadata["amount_manwon"] = amount
    if building_area is not None:
        metadata["building_area_m2"] = building_area
    if land_area is not None:
        metadata["land_area_m2"] = land_area

    return text, metadata


def load_csv_documents(csv_path: str) -> tuple[list[str], list[dict[str, str | int | float]], list[str]]:
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float]] = []
    ids: list[str] = []

    for idx, row in df.iterrows():
        document, metadata = row_to_document(row)
        metadata["row_index"] = int(idx)
        documents.append(document)
        metadatas.append(metadata)
        ids.append(f"deal-{idx}")

    return documents, metadatas, ids


def ingest_csv(config: RagConfig, batch_size: int = 128, reset: bool = True) -> int:
    csv_path = Path(config.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    client = chromadb.PersistentClient(path=config.chroma_dir)
    if reset:
        try:
            client.delete_collection(config.collection_name)
        except ValueError:
            pass

    collection = get_collection(config)
    documents, metadatas, ids = load_csv_documents(str(csv_path))

    for start in tqdm(range(0, len(documents), batch_size), desc="Indexing"):
        end = start + batch_size
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    return len(documents)


def retrieve(config: RagConfig, question: str, top_k: int = 5) -> list[dict[str, Any]]:
    collection = get_collection(config)
    result = collection.query(query_texts=[question], n_results=top_k)

    items: list[dict[str, Any]] = []
    for idx, doc in enumerate(result.get("documents", [[]])[0]):
        items.append(
            {
                "document": doc,
                "metadata": result.get("metadatas", [[]])[0][idx],
                "distance": result.get("distances", [[]])[0][idx],
                "id": result.get("ids", [[]])[0][idx],
            }
        )
    return items


def answer_question(config: RagConfig, question: str, top_k: int = 5) -> tuple[str, list[dict[str, Any]]]:
    contexts = retrieve(config, question, top_k=top_k)
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
#outputs/rag_bge_m3_c