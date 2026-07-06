from __future__ import annotations

import argparse
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

DEFAULT_CSV_PATH = r"D:\AI_학습\AI_test\AI_3\dataset\서울시 부동산 실거래가 정보_202606.csv"


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


def clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_number(value: Any) -> float | None:
    text = clean_value(value).replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_date(value: Any) -> str:
    text = clean_value(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def format_price(amount_manwon: float | None) -> str:
    if amount_manwon is None:
        return "미상"
    eok = amount_manwon / 10000
    return f"{amount_manwon:,.0f}만원 ({eok:,.2f}억원)"


def row_to_document(row: pd.Series) -> tuple[str, dict[str, str | int | float]]:
    amount = parse_number(row.get("물건금액(만원)"))
    building_area = parse_number(row.get("건물면적(㎡)"))
    land_area = parse_number(row.get("토지면적(㎡)"))
    contract_date = format_date(row.get("계약일"))

    gu = clean_value(row.get("자치구명"))
    dong = clean_value(row.get("법정동명"))
    main_no = clean_value(row.get("본번")).lstrip("0")
    sub_no = clean_value(row.get("부번")).lstrip("0")
    lot_no = "-".join(part for part in [main_no, sub_no] if part)
    building = clean_value(row.get("건물명"))
    building_use = clean_value(row.get("건물용도"))

    location_parts = [part for part in [gu, dong, lot_no, building] if part]