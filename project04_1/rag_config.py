from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()

DEFAULT_CSV_PATH = r"./서울시 부동산 실거래가 정보_202606.csv"
AUXILIARY_EMBEDDING_MODELS = [
    "intfloat/multilingual-e5-large",
    "BAAI/bge-large-zh-v1.5",
]


@dataclass(frozen=True)
class RagConfig:
    csv_path: str = os.getenv("CSV_PATH", DEFAULT_CSV_PATH)
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "seoul_real_estate_202606")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0"))


def safe_model_suffix(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def collection_name_for_model(base_collection_name: str, model_name: str, primary_model_name: str) -> str:
    if model_name == primary_model_name:
        return base_collection_name
    return f"{base_collection_name}__{safe_model_suffix(model_name)}"


def config_for_embedding_model(base_config: RagConfig, model_name: str) -> RagConfig:
    collection_name = collection_name_for_model(
        base_config.collection_name,
        model_name,
        base_config.embedding_model,
    )
    return RagConfig(
        csv_path=base_config.csv_path,
        chroma_dir=base_config.chroma_dir,
        collection_name=collection_name,
        embedding_model=model_name,
        ollama_model=base_config.ollama_model,
        ollama_base_url=base_config.ollama_base_url,
        ollama_temperature=base_config.ollama_temperature,
    )
