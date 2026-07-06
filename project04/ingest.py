from __future__ import annotations

import argparse
from http import client
from turtle import reset

from rag_core import RagConfig, ingest_csv


def main() -> None:
    client = chromadb.PersistentClient(path=config.chroma_dir)
    collections = [c.name for c in client.list_collections()]

    if reset and config.collection_name in collections:
        client.delete_collection(config.collection_name)
    
    parser = argparse.ArgumentParser(description="CSV를 BGE-M3 임베딩으로 ChromaDB에 적재합니다.")
    parser.add_argument("--csv", dest="csv_path", default=None, help="서울시 부동산 실거래가 CSV 경로")
    parser.add_argument("--chroma-dir", default=None, help="ChromaDB 저장 디렉터리")
    parser.add_argument("--collection", default=None, help="ChromaDB 컬렉션 이름")
    parser.add_argument("--embedding-model", default=None, help="SentenceTransformer 임베딩 모델")
    parser.add_argument("--batch-size", type=int, default=128, help="적재 배치 크기")
    parser.add_argument("--append", action="store_true", help="기존 컬렉션을 삭제하지 않고 추가")
    args = parser.parse_args()

    base = RagConfig()
    config = RagConfig(
        csv_path=args.csv_path or base.csv_path,
        chroma_dir=args.chroma_dir or base.chroma_dir,
        collection_name=args.collection or base.collection_name,
        embedding_model=args.embedding_model or base.embedding_model,
        ollama_model=base.ollama_model,
    )

    count = ingest_csv(config, batch_size=args.batch_size, reset=not args.append)
    print(f"완료: {count:,}건을 '{config.collection_name}' 컬렉션에 저장했습니다.")


if __name__ == "__main__":
    main()
