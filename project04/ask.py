from __future__ import annotations

import argparse

from rag_core import RagConfig, answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="ChromaDB 검색 결과와 Ollama로 질문에 답합니다.")
    parser.add_argument("question", help="질문")
    parser.add_argument("--top-k", type=int, default=5, help="검색할 문서 수")
    parser.add_argument("--chroma-dir", default=None, help="ChromaDB 저장 디렉터리")
    parser.add_argument("--collection", default=None, help="ChromaDB 컬렉션 이름")
    parser.add_argument("--embedding-model", default=None, help="SentenceTransformer 임베딩 모델")
    parser.add_argument("--ollama-model", default=None, help="Ollama 모델 이름")
    parser.add_argument("--show-sources", action="store_true", help="검색된 원문을 함께 출력")
    args = parser.parse_args()

    base = RagConfig()
    config = RagConfig(
        csv_path=base.csv_path,
        chroma_dir=args.chroma_dir or base.chroma_dir,
        collection_name=args.collection or base.collection_name,
        embedding_model=args.embedding_model or base.embedding_model,
        ollama_model=args.ollama_model or base.ollama_model,
    )

    answer, sources = answer_question(config, args.question, top_k=args.top_k)
    print(answer)

    if args.show_sources:
        print("\n\n[검색 근거]")
        for idx, source in enumerate(sources, start=1):
            meta = source["metadata"]
            print(
                f"{idx}. {meta.get('contract_date')} | {meta.get('gu')} {meta.get('dong')} "
                f"| {meta.get('building') or meta.get('building_use')} "
                f"| {meta.get('amount_manwon', '금액 미상')}만원 "
                f"| distance={source['distance']:.4f}"
            )


if __name__ == "__main__":
    main()