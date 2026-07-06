from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from tqdm import tqdm

from documents import list_gu_names, load_csv_documents
from embeddings import BgeM3EmbeddingFunction
from rag_config import AUXILIARY_EMBEDDING_MODELS, RagConfig, config_for_embedding_model


def get_or_create_collection(config: RagConfig):
    client = chromadb.PersistentClient(path=config.chroma_dir)
    return client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=BgeM3EmbeddingFunction(config.embedding_model),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(config: RagConfig) -> None:
    client = chromadb.PersistentClient(path=config.chroma_dir)
    try:
        client.delete_collection(config.collection_name)
    except Exception as exc:
        if "does not exist" not in str(exc).lower():
            raise


def ingest_csv(
    config: RagConfig,
    batch_size: int = 128,
    reset: bool = True,
    gu_name: str | None = None,
    chunk_size: int = 1,
    skip_existing: bool = True,
    skip_if_indexed: bool = False,
    progress_callback=None,
) -> int:
    def report(message: str, done: int | None = None, total: int | None = None) -> None:
        if progress_callback is not None:
            progress_callback(message, done, total)

    csv_path = Path(config.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    if reset:
        report("기존 컬렉션 삭제")
        reset_collection(config)

    report("컬렉션 준비")
    collection = get_or_create_collection(config)
    if skip_if_indexed and not reset:
        existing_count = collection.count()
        if existing_count > 0:
            report(f"기존 인덱싱 발견: {existing_count:,}건. 인덱싱을 스킵합니다.", 0, 0)
            return 0

    report("CSV 읽기 및 문서 변환")
    documents, metadatas, ids = load_csv_documents(str(csv_path), gu_name=gu_name, chunk_size=chunk_size)
    total = len(documents)
    if total == 0:
        report("인덱싱할 데이터가 없습니다.", 0, 0)
        return 0

    if skip_existing:
        report("기존 임베딩 ID 확인")
        new_documents: list[str] = []
        new_metadatas: list[dict[str, str | int | float]] = []
        new_ids: list[str] = []
        skipped = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_ids = ids[start:end]
            existing = collection.get(ids=batch_ids)
            existing_ids = set(existing.get("ids", []))
            for document, metadata, doc_id in zip(documents[start:end], metadatas[start:end], batch_ids):
                if doc_id in existing_ids:
                    skipped += 1
                    continue
                new_documents.append(document)
                new_metadatas.append(metadata)
                new_ids.append(doc_id)
            report(f"기존 ID 확인: {end:,}/{total:,}건, 스킵 {skipped:,}건", end, total)

        documents, metadatas, ids = new_documents, new_metadatas, new_ids
        report(f"스킵 완료: 기존 {skipped:,}건, 신규 {len(ids):,}건", 0, len(ids))

    total_to_add = len(documents)
    if total_to_add == 0:
        report("새로 임베딩할 데이터가 없습니다.", 0, 0)
        return 0

    report(f"임베딩 및 저장 시작: {total_to_add:,}건", 0, total_to_add)
    for start in tqdm(range(0, total_to_add, batch_size), desc="Indexing"):
        end = min(start + batch_size, total_to_add)
        batch = {
            "documents": documents[start:end],
            "metadatas": metadatas[start:end],
            "ids": ids[start:end],
        }
        if skip_existing:
            collection.add(**batch)
        else:
            collection.upsert(**batch)
        report(f"저장 완료: {end:,}/{total_to_add:,}건", end, total_to_add)

    report("인덱싱 완료", total_to_add, total_to_add)
    return total_to_add


def ingest_csv_for_models(
    config: RagConfig,
    model_names: list[str],
    batch_size: int = 128,
    reset: bool = True,
    gu_name: str | None = None,
    chunk_size: int = 1,
    skip_existing: bool = True,
    skip_if_indexed: bool = False,
    progress_callback=None,
) -> dict[str, int]:
    results: dict[str, int] = {}
    for model_name in model_names:
        model_config = config_for_embedding_model(config, model_name)
        if progress_callback is not None:
            progress_callback(f"[{model_name}] 인덱싱 시작", None, None)
        results[model_name] = ingest_csv(
            model_config,
            batch_size=batch_size,
            reset=reset,
            gu_name=gu_name,
            chunk_size=chunk_size,
            skip_existing=skip_existing,
            skip_if_indexed=skip_if_indexed,
            progress_callback=progress_callback,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV를 BGE-M3 임베딩으로 ChromaDB에 인덱싱합니다.")
    parser.add_argument("--csv", default=None, help="CSV 파일 경로")
    parser.add_argument("--chroma-dir", default=None, help="ChromaDB 저장 폴더")
    parser.add_argument("--collection", default=None, help="컬렉션 이름")
    parser.add_argument("--embedding-model", default=None, help="임베딩 모델")
    parser.add_argument(
        "--aux-models",
        nargs="*",
        default=[],
        help="함께 인덱싱할 보조 임베딩 모델 목록",
    )
    parser.add_argument(
        "--include-recommended-aux",
        action="store_true",
        help="추천 보조 모델 2개를 함께 인덱싱",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="배치 크기")
    parser.add_argument("--chunk-size", type=int, default=1, help="검색 문서 하나로 묶을 CSV 행 수")
    parser.add_argument("--append", action="store_true", help="기존 컬렉션에 추가")
    parser.add_argument("--no-skip-existing", action="store_true", help="기존 ID도 다시 인덱싱 시도")
    parser.add_argument("--skip-if-indexed", action="store_true", help="컬렉션에 기존 데이터가 있으면 인덱싱하지 않음")
    parser.add_argument("--gu", default=None, help="특정 자치구만 인덱싱. 예: 서초구")
    parser.add_argument("--list-gu", action="store_true", help="CSV의 자치구 목록 출력")
    args = parser.parse_args()

    base = RagConfig()
    config = RagConfig(
        csv_path=args.csv or base.csv_path,
        chroma_dir=args.chroma_dir or base.chroma_dir,
        collection_name=args.collection or base.collection_name,
        embedding_model=args.embedding_model or base.embedding_model,
        ollama_model=base.ollama_model,
    )

    if args.list_gu:
        for gu_name in list_gu_names(config.csv_path):
            print(gu_name)
        return

    model_names = [config.embedding_model]
    if args.include_recommended_aux:
        model_names.extend(AUXILIARY_EMBEDDING_MODELS)
    model_names.extend(args.aux_models)
    model_names = list(dict.fromkeys(model_names))

    results = ingest_csv_for_models(
        config,
        model_names=model_names,
        batch_size=args.batch_size,
        reset=not args.append,
        gu_name=args.gu,
        chunk_size=args.chunk_size,
        skip_existing=not args.no_skip_existing,
        skip_if_indexed=args.skip_if_indexed,
    )
    for model_name, count in results.items():
        print(f"[{model_name}] 인덱싱 완료: {count:,}건")


if __name__ == "__main__":
    main()
