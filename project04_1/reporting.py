from __future__ import annotations

from datetime import datetime
from typing import Any


def format_seconds(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}초"


def format_source_line(idx: int, source: dict[str, Any]) -> str:
    metadata = source.get("metadata") or {}
    distance = source.get("distance")
    distance_text = f"{distance:.4f}" if distance is not None else "N/A"

    return (
        f"{idx}. "
        f"{metadata.get('contract_date', '날짜 미상')} | "
        f"{metadata.get('gu', '')} {metadata.get('dong', '')} | "
        f"{metadata.get('building') or metadata.get('building_use') or source.get('id') or '검색 결과'} | "
        f"model={source.get('model', '')} | "
        f"distance={distance_text}"
    )


def build_sar_report(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    timings: dict[str, float] | None = None,
    ollama_model: str = "",
    embedding_model: str = "",
    auxiliary_models: list[str] | None = None,
) -> str:
    timings = timings or {}
    auxiliary_models = auxiliary_models or []

    source_lines = "\n".join(
        f"- {format_source_line(idx, source)}"
        for idx, source in enumerate(sources, start=1)
    )

    source_details = "\n\n".join(
        [
            f"### 근거 {idx}\n"
            f"- ID: `{source.get('id', '')}`\n"
            f"- Model: `{source.get('model', '')}`\n"
            f"- Collection: `{source.get('collection', '')}`\n"
            f"- Metadata:\n\n```text\n{source.get('metadata')}\n```\n\n"
            f"- Document:\n\n```text\n{source.get('document', '')}\n```"
            for idx, source in enumerate(sources, start=1)
        ]
    )

    return f"""# SAR Report

생성일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## S. Summary

- 질문: {question}
- Ollama 모델: `{ollama_model}`
- 기본 임베딩 모델: `{embedding_model}`
- 보조 임베딩 모델: `{", ".join(auxiliary_models) if auxiliary_models else "없음"}`
- 검색 근거 수: {len(sources)}
- 검색 시간: {format_seconds(timings.get("retrieve_seconds"))}
- 답변 생성 시간: {format_seconds(timings.get("generation_seconds"))}
- 전체 시간: {format_seconds(timings.get("total_seconds"))}

## A. Analysis

아래 답변은 검색된 실거래가 근거를 바탕으로 생성되었습니다.

{answer}

## R. Result / Recommendation

### 주요 검색 근거

{source_lines if source_lines else "- 검색 근거 없음"}

### 참고 사항

- 본 보고서는 업로드/인덱싱된 CSV와 ChromaDB 검색 결과에 기반합니다.
- 검색 결과에 없는 내용은 확정 정보로 해석하면 안 됩니다.
- 면적당금액이 포함된 인덱스라면 만원/㎡ 및 만원/평 비교에 활용할 수 있습니다.

## Appendix. Source Details

{source_details if source_details else "검색 근거 상세 없음"}
"""
