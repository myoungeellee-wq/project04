# 서울시 부동산 실거래가 RAG

CSV 파일을 `BGE-M3` 임베딩으로 벡터화해 `ChromaDB`에 저장하고, `Ollama` 로컬 LLM으로 질의응답하는 예제입니다.

## 1. 준비

```powershell
cd C:\Users\human-17\Documents\Codex\2026-07-03\cja\outputs\rag_bge_m3_chromadb_ollama
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Ollama가 설치되어 있어야 합니다.

```powershell
ollama pull qwen2.5:7b
```

다른 모델을 쓰려면 `.env`의 `OLLAMA_MODEL` 값을 바꾸면 됩니다.

## 2. 인덱싱

```powershell
python ingest.py
```

특정 CSV 경로를 직접 지정할 수도 있습니다.

```powershell
python ingest.py --csv "D:\AI_학습\AI_test\AI_3\dataset\서울시 부동산 실거래가 정보_202606.csv"
```

## 3. 질문

```powershell
python ask.py "2026년 6월 서초구 오피스텔 거래 중 3억원대 사례를 알려줘"
```

검색 결과를 더 많이 넣고 싶으면:

```powershell
python ask.py "양재동 연립다세대 거래를 요약해줘" --top-k 8
```

## 파일 구성

- `rag_core.py`: CSV 문서화, BGE-M3 임베딩 함수, ChromaDB, Ollama RAG 공통 로직
- `ingest.py`: CSV를 ChromaDB에 적재
- `ask.py`: 질문 검색 및 답변 생성
- `.env.example`: 실행 환경 설정 예시