# 서울시 부동산 실거래가 RAG

CSV 파일을 `BGE-M3` 임베딩으로 벡터화해 `ChromaDB`에 저장하고, `Ollama` 로컬 LLM으로 질의응답하는 단일 Python 파일 예제입니다. CLI와 Streamlit 화면을 모두 `rag_app.py` 하나로 실행합니다.

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

## 2. Streamlit 실행

```powershell
streamlit run rag_app.py
```

화면 왼쪽 사이드바에서 CSV 경로, ChromaDB 저장 위치, 컬렉션 이름, 임베딩 모델, Ollama 모델을 설정할 수 있습니다. `CSV 인덱싱` 버튼으로 벡터 DB를 만들고, 질문 입력 후 `답변 생성`을 누르면 검색 근거와 답변이 함께 표시됩니다.

## 3. CLI 인덱싱

```powershell
python rag_app.py ingest
```

특정 CSV 경로를 직접 지정할 수도 있습니다.

```powershell
python rag_app.py ingest --csv "D:\AI_학습\AI_test\AI_3\dataset\서울시 부동산 실거래가 정보_202606.csv"
```

## 4. CLI 질문

```powershell
python rag_app.py ask "2026년 6월 서초구 오피스텔 거래 중 3억원대 사례를 알려줘"
```

검색 결과를 더 많이 넣고 싶으면:

```powershell
python rag_app.py ask "양재동 연립다세대 거래를 요약해줘" --top-k 8
```

검색 근거를 함께 보고 싶으면:

```powershell
python rag_app.py ask "양재동 연립다세대 거래를 요약해줘" --show-sources
```

## 파일 구성

- `rag_app.py`: Streamlit UI, CSV 문서화, BGE-M3 임베딩, ChromaDB 적재/검색, Ollama 답변 생성
- `.env.example`: 실행 환경 설정 예시
- `requirements.txt`: Python 의존성
