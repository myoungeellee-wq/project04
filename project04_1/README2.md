# 서울시 부동산 실거래가 RAG

`rag_app.py`는 Streamlit 화면에서 CSV와 질문을 입력받아 실행하는 RAG 프로그램입니다.

- 임베딩: `BAAI/bge-m3`
- 벡터 DB: `ChromaDB`
- 답변 생성: `Ollama`

## 1. 설치

```powershell
cd C:\Users\human-17\Documents\Codex\2026-07-03\cja\outputs\rag_bge_m3_chromadb_ollama
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Ollama 모델도 준비합니다.

```powershell
ollama pull qwen2.5:7b
```

## 2. 실행

```powershell
streamlit run rag_app.py
```

## 3. 화면에서 입력하는 값

- CSV 파일 업로드 또는 CSV 경로 직접 입력
- ChromaDB 저장 폴더
- 컬렉션 이름
- 임베딩 모델
- Ollama 모델
- 인덱싱 배치 크기
- 검색 문서 수
- 질문

먼저 `CSV 인덱싱 실행`을 눌러 벡터 DB를 만든 뒤, 질문을 입력하고 `답변 생성`을 누르면 됩니다.

## 파일 구성

- `rag_app.py`: Streamlit UI, CSV 문서화, BGE-M3 임베딩, ChromaDB 적재/검색, Ollama 답변 생성
- `.env.example`: 기본 설정 예시
- `requirements.txt`: Python 의존성
