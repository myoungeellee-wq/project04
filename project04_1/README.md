# 서울시 부동산 실거래가 RAG

인덱싱/임베딩 처리와 질문 화면을 분리한 RAG 예제입니다.

- 임베딩: `BAAI/bge-m3`
- 벡터 DB: `ChromaDB`
- 답변 생성: `Ollama`

## 설치

```powershell
cd C:\Users\human-17\Documents\Codex\2026-07-03\cja\outputs\rag_bge_m3_chromadb_ollama
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
ollama pull qwen2.5:7b
```

Word2Vec 시각화에서 `No module named 'gensim'` 오류가 나면 현재 실행 중인 가상환경에 의존성이 설치되지 않은 것입니다.

```powershell
pip install -r requirements.txt
```

또는 Word2Vec 관련 패키지만 설치할 수 있습니다.

```powershell
pip install gensim "scipy<1.14" plotly scikit-learn
```

## 1. 인덱싱 CLI

CSV를 임베딩해서 ChromaDB에 저장합니다.

```powershell
python index_data.py
```

CSV 경로를 직접 지정할 수도 있습니다.

```powershell
python index_data.py --csv "D:\AI_학습\AI_test\AI_3\dataset\서울시 부동산 실거래가 정보_202606.csv"
```

기존 컬렉션에 추가하려면:

```powershell
python index_data.py --append
```

특정 자치구만 추가하려면:

```powershell
python index_data.py --append --gu "서초구"
```

청킹 크기를 조정하려면:

```powershell
python index_data.py --append --gu "서초구" --chunk-size 5
```

이미 인덱싱된 ID는 기본적으로 스킵합니다. 같은 ID를 다시 임베딩하고 갱신하려면:

```powershell
python index_data.py --append --no-skip-existing
```

컬렉션에 기존 인덱싱 데이터가 있으면 실행 자체를 건너뛰려면:

```powershell
python index_data.py --append --skip-if-indexed
```

추천 보조 임베딩 모델 2개를 함께 인덱싱하려면:

```powershell
python index_data.py --append --include-recommended-aux
```

보조 모델을 직접 지정하려면:

```powershell
python index_data.py --append --aux-models intfloat/multilingual-e5-large BAAI/bge-large-zh-v1.5
```

CSV 안의 자치구 목록을 확인하려면:

```powershell
python index_data.py --list-gu
```

## 2. Streamlit 화면

Streamlit 화면에서는 `추가 인덱싱`, `질문`, `면적당금액 3D`, `법정동/용도 3D`, `Word2Vec 시각화` 탭을 함께 제공합니다.

```powershell
streamlit run rag_app.py
```

`추가 인덱싱` 탭에서는 CSV 경로를 기준으로 전체 또는 특정 자치구를 선택해 기존 컬렉션에 추가 인덱싱합니다. 이미 임베딩된 ID는 기본적으로 스킵하며, `컬렉션에 기존 인덱싱이 있으면 실행 안 함`을 켜면 문서 수가 1건 이상인 컬렉션은 인덱싱 자체를 건너뜁니다. 청킹 크기를 조정해 여러 CSV 행을 하나의 검색 문서로 묶을 수 있습니다. 진행률과 처리 로그가 화면에 표시됩니다.

보조 임베딩 모델을 선택하면 모델별 별도 컬렉션에 저장됩니다. 예를 들어 기본 컬렉션이 `seoul_real_estate_202606`이면 보조 모델 컬렉션은 `seoul_real_estate_202606__intfloat_multilingual_e5_large`처럼 생성됩니다.

`질문` 탭에서는 기본 컬렉션과 선택한 보조 모델 컬렉션을 함께 검색하고 Ollama로 답변을 생성합니다.
답변 생성 후 SAR 보고서를 화면에서 확인하고 Markdown 파일로 다운로드할 수 있습니다.

### 외부 Ollama 연결

Streamlit 사이드바의 `Ollama 연결 방식`에서 `Local/외부 Ollama 서버`를 선택하고 `Ollama Base URL`에 외부 Ollama 서버 주소를 입력합니다.

```text
http://192.168.0.10:11434
http://your-ollama-server:11434
```

입력 후 `Ollama 연결 확인` 버튼을 누르면 `/api/tags`를 호출해 서버 연결과 모델 존재 여부를 확인합니다.

외부 Ollama 서버에서는 다음처럼 외부 접속을 허용한 상태로 실행해야 합니다.

```powershell
$env:OLLAMA_HOST="0.0.0.0:11434"
ollama serve
```

사용할 모델이 외부 서버에 없다면 외부 서버에서 먼저 내려받습니다.

```powershell
ollama pull qwen2.5:7b
```

### Ollama Cloud 또는 무료 Ollama-compatible 서버 연결

Streamlit 사이드바에서 `Ollama 연결 방식`을 `Ollama Cloud/Compatible API`로 선택합니다.

- `Ollama Base URL`: Cloud 또는 무료 서버에서 제공한 Ollama-compatible API 주소
- `Ollama API Key`: 서버에서 발급받은 API Key
- `Ollama 모델`: 서버에 등록된 모델명

`Ollama 연결 확인` 버튼은 먼저 `/api/tags`를 확인하고, 실패하면 OpenAI-compatible 서버를 위해 `/v1/models`를 한 번 더 확인합니다.

`.env`로 설정하려면 다음처럼 입력합니다.

```text
OLLAMA_PROVIDER=cloud
OLLAMA_BASE_URL=https://your-cloud-ollama-endpoint
OLLAMA_API_KEY=your_api_key
OLLAMA_MODEL=qwen2.5:7b
```

공급자가 무료 서버를 제공하더라도 URL과 모델명은 서비스마다 다릅니다. Streamlit 화면에서 받은 값 그대로 입력하면 됩니다.

Streamlit Cloud에 배포할 때는 API Key를 코드나 공개 저장소의 `.env`에 넣지 말고 `Manage app` -> `Secrets`에 등록하세요.

```toml
OLLAMA_PROVIDER = "cloud"
OLLAMA_BASE_URL = "https://your-cloud-ollama-endpoint"
OLLAMA_API_KEY = "your_api_key"
OLLAMA_MODEL = "qwen2.5:7b"
```

이미 노출된 API Key는 폐기하고 새 키를 발급받는 것을 권장합니다.

`면적당금액 3D` 탭에서는 거래금액, 건물면적, 면적당금액을 축으로 하는 3D 산점도와 자치구/용도별 면적당금액 분포를 제공합니다.

`법정동/용도 3D` 탭에서는 법정동, 건물용도, 면적당금액을 축으로 하는 3D 산점도와 법정동/용도별 면적당금액 중앙값 heatmap을 제공합니다.

`Word2Vec 시각화` 탭에서는 CSV 문서 텍스트로 Word2Vec을 학습하고 다음 기능을 제공합니다.

- 빈도 상위 단어 표
- 기준 단어 유사어 막대 그래프
- 기준 단어 유사어 네트워크
- PCA 기반 2D 단어 산점도

## 파일 구조

- `rag_config.py`: 환경 설정과 기본값
- `embeddings.py`: BGE-M3 임베딩 함수
- `documents.py`: CSV 행을 검색 문서와 메타데이터로 변환
- `index_data.py`: CSV 인덱싱 전용 CLI
- `rag_query.py`: ChromaDB 검색과 Ollama 답변 생성
- `rag_app.py`: Streamlit 질문 화면
- `reporting.py`: SAR 보고서 생성

## 계산 필드

인덱싱 시 `물건금액(만원) / 건물면적(㎡)` 기준으로 면적당금액을 계산해 문서 텍스트와 메타데이터에 함께 저장합니다.

- `price_per_m2_manwon`: 만원/㎡
- `price_per_pyeong_manwon`: 만원/평
- 청킹 문서에는 `avg_price_per_m2_manwon`, `min_price_per_m2_manwon`, `max_price_per_m2_manwon`도 저장
