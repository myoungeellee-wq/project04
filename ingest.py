# 사전설치: pip install pypdf
import os # 폴더 존재 여부 확인
import time # 작업 시간 측정
import math # 배치 개수 계산
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 설정 ---
print(os.getcwd())
DATA_PATH = "./documents"
DB_PATH = "./faiss_index"
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = "BAAI/bge-m3" # 현재 한국어/다국어 성능이 매우 좋은 모델
BATCH_SIZE = 16  # 한 번에 처리할 청크 개수 (RAM 절약을 위해 조절 가능)

def create_vector_db():
    start_total_time = time.time()
    
    # 1. 문서 로드
    print("\n" + "="*50)
    print(f"[1/4] 문서 로드 시작: '{DATA_PATH}'")
    
    if not os.path.exists(DATA_PATH):
        print(f"오류: '{DATA_PATH}' 폴더가 없습니다. 폴더를 생성하고 PDF를 넣어주세요.")
        return

    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        print("로드할 PDF 문서가 없습니다.")
        return

    print(f"   -> 총 {len(documents)} 페이지의 문서를 읽어왔습니다.")

    # 2. 텍스트 분할
    print(f"\n  [2/4] 텍스트 분할 (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    total_chunks = len(texts)
    print(f"   -> 총 {total_chunks}개의 청크(Chunk)로 분할되었습니다.")

    # 3. 임베딩 및 벡터 저장소 생성 (배치 처리)
    print(f"\n [3/4] 임베딩 모델 로드 및 벡터 변환 시작...")
    print(f"   -> 모델: {EMBEDDING_MODEL}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},   # GPU 쓰고 싶으면 'cuda'로 변경
        encode_kwargs={'normalize_embeddings': True}  # 정규화 (코사인 유사도에 좋음)
    )

    # 벡터 저장소 초기화 변수
    vectorstore = None
    
    # 배치 처리 로직 (RAM 보호 및 진행상황 표시)
    total_batches = math.ceil(total_chunks / BATCH_SIZE)
    print(f"   -> 총 {total_batches}번의 배치를 수행합니다. (Batch Size: {BATCH_SIZE})")

    for i in range(0, total_chunks, BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        current_batch = (i // BATCH_SIZE) + 1
        
        start_batch_time = time.time()
        
        if vectorstore is None:
            # 첫 번째 배치는 벡터 저장소를 생성
            vectorstore = FAISS.from_documents(batch_texts, embeddings)
        else:
            # 이후 배치는 기존 저장소에 추가
            vectorstore.add_documents(batch_texts)
            
        elapsed = time.time() - start_batch_time
        print(f"      [진행률: {current_batch}/{total_batches}] {len(batch_texts)}개 처리 완료 ({elapsed:.2f}초)")

    print(f"   -> 모든 문서의 벡터 변환이 완료되었습니다.")

    # 4. 저장
    print(f"\n [4/4] 로컬 디스크에 저장 중...")
    vectorstore.save_local(DB_PATH)
    
    total_time = time.time() - start_total_time
    print("="*50)
    print(f"모든 작업 완료! (소요 시간: {total_time:.2f}초)")
    print(f"저장 경로: {DB_PATH}")
    print("="*50 + "\n")

if __name__ == "__main__":
    create_vector_db()