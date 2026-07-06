# 사전설치 : pip install -U langchain-ollama
import gradio as gr
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 설정
DB_PATH = "./faiss_index"
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "gemma4:e2b"

# --- RAG 체인 설정 ---
def get_lcel_chain():
    # 1. 임베딩 및 벡터 DB 로드
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("벡터 DB가 없습니다. ingest.py를 먼저 실행하세요.")

    vectorstore = FAISS.load_local(
        DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True   # 시스템에게 허락을 해주는 보안 승인 스위치
    )
    
    # 검색기 (Retriever)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. LLM 설정
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    # 3. 프롬프트 템플릿
    template = """
    당신은 문서를 기반으로 질문에 답변하는 AI입니다.
    아래 [Context]를 참고하여 질문에 한국어로 답변하세요.
    모르는 내용은 솔직히 모른다고 말하세요.

    [Context]:
    {context}

    질문: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. 문서 포맷팅 함수 (검색된 문서들을 하나의 글자로 합침)
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # 5. LCEL 체인 구성 (핵심 부분!)
    # retriever | format_docs : 검색된 문서 리스트를 텍스트로 변환하여 context에 넣음
    # RunnablePassthrough() : 사용자의 질문을 그대로 question에 넣음
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# 체인 전역 로드
rag_chain = None
try:
    rag_chain = get_lcel_chain()
    print("LCEL 방식 RAG 시스템 준비 완료!")
except Exception as e:
    print(f"초기화 실패: {e}")

# --- Gradio 인터페이스 ---
def chat(message, history):
    if rag_chain is None:
        return "시스템 오류: DB 로드 실패"
    
    # stream()을 쓰면 타자 치듯 답변이 나옵니다. (Gradio에서 지원)
    # 여기서는 간단히 invoke() 사용
    return rag_chain.invoke(message)

if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=chat,
        title="RAG 챗봇 (LCEL 버전)",
        description=f"LCEL 파이프라인으로 구성된 문서 검색 챗봇입니다. ({LLM_MODEL})"
    )
    demo.launch()