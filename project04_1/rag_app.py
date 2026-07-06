from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st

from analysis_viz import (
    dong_use_price_3d_figure,
    dong_use_price_heatmap,
    list_building_uses,
    list_dongs,
    prepare_dong_use_price_dataframe,
    prepare_price_area_dataframe,
    price_area_3d_figure,
    price_per_m2_box_figure,
)
from documents import list_gu_names
from index_data import ingest_csv_for_models
from rag_config import AUXILIARY_EMBEDDING_MODELS, RagConfig
from rag_query import answer_question, get_existing_collection
from word2vec_viz import (
    network_figure,
    scatter_figure,
    similar_words_figure,
    train_word2vec,
    vocabulary_table,
)


def save_uploaded_csv(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None
    upload_dir = Path("uploaded_data")
    upload_dir.mkdir(exist_ok=True)
    uploaded_path = upload_dir / uploaded_file.name
    uploaded_path.write_bytes(uploaded_file.getbuffer())
    return str(uploaded_path)


def build_config() -> RagConfig:
    base = RagConfig()
    return RagConfig(
        csv_path=st.session_state.csv_path.strip() or base.csv_path,
        chroma_dir=st.session_state.chroma_dir.strip() or base.chroma_dir,
        collection_name=st.session_state.collection_name.strip() or base.collection_name,
        embedding_model=st.session_state.embedding_model.strip() or base.embedding_model,
        ollama_model=st.session_state.ollama_model.strip() or base.ollama_model,
        ollama_base_url=st.session_state.ollama_base_url.strip() or base.ollama_base_url,
        ollama_temperature=float(st.session_state.ollama_temperature),
    )


def show_source(idx: int, source: dict) -> None:
    metadata = source["metadata"] or {}
    title = (
        f"{idx}. {metadata.get('contract_date', '날짜 미상')} | "
        f"{metadata.get('gu', '')} {metadata.get('dong', '')} | "
        f"{metadata.get('building') or metadata.get('building_use') or source.get('id') or '검색 결과'}"
    )
    with st.expander(title):
        distance = source.get("distance")
        if distance is not None:
            st.caption(f"distance: {distance:.4f}")
        st.caption(f"model: {source.get('model', '')} | collection: {source.get('collection', '')}")
        st.json(metadata)
        st.text(source["document"])


def app() -> None:
    st.set_page_config(page_title="서울시 부동산 RAG 질문", layout="wide")

    st.title("서울시 부동산 실거래가 RAG 질문")
    st.write("이미 인덱싱된 ChromaDB 컬렉션을 사용해 질문 결과만 가져옵니다.")

    with st.sidebar:
        st.header("설정")
        base = RagConfig()
        uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
        uploaded_path = save_uploaded_csv(uploaded_file)
        if uploaded_path:
            st.session_state.csv_path = uploaded_path
            st.success(f"업로드 CSV 사용: {uploaded_file.name}")
        st.text_input("CSV 경로", value=os.getenv("CSV_PATH", base.csv_path), key="csv_path")
        st.text_input("ChromaDB 저장 폴더", value=os.getenv("CHROMA_DIR", "./chroma_db"), key="chroma_dir")
        st.text_input(
            "컬렉션 이름",
            value=os.getenv("COLLECTION_NAME", "seoul_real_estate_202606"),
            key="collection_name",
        )
        st.text_input("임베딩 모델", value=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"), key="embedding_model")
        st.text_input("Ollama 모델", value=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"), key="ollama_model")
        st.text_input(
            "Ollama Base URL",
            value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            key="ollama_base_url",
        )
        st.number_input(
            "Ollama Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
            step=0.1,
            key="ollama_temperature",
        )
        top_k = st.slider("검색 문서 수", min_value=1, max_value=15, value=5)

        if st.button("컬렉션 확인", use_container_width=True):
            try:
                collection = get_existing_collection(build_config())
                st.success(f"문서 수: {collection.count():,}건")
            except Exception as exc:
                st.error(f"컬렉션 확인 실패: {exc}")

    tab_index, tab_ask, tab_analysis, tab_dong_use, tab_word2vec = st.tabs([
        "추가 인덱싱",
        "질문",
        "면적당금액 3D",
        "법정동/용도 3D",
        "Word2Vec 시각화",
    ])

    with tab_index:
        st.subheader("자치구별 또는 전체 추가 인덱싱")
        st.caption("기존 컬렉션을 유지하고 선택한 데이터만 upsert로 추가/갱신합니다.")
        config = build_config()
        st.code(config.csv_path, language="text")

        gu_options = ["전체"]
        try:
            gu_options += list_gu_names(config.csv_path)
        except Exception as exc:
            st.warning(f"자치구 목록을 불러오지 못했습니다: {exc}")

        selected_gu = st.selectbox("인덱싱 범위", gu_options)
        aux_models_for_index = st.multiselect(
            "함께 인덱싱할 보조 임베딩 모델",
            AUXILIARY_EMBEDDING_MODELS,
            default=[],
        )
        batch_size = st.number_input("배치 크기", min_value=16, max_value=1024, value=128, step=16)
        chunk_size = st.number_input(
            "청킹 크기",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            help="CSV 몇 행을 하나의 검색 문서로 묶을지 정합니다. 1이면 거래 1건이 문서 1개입니다.",
        )
        skip_existing = st.checkbox("이미 임베딩된 ID는 스킵", value=True)
        skip_if_indexed = st.checkbox(
            "컬렉션에 기존 인덱싱이 있으면 실행 안 함",
            value=False,
            help="선택한 모델별 컬렉션에 문서가 1건 이상 있으면 CSV 읽기/임베딩을 하지 않고 스킵합니다.",
        )
        reset_before_index = st.checkbox("기존 컬렉션 삭제 후 새로 인덱싱", value=False)
        progress_bar = st.progress(0, text="대기 중")
        log_area = st.empty()

        if "index_logs" not in st.session_state:
            st.session_state.index_logs = []

        def progress_callback(message: str, done: int | None, total: int | None) -> None:
            st.session_state.index_logs.append(message)
            log_area.text("\n".join(st.session_state.index_logs[-30:]))
            if total and done is not None:
                progress_bar.progress(done / total, text=message)
            else:
                progress_bar.progress(0, text=message)

        if st.button("추가 인덱싱 실행", type="primary"):
            st.session_state.index_logs = []
            try:
                model_names = [config.embedding_model] + aux_models_for_index
                results = ingest_csv_for_models(
                    config,
                    model_names=list(dict.fromkeys(model_names)),
                    batch_size=int(batch_size),
                    reset=reset_before_index,
                    gu_name=None if selected_gu == "전체" else selected_gu,
                    chunk_size=int(chunk_size),
                    skip_existing=skip_existing,
                    skip_if_indexed=skip_if_indexed,
                    progress_callback=progress_callback,
                )
                result_text = ", ".join(f"{model}: {count:,}건" for model, count in results.items())
                st.success(f"인덱싱 완료: {result_text}")
            except Exception as exc:
                st.error(f"인덱싱 실패: {exc}")

    with tab_ask:
        aux_models_for_query = st.multiselect(
            "검색에 사용할 보조 임베딩 모델",
            AUXILIARY_EMBEDDING_MODELS,
            default=[],
        )
        question = st.text_area(
            "질문 입력",
            placeholder="예: 2026년 6월 서초구 오피스텔 거래 중 3억원대 사례를 알려줘",
            height=140,
        )

        col_answer, col_clear = st.columns([1, 1])
        with col_answer:
            ask_clicked = st.button("질문하기", type="primary", use_container_width=True)
        with col_clear:
            if st.button("결과 지우기", use_container_width=True):
                st.session_state.pop("last_answer", None)
                st.session_state.pop("last_sources", None)
                st.session_state.pop("last_timings", None)

        if ask_clicked:
            if not question.strip():
                st.warning("질문을 입력해주세요.")
                return

            try:
                timer_area = st.empty()
                status_area = st.empty()
                started_at = time.perf_counter()

                status_area.info("기존 ChromaDB에서 검색하고 Ollama로 답변을 생성하는 중입니다...")
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        answer_question,
                        build_config(),
                        question.strip(),
                        int(top_k),
                        aux_models_for_query,
                    )
                    while not future.done():
                        elapsed = time.perf_counter() - started_at
                        timer_area.metric("답변 생성 중 경과 시간", f"{elapsed:.1f}초")
                        time.sleep(0.2)

                    answer, sources, timings = future.result()

                timer_area.metric("답변 생성 완료 시간", f"{timings.get('total_seconds', 0):.2f}초")
                status_area.success("답변 생성 완료")
                st.session_state.last_answer = answer
                st.session_state.last_sources = sources
                st.session_state.last_timings = timings
            except Exception as exc:
                st.error(f"답변 생성 실패: {exc}")
                st.info("ChromaDB 폴더, 컬렉션 이름, 임베딩 모델이 기존 인덱스와 같은지 확인해주세요.")

        if "last_answer" in st.session_state:
            st.subheader("답변")
            if "last_timings" in st.session_state:
                timings = st.session_state.last_timings
                timing_cols = st.columns(3)
                timing_cols[0].metric("검색 시간", f"{timings.get('retrieve_seconds', 0):.2f}초")
                timing_cols[1].metric("답변 생성 시간", f"{timings.get('generation_seconds', 0):.2f}초")
                timing_cols[2].metric("전체 시간", f"{timings.get('total_seconds', 0):.2f}초")
            st.write(st.session_state.last_answer)

        if "last_sources" in st.session_state:
            st.subheader("검색 근거")
            for idx, source in enumerate(st.session_state.last_sources, start=1):
                show_source(idx, source)

    with tab_analysis:
        st.subheader("거래금액 / 건물면적 / 면적당금액 3D 분석")
        config = build_config()
        st.code(config.csv_path, language="text")

        analysis_gu_options = ["전체"]
        try:
            analysis_gu_options += list_gu_names(config.csv_path)
        except Exception as exc:
            st.warning(f"자치구 목록을 불러오지 못했습니다: {exc}")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            analysis_gu = st.selectbox("분석 자치구", analysis_gu_options)
        with col_b:
            try:
                building_use_options = ["전체"] + list_building_uses(
                    config.csv_path,
                    gu_name=None if analysis_gu == "전체" else analysis_gu,
                )
            except Exception:
                building_use_options = ["전체"]
            analysis_building_use = st.selectbox("건물용도", building_use_options)
        with col_c:
            max_rows = st.number_input("최대 표시 건수", min_value=100, max_value=50000, value=5000, step=100)

        if st.button("3D 시각화 생성", type="primary"):
            try:
                df = prepare_price_area_dataframe(
                    config.csv_path,
                    gu_name=None if analysis_gu == "전체" else analysis_gu,
                    building_use=None if analysis_building_use == "전체" else analysis_building_use,
                    max_rows=int(max_rows),
                )
                st.session_state.price_area_df = df
                st.success(f"분석 데이터 준비 완료: {len(df):,}건")
            except Exception as exc:
                st.error(f"3D 시각화 데이터 준비 실패: {exc}")

        if "price_area_df" in st.session_state:
            df = st.session_state.price_area_df
            metric_cols = st.columns(4)
            metric_cols[0].metric("거래 건수", f"{len(df):,}")
            metric_cols[1].metric("평균 만원/㎡", f"{df['price_per_m2_manwon'].mean():,.2f}")
            metric_cols[2].metric("중앙값 만원/㎡", f"{df['price_per_m2_manwon'].median():,.2f}")
            metric_cols[3].metric("최대 만원/㎡", f"{df['price_per_m2_manwon'].max():,.2f}")
            st.plotly_chart(price_area_3d_figure(df), use_container_width=True)
            st.plotly_chart(price_per_m2_box_figure(df), use_container_width=True)
            st.dataframe(
                df[
                    [
                        "contract_date",
                        "gu",
                        "dong",
                        "building",
                        "building_use",
                        "amount_manwon",
                        "building_area_m2",
                        "price_per_m2_manwon",
                        "price_per_pyeong_manwon",
                    ]
                ].head(100),
                use_container_width=True,
            )

    with tab_dong_use:
        st.subheader("면적당금액 / 법정동 / 건물용도 3D 분석")
        config = build_config()
        st.code(config.csv_path, language="text")

        dong_gu_options = ["전체"]
        try:
            dong_gu_options += list_gu_names(config.csv_path)
        except Exception as exc:
            st.warning(f"자치구 목록을 불러오지 못했습니다: {exc}")

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            dong_gu = st.selectbox("자치구", dong_gu_options, key="dong_use_gu")
        with col_b:
            try:
                dong_options = ["전체"] + list_dongs(
                    config.csv_path,
                    gu_name=None if dong_gu == "전체" else dong_gu,
                )
            except Exception:
                dong_options = ["전체"]
            selected_dong = st.selectbox("법정동", dong_options)
        with col_c:
            try:
                use_options = ["전체"] + list_building_uses(
                    config.csv_path,
                    gu_name=None if dong_gu == "전체" else dong_gu,
                )
            except Exception:
                use_options = ["전체"]
            selected_use = st.selectbox("건물용도", use_options, key="dong_use_building_use")
        with col_d:
            dong_use_max_rows = st.number_input(
                "최대 표시 건수",
                min_value=100,
                max_value=50000,
                value=5000,
                step=100,
                key="dong_use_max_rows",
            )

        if st.button("법정동/용도 3D 생성", type="primary"):
            try:
                df = prepare_dong_use_price_dataframe(
                    config.csv_path,
                    gu_name=None if dong_gu == "전체" else dong_gu,
                    dong_name=None if selected_dong == "전체" else selected_dong,
                    building_use=None if selected_use == "전체" else selected_use,
                    max_rows=int(dong_use_max_rows),
                )
                st.session_state.dong_use_price_df = df
                st.success(f"분석 데이터 준비 완료: {len(df):,}건")
            except Exception as exc:
                st.error(f"법정동/용도 3D 데이터 준비 실패: {exc}")

        if "dong_use_price_df" in st.session_state:
            df = st.session_state.dong_use_price_df
            metric_cols = st.columns(4)
            metric_cols[0].metric("거래 건수", f"{len(df):,}")
            metric_cols[1].metric("법정동 수", f"{df['dong'].nunique():,}")
            metric_cols[2].metric("건물용도 수", f"{df['building_use'].nunique():,}")
            metric_cols[3].metric("중앙값 만원/㎡", f"{df['price_per_m2_manwon'].median():,.2f}")
            st.plotly_chart(dong_use_price_3d_figure(df), use_container_width=True)
            st.plotly_chart(dong_use_price_heatmap(df), use_container_width=True)
            st.dataframe(
                df[
                    [
                        "contract_date",
                        "gu",
                        "dong",
                        "building",
                        "building_use",
                        "amount_manwon",
                        "building_area_m2",
                        "price_per_m2_manwon",
                        "price_per_pyeong_manwon",
                    ]
                ].head(100),
                use_container_width=True,
            )

    with tab_word2vec:
        st.subheader("Word2Vec 단어 관계 시각화")
        st.caption("CSV 문서 텍스트를 Word2Vec으로 학습해 유사어와 단어 분포를 탐색합니다.")

        config = build_config()
        st.code(config.csv_path, language="text")

        viz_gu_options = ["전체"]
        try:
            viz_gu_options += list_gu_names(config.csv_path)
        except Exception as exc:
            st.warning(f"자치구 목록을 불러오지 못했습니다: {exc}")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            viz_gu = st.selectbox("시각화 범위", viz_gu_options, key="w2v_gu")
            w2v_chunk_size = st.number_input("Word2Vec 청킹 크기", min_value=1, max_value=100, value=1, step=1)
            max_documents = st.number_input("최대 문서 수", min_value=100, max_value=100000, value=10000, step=100)
        with col_b:
            vector_size = st.number_input("벡터 차원", min_value=20, max_value=500, value=100, step=10)
            window = st.number_input("윈도우 크기", min_value=2, max_value=20, value=5, step=1)
            min_count = st.number_input("최소 단어 빈도", min_value=1, max_value=50, value=2, step=1)
        with col_c:
            epochs = st.number_input("학습 epoch", min_value=1, max_value=100, value=20, step=1)
            sg_label = st.selectbox("학습 방식", ["Skip-gram", "CBOW"])
            vocab_limit = st.number_input("2D 표시 단어 수", min_value=20, max_value=500, value=100, step=10)

        if st.button("Word2Vec 학습", type="primary"):
            try:
                with st.spinner("Word2Vec 모델을 학습하는 중입니다..."):
                    st.session_state.word2vec_artifacts = train_word2vec(
                        csv_path=config.csv_path,
                        gu_name=None if viz_gu == "전체" else viz_gu,
                        chunk_size=int(w2v_chunk_size),
                        max_documents=int(max_documents),
                        vector_size=int(vector_size),
                        window=int(window),
                        min_count=int(min_count),
                        epochs=int(epochs),
                        sg=1 if sg_label == "Skip-gram" else 0,
                    )
                artifacts = st.session_state.word2vec_artifacts
                st.success(
                    f"학습 완료: 문장 {len(artifacts.sentences):,}개, 어휘 {len(artifacts.model.wv):,}개"
                )
            except Exception as exc:
                st.error(f"Word2Vec 학습 실패: {exc}")

        if "word2vec_artifacts" in st.session_state:
            artifacts = st.session_state.word2vec_artifacts
            st.subheader("빈도 상위 단어")
            st.dataframe(vocabulary_table(artifacts, limit=30), use_container_width=True)

            seed_word = st.text_input("기준 단어", value="")
            topn = st.slider("유사어 수", min_value=3, max_value=30, value=10)

            if seed_word:
                try:
                    st.plotly_chart(similar_words_figure(artifacts, seed_word.strip(), topn=topn), use_container_width=True)
                    st.plotly_chart(network_figure(artifacts, seed_word.strip(), topn=topn), use_container_width=True)
                except Exception as exc:
                    st.warning(f"유사어 시각화를 만들 수 없습니다: {exc}")

            try:
                st.plotly_chart(scatter_figure(artifacts, limit=int(vocab_limit)), use_container_width=True)
            except Exception as exc:
                st.warning(f"2D 시각화를 만들 수 없습니다: {exc}")


app()
