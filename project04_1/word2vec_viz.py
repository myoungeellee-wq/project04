from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from documents import load_csv_documents


DEFAULT_STOPWORDS = {
    "서울시",
    "부동산",
    "실거래가",
    "기록",
    "계약일",
    "위치",
    "자치구",
    "법정동",
    "건물명",
    "건물용도",
    "거래금액",
    "건물면적",
    "토지면적",
    "건축년도",
    "거래유형",
    "중개사",
    "소재지",
    "미상",
}


@dataclass
class Word2VecArtifacts:
    model: Any
    sentences: list[list[str]]
    token_counts: Counter


def tokenize(text: str, stopwords: set[str] | None = None) -> list[str]:
    stopwords = stopwords or DEFAULT_STOPWORDS
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", text)
    return [token for token in tokens if len(token) >= 2 and token not in stopwords]


def build_sentences(
    csv_path: str,
    gu_name: str | None = None,
    chunk_size: int = 1,
    max_documents: int | None = None,
) -> list[list[str]]:
    documents, _, _ = load_csv_documents(csv_path, gu_name=gu_name, chunk_size=chunk_size)
    if max_documents:
        documents = documents[:max_documents]
    return [tokens for document in documents if (tokens := tokenize(document))]


def train_word2vec(
    csv_path: str,
    gu_name: str | None = None,
    chunk_size: int = 1,
    max_documents: int | None = None,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 20,
    sg: int = 1,
) -> Word2VecArtifacts:
    try:
        from gensim.models import Word2Vec
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Word2Vec 시각화에는 gensim이 필요합니다. "
            "현재 가상환경에서 `pip install -r requirements.txt` 또는 "
            "`pip install gensim \"scipy<1.14\"`를 실행해주세요."
        ) from exc

    sentences = build_sentences(
        csv_path=csv_path,
        gu_name=gu_name,
        chunk_size=chunk_size,
        max_documents=max_documents,
    )
    if not sentences:
        raise ValueError("Word2Vec을 학습할 토큰이 없습니다.")

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        sg=sg,
        epochs=epochs,
        seed=42,
    )
    token_counts = Counter(token for sentence in sentences for token in sentence)
    return Word2VecArtifacts(model=model, sentences=sentences, token_counts=token_counts)


def vocabulary_table(artifacts: Word2VecArtifacts, limit: int = 50) -> list[dict[str, Any]]:
    vocab = artifacts.model.wv.key_to_index
    rows: list[dict[str, Any]] = []
    for word, count in artifacts.token_counts.most_common():
        if word in vocab:
            rows.append({"word": word, "count": count})
        if len(rows) >= limit:
            break
    return rows


def similar_words(artifacts: Word2VecArtifacts, word: str, topn: int = 10) -> list[dict[str, Any]]:
    if word not in artifacts.model.wv:
        raise KeyError(f"'{word}' 단어가 Word2Vec 어휘에 없습니다.")
    return [
        {"word": item_word, "similarity": float(score)}
        for item_word, score in artifacts.model.wv.most_similar(word, topn=topn)
    ]


def similar_words_figure(artifacts: Word2VecArtifacts, word: str, topn: int = 10):
    import plotly.express as px

    rows = similar_words(artifacts, word, topn=topn)
    fig = px.bar(
        rows,
        x="similarity",
        y="word",
        orientation="h",
        title=f"'{word}' 유사어",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def scatter_figure(artifacts: Word2VecArtifacts, limit: int = 100):
    import pandas as pd
    import plotly.express as px
    from sklearn.decomposition import PCA

    words = [row["word"] for row in vocabulary_table(artifacts, limit=limit)]
    if len(words) < 2:
        raise ValueError("2D 시각화를 만들 어휘가 부족합니다.")

    vectors = [artifacts.model.wv[word] for word in words]
    coords = PCA(n_components=2, random_state=42).fit_transform(vectors)
    df = pd.DataFrame(
        {
            "word": words,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "count": [artifacts.token_counts[word] for word in words],
        }
    )
    return px.scatter(
        df,
        x="x",
        y="y",
        text="word",
        size="count",
        hover_data=["count"],
        title="Word2Vec 어휘 2D 시각화",
    )


def network_figure(artifacts: Word2VecArtifacts, seed_word: str, topn: int = 8):
    import plotly.graph_objects as go

    if seed_word not in artifacts.model.wv:
        raise KeyError(f"'{seed_word}' 단어가 Word2Vec 어휘에 없습니다.")

    neighbors = artifacts.model.wv.most_similar(seed_word, topn=topn)
    nodes = [seed_word] + [word for word, _ in neighbors]
    angle_step = 2 * math.pi / max(1, len(neighbors))
    positions = {seed_word: (0.0, 0.0)}
    for idx, (word, _) in enumerate(neighbors):
        angle = idx * angle_step
        positions[word] = (math.cos(angle), math.sin(angle))

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for word, _ in neighbors:
        x0, y0 = positions[seed_word]
        x1, y1 = positions[word]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [positions[word][0] for word in nodes]
    node_y = [positions[word][1] for word in nodes]
    node_sizes = [34] + [18 + 20 * score for _, score in neighbors]
    labels = [seed_word] + [f"{word}<br>{score:.3f}" for word, score in neighbors]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1), hoverinfo="none"))
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=nodes,
            hovertext=labels,
            hoverinfo="text",
            textposition="top center",
            marker=dict(size=node_sizes),
        )
    )
    fig.update_layout(
        title=f"'{seed_word}' 유사어 네트워크",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
