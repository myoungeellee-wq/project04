from __future__ import annotations

import pandas as pd

from documents import load_analysis_dataframe


def prepare_price_area_dataframe(
    csv_path: str,
    gu_name: str | None = None,
    building_use: str | None = None,
    max_rows: int = 5000,
) -> pd.DataFrame:
    df = load_analysis_dataframe(csv_path, gu_name=gu_name)
    df = df.dropna(subset=["amount_manwon", "building_area_m2", "price_per_m2_manwon"]).copy()
    df = df[df["building_area_m2"] > 0]
    if building_use and building_use != "전체":
        df = df[df["building_use"] == building_use]
    if max_rows and len(df) > max_rows:
        df = df.nlargest(max_rows, "amount_manwon")
    return df


def list_building_uses(csv_path: str, gu_name: str | None = None) -> list[str]:
    df = load_analysis_dataframe(csv_path, gu_name=gu_name)
    values = df["building_use"].dropna().astype(str).str.strip()
    return sorted(value for value in values.unique().tolist() if value)


def list_dongs(csv_path: str, gu_name: str | None = None) -> list[str]:
    df = load_analysis_dataframe(csv_path, gu_name=gu_name)
    values = df["dong"].dropna().astype(str).str.strip()
    return sorted(value for value in values.unique().tolist() if value)


def prepare_dong_use_price_dataframe(
    csv_path: str,
    gu_name: str | None = None,
    dong_name: str | None = None,
    building_use: str | None = None,
    max_rows: int = 5000,
) -> pd.DataFrame:
    df = prepare_price_area_dataframe(
        csv_path=csv_path,
        gu_name=gu_name,
        building_use=building_use,
        max_rows=0,
    )
    if dong_name and dong_name != "전체":
        df = df[df["dong"] == dong_name]
    if max_rows and len(df) > max_rows:
        df = df.nlargest(max_rows, "price_per_m2_manwon")
    df = df.copy()
    df["dong_code"] = pd.Categorical(df["dong"]).codes
    df["building_use_code"] = pd.Categorical(df["building_use"]).codes
    return df


def price_area_3d_figure(df: pd.DataFrame):
    import plotly.express as px

    fig = px.scatter_3d(
        df,
        x="building_area_m2",
        y="amount_manwon",
        z="price_per_m2_manwon",
        color="gu",
        symbol="building_use",
        hover_data={
            "contract_date": True,
            "gu": True,
            "dong": True,
            "building": True,
            "building_use": True,
            "building_area_m2": ":.2f",
            "amount_manwon": ":,.0f",
            "price_per_m2_manwon": ":,.2f",
            "price_per_pyeong_manwon": ":,.2f",
        },
        labels={
            "building_area_m2": "건물면적(㎡)",
            "amount_manwon": "거래금액(만원)",
            "price_per_m2_manwon": "면적당금액(만원/㎡)",
            "gu": "자치구",
            "building_use": "건물용도",
        },
        title="거래금액 / 건물면적 / 면적당금액 3D 시각화",
    )
    fig.update_traces(marker=dict(size=4, opacity=0.75))
    fig.update_layout(scene=dict(xaxis_title="건물면적(㎡)", yaxis_title="거래금액(만원)", zaxis_title="만원/㎡"))
    return fig


def dong_use_price_3d_figure(df: pd.DataFrame):
    import plotly.express as px

    fig = px.scatter_3d(
        df,
        x="dong_code",
        y="building_use_code",
        z="price_per_m2_manwon",
        color="building_use",
        size="amount_manwon",
        hover_data={
            "contract_date": True,
            "gu": True,
            "dong": True,
            "building": True,
            "building_use": True,
            "amount_manwon": ":,.0f",
            "building_area_m2": ":.2f",
            "price_per_m2_manwon": ":,.2f",
            "price_per_pyeong_manwon": ":,.2f",
            "dong_code": False,
            "building_use_code": False,
        },
        labels={
            "dong_code": "법정동",
            "building_use_code": "건물용도",
            "price_per_m2_manwon": "면적당금액(만원/㎡)",
            "building_use": "건물용도",
            "amount_manwon": "거래금액(만원)",
        },
        title="법정동 / 건물용도 / 면적당금액 3D 시각화",
    )

    dong_categories = list(pd.Categorical(df["dong"]).categories)
    use_categories = list(pd.Categorical(df["building_use"]).categories)
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="법정동",
                tickmode="array",
                tickvals=list(range(len(dong_categories))),
                ticktext=dong_categories,
            ),
            yaxis=dict(
                title="건물용도",
                tickmode="array",
                tickvals=list(range(len(use_categories))),
                ticktext=use_categories,
            ),
            zaxis_title="면적당금액(만원/㎡)",
        )
    )
    fig.update_traces(marker=dict(opacity=0.75))
    return fig


def dong_use_price_heatmap(df: pd.DataFrame):
    import plotly.express as px

    pivot = (
        df.pivot_table(
            index="dong",
            columns="building_use",
            values="price_per_m2_manwon",
            aggfunc="median",
        )
        .sort_index()
    )
    return px.imshow(
        pivot,
        aspect="auto",
        labels=dict(x="건물용도", y="법정동", color="중앙값 만원/㎡"),
        title="법정동/건물용도별 면적당금액 중앙값",
    )


def price_per_m2_box_figure(df: pd.DataFrame):
    import plotly.express as px

    return px.box(
        df,
        x="gu",
        y="price_per_m2_manwon",
        color="building_use",
        points="outliers",
        labels={
            "gu": "자치구",
            "price_per_m2_manwon": "면적당금액(만원/㎡)",
            "building_use": "건물용도",
        },
        title="자치구/용도별 면적당금액 분포",
    )
