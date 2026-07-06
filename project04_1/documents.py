from __future__ import annotations

from typing import Any

import pandas as pd


def clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_number(value: Any) -> float | None:
    text = clean_value(value).replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_date(value: Any) -> str:
    text = clean_value(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def format_price(amount_manwon: float | None) -> str:
    if amount_manwon is None:
        return "미상"
    eok = amount_manwon / 10000
    return f"{amount_manwon:,.0f}만원 ({eok:,.2f}억원)"


def calculate_price_per_m2(amount_manwon: float | None, building_area: float | None) -> float | None:
    if amount_manwon is None or building_area is None or building_area <= 0:
        return None
    return amount_manwon / building_area


def format_price_per_m2(value: float | None) -> str:
    if value is None:
        return "미상"
    pyeong_value = value * 3.305785
    return f"{value:,.2f}만원/㎡ ({pyeong_value:,.2f}만원/평)"


def row_to_document(row: pd.Series) -> tuple[str, dict[str, str | int | float]]:
    amount = parse_number(row.get("물건금액(만원)"))
    building_area = parse_number(row.get("건물면적(㎡)"))
    land_area = parse_number(row.get("토지면적(㎡)"))
    price_per_m2 = calculate_price_per_m2(amount, building_area)
    contract_date = format_date(row.get("계약일"))

    gu = clean_value(row.get("자치구명"))
    dong = clean_value(row.get("법정동명"))
    main_no = clean_value(row.get("본번")).lstrip("0")
    sub_no = clean_value(row.get("부번")).lstrip("0")
    lot_no = "-".join(part for part in [main_no, sub_no] if part)
    building = clean_value(row.get("건물명"))
    building_use = clean_value(row.get("건물용도"))
    floor = clean_value(row.get("층"))
    built_year = clean_value(row.get("건축년도"))
    deal_type = clean_value(row.get("신고구분"))
    realtor_area = clean_value(row.get("신고한 개업공인중개사 시군구명"))

    location_parts = [part for part in [gu, dong, lot_no, building] if part]
    location = " ".join(location_parts) if location_parts else "위치 미상"

    text = "\n".join(
        [
            "서울시 부동산 실거래가 기록",
            f"계약일: {contract_date}",
            f"위치: {location}",
            f"자치구: {gu}",
            f"법정동: {dong}",
            f"건물명: {building or '미상'}",
            f"건물용도: {building_use or '미상'}",
            f"거래금액: {format_price(amount)}",
            f"건물면적: {building_area if building_area is not None else '미상'}㎡",
            f"면적당금액: {format_price_per_m2(price_per_m2)}",
            f"토지면적: {land_area if land_area is not None else '미상'}㎡",
            f"층: {floor or '미상'}",
            f"건축년도: {built_year or '미상'}",
            f"거래유형: {deal_type or '미상'}",
            f"중개사 소재지: {realtor_area or '미상'}",
        ]
    )

    metadata: dict[str, str | int | float] = {
        "contract_date": contract_date,
        "gu": gu,
        "dong": dong,
        "building": building,
        "building_use": building_use,
        "deal_type": deal_type,
        "floor": floor,
        "built_year": built_year,
    }
    if amount is not None:
        metadata["amount_manwon"] = amount
    if building_area is not None:
        metadata["building_area_m2"] = building_area
    if price_per_m2 is not None:
        metadata["price_per_m2_manwon"] = round(price_per_m2, 4)
        metadata["price_per_pyeong_manwon"] = round(price_per_m2 * 3.305785, 4)
    if land_area is not None:
        metadata["land_area_m2"] = land_area

    return text, metadata


def read_csv_with_fallback(csv_path: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(csv_path, dtype=str, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(f"CSV 인코딩을 판별하지 못했습니다. 마지막 오류: {last_error}")


def filter_by_gu(df: pd.DataFrame, gu_name: str | None) -> pd.DataFrame:
    if not gu_name or gu_name == "전체":
        return df
    if "자치구명" not in df.columns:
        raise KeyError("CSV에 '자치구명' 컬럼이 없습니다.")
    return df[df["자치구명"].fillna("").astype(str).str.strip() == gu_name].copy()


def list_gu_names(csv_path: str) -> list[str]:
    df = read_csv_with_fallback(csv_path)
    if "자치구명" not in df.columns:
        raise KeyError("CSV에 '자치구명' 컬럼이 없습니다.")
    names = df["자치구명"].dropna().astype(str).str.strip()
    return sorted(name for name in names.unique().tolist() if name)


def dataframe_to_documents(
    df: pd.DataFrame,
    chunk_size: int = 1,
) -> tuple[list[str], list[dict[str, str | int | float]], list[str]]:
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float]] = []
    ids: list[str] = []
    chunk_size = max(1, int(chunk_size))

    indexed_rows = list(df.iterrows())
    for start in range(0, len(indexed_rows), chunk_size):
        chunk = indexed_rows[start : start + chunk_size]
        row_indexes = [int(idx) for idx, _ in chunk]
        chunk_documents: list[str] = []
        chunk_metadatas: list[dict[str, str | int | float]] = []

        for idx, row in chunk:
            document, metadata = row_to_document(row)
            metadata["row_index"] = int(idx)
            chunk_documents.append(document)
            chunk_metadatas.append(metadata)

        if chunk_size == 1:
            documents.append(chunk_documents[0])
            metadatas.append(chunk_metadatas[0])
            ids.append(f"deal-{row_indexes[0]}")
            continue

        first_meta = chunk_metadatas[0]
        price_values = [
            float(meta["price_per_m2_manwon"])
            for meta in chunk_metadatas
            if "price_per_m2_manwon" in meta
        ]
        metadata: dict[str, str | int | float] = {
            "row_start": row_indexes[0],
            "row_end": row_indexes[-1],
            "row_count": len(row_indexes),
            "chunk_size": chunk_size,
            "gu": first_meta.get("gu", ""),
            "dong": first_meta.get("dong", ""),
            "building_use": first_meta.get("building_use", ""),
        }
        if price_values:
            avg_price = sum(price_values) / len(price_values)
            metadata["avg_price_per_m2_manwon"] = round(avg_price, 4)
            metadata["min_price_per_m2_manwon"] = round(min(price_values), 4)
            metadata["max_price_per_m2_manwon"] = round(max(price_values), 4)
            metadata["avg_price_per_pyeong_manwon"] = round(avg_price * 3.305785, 4)
        documents.append("\n\n---\n\n".join(chunk_documents))
        metadatas.append(metadata)
        ids.append(f"deal-chunk-{row_indexes[0]}-{row_indexes[-1]}")

    return documents, metadatas, ids


def load_csv_documents(
    csv_path: str,
    gu_name: str | None = None,
    chunk_size: int = 1,
) -> tuple[list[str], list[dict[str, str | int | float]], list[str]]:
    df = read_csv_with_fallback(csv_path)
    df = filter_by_gu(df, gu_name)
    return dataframe_to_documents(df, chunk_size=chunk_size)


def load_analysis_dataframe(csv_path: str, gu_name: str | None = None) -> pd.DataFrame:
    df = read_csv_with_fallback(csv_path)
    df = filter_by_gu(df, gu_name)
    df = df.copy()
    df["amount_manwon"] = df["물건금액(만원)"].map(parse_number)
    df["building_area_m2"] = df["건물면적(㎡)"].map(parse_number)
    df["price_per_m2_manwon"] = [
        calculate_price_per_m2(amount, area)
        for amount, area in zip(df["amount_manwon"], df["building_area_m2"])
    ]
    df["price_per_pyeong_manwon"] = df["price_per_m2_manwon"].map(
        lambda value: value * 3.305785 if value is not None else None
    )
    df["contract_date"] = df["계약일"].map(format_date)
    df["gu"] = df["자치구명"].fillna("").astype(str).str.strip()
    df["dong"] = df["법정동명"].fillna("").astype(str).str.strip()
    df["building"] = df["건물명"].fillna("").astype(str).str.strip()
    df["building_use"] = df["건물용도"].fillna("").astype(str).str.strip()
    return df
