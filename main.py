import io
import json
import zipfile

import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="전국 고령화 단계구분도",
    page_icon="🗺️",
    layout="wide",
)


# =========================================================
# 데이터 주소
# =========================================================

POPULATION_URL = (
    "https://raw.githubusercontent.com/first2022-cyber/contents/"
    "/main/population_ts_yearly_gender.zip"
)

GEOJSON_URL = (
    "https://raw.githubusercontent.com/first2022-cyber/contents/"
    "/main/sigungu_kr.geojson"
)


# =========================================================
# 색상 설정
# =========================================================

# 고령화율
# 19% 미만
# 19~23%
# 23~28%
# 28~38%
# 38% 이상

AGING_BINS = [0, 19, 23, 28, 38, 100]

AGING_LABELS = [
    "19% 미만",
    "19~23%",
    "23~28%",
    "28~38%",
    "38% 이상",
]


# 유소년 비율
# 5% 미만
# 5~10%
# 10~15%
# 15~20%
# 20% 이상

YOUTH_BINS = [0, 5, 10, 15, 20, 100]

YOUTH_LABELS = [
    "5% 미만",
    "5~10%",
    "10~15%",
    "15~20%",
    "20% 이상",
]


# 단계별 색상
COLOR_SCALE = [
    "#fff5eb",
    "#fdcc8a",
    "#fc8d59",
    "#e34a33",
    "#b30000",
]

MISSING_COLOR = "#d9d9d9"


# =========================================================
# 데이터 다운로드
# =========================================================

@st.cache_data
def load_population():
    response = requests.get(
        POPULATION_URL,
        timeout=120,
    )

    response.raise_for_status()

    zip_data = io.BytesIO(response.content)

    with zipfile.ZipFile(zip_data) as z:
        csv_files = [
            name
            for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise ValueError(
                "ZIP 파일 안에서 CSV 파일을 찾을 수 없습니다."
            )

        # 첫 번째 CSV 사용
        csv_name = csv_files[0]

        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    return df


# =========================================================
# GeoJSON 다운로드
# =========================================================

@st.cache_data
def load_geojson():
    response = requests.get(
        GEOJSON_URL,
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


# =========================================================
# GeoJSON 코드 정리
# =========================================================

@st.cache_data
def normalize_geojson(geojson):
    # 원본을 직접 변경하지 않도록 복사
    geojson = json.loads(json.dumps(geojson))

    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})

        code = properties.get("코드")

        if code is not None:
            properties["코드"] = (
                str(code)
                .strip()
                .zfill(5)
            )

    return geojson


# =========================================================
# 인구 데이터 전처리
# =========================================================

@st.cache_data
def prepare_population(df):
    df = df.copy()

    # -----------------------------------------------------
    # 연도 숫자화
    # -----------------------------------------------------

    df["연도"] = pd.to_numeric(
        df["연도"],
        errors="coerce",
    )

    # -----------------------------------------------------
    # 코드 문자열 처리
    # -----------------------------------------------------

    df["코드"] = (
        df["코드"]
        .astype("string")
        .str.strip()
        .str.zfill(10)
    )

    # -----------------------------------------------------
    # 행정구역 코드 개편 보정
    #
    # 옛 시도 코드
    # 42 → 51 (강원)
    # 45 → 52 (전북)
    #
    # 군위군
    # 47720 → 27720
    # -----------------------------------------------------

    df["시군구코드"] = (
        df["코드"]
        .str[:5]
        .astype("string")
    )

    # 시군구 코드의 앞 2자리를 변경

    mask_42 = df["시군구코드"].str[:2] == "42"

    df.loc[mask_42, "시군구코드"] = (
        "51"
        + df.loc[mask_42, "시군구코드"].str[2:]
    )

    mask_45 = df["시군구코드"].str[:2] == "45"

    df.loc[mask_45, "시군구코드"] = (
        "52"
        + df.loc[mask_45, "시군구코드"].str[2:]
    )

    # 군위군
    df.loc[
        df["시군구코드"] == "47720",
        "시군구코드",
    ] = "27720"

    return df


# =========================================================
# 나이별 인구 열 찾기
# =========================================================

@st.cache_data
def get_age_columns(columns):
    total_columns = [
        col
        for col in columns
        if str(col).startswith("계_")
    ]

    youth_columns = []
    senior_columns = []

    for col in total_columns:
        age_text = str(col).replace(
            "계_",
            "",
            1,
        ).strip()

        # -----------------------------------------------
        # 100세 이상
        # -----------------------------------------------

        if age_text == "100세 이상":
            age = 100

        else:
            try:
                age = int(
                    age_text
                    .replace("세", "")
                    .strip()
                )

            except ValueError:
                continue

        # 0~14세
        if 0 <= age <= 14:
            youth_columns.append(col)

        # 65세 이상
        if age >= 65:
            senior_columns.append(col)

    return (
        total_columns,
        youth_columns,
        senior_columns,
    )


# =========================================================
# 연도별 시군구 지표 계산
# =========================================================

@st.cache_data
def calculate_indicators(df):
    df = df.copy()

    (
        total_columns,
        youth_columns,
        senior_columns,
    ) = get_age_columns(tuple(df.columns))

    if not total_columns:
        raise ValueError(
            "'계_'로 시작하는 인구 열을 찾을 수 없습니다."
        )

    if not youth_columns:
        raise ValueError(
            "0~14세 인구 열을 찾을 수 없습니다."
        )

    if not senior_columns:
        raise ValueError(
            "65세 이상 인구 열을 찾을 수 없습니다."
        )

    # 숫자로 변환
    for col in total_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0)

    # -----------------------------------------------------
    # 전체 인구
    # -----------------------------------------------------

    df["전체인구"] = df[total_columns].sum(axis=1)

    # -----------------------------------------------------
    # 65세 이상
    # -----------------------------------------------------

    df["65세이상인구"] = (
        df[senior_columns].sum(axis=1)
    )

    # -----------------------------------------------------
    # 0~14세
    # -----------------------------------------------------

    df["유소년인구"] = (
        df[youth_columns].sum(axis=1)
    )

    # -----------------------------------------------------
    # 시군구별 집계
    # -----------------------------------------------------

    result = (
        df.groupby(
            [
                "연도",
                "시군구코드",
            ],
            as_index=False,
        )
        .agg(
            전체인구=("전체인구", "sum"),
            65세이상인구=("65세이상인구", "sum"),
            유소년인구=("유소년인구", "sum"),
        )
    )

    # -----------------------------------------------------
    # 비율 계산
    # -----------------------------------------------------

    result["고령화율"] = (
        result["65세이상인구"]
        / result["전체인구"]
        * 100
    )

    result["유소년율"] = (
        result["유소년인구"]
        / result["전체인구"]
        * 100
    )

    # 무한대 / 결측 제거
    result["고령화율"] = (
        result["고령화율"]
        .replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )
    )

    result["유소년율"] = (
        result["유소년율"]
        .replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )
    )

    # 코드 문자열 유지
    result["시군구코드"] = (
        result["시군구코드"]
        .astype("string")
        .str.zfill(5)
    )

    return result


# =========================================================
# GeoJSON 지역 정보
# =========================================================

@st.cache_data
def get_geo_info(geojson):
    rows = []

    for feature in geojson.get("features", []):
        properties = feature.get(
            "properties",
            {},
        )

        code = properties.get("코드")

        if code is None:
            continue

        code = (
            str(code)
            .strip()
            .zfill(5)
        )

        rows.append(
            {
                "시군구코드": code,
                "시군구": properties.get(
                    "시군구",
                    "",
                ),
                "시도": properties.get(
                    "시도",
                    "",
                ),
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# 단계 구간 만들기
# =========================================================

def make_stage(value, metric):
    if pd.isna(value):
        return "자료 없음"

    if metric == "고령화율":
        bins = AGING_BINS
        labels = AGING_LABELS

    else:
        bins = YOUTH_BINS
        labels = YOUTH_LABELS

    for i in range(len(labels)):
        if i == len(labels) - 1:
            if value >= bins[i]:
                return labels[i]

        else:
            if bins[i] <= value < bins[i + 1]:
                return labels[i]

    return "자료 없음"


# =========================================================
# 지도용 GeoJSON에 시도/시군구 정보 연결
# =========================================================

def filter_geojson(
    geojson,
    selected_province,
):
    if selected_province == "전국":
        return geojson

    selected_features = []

    for feature in geojson.get("features", []):
        properties = feature.get(
            "properties",
            {},
        )

        if properties.get("시도") == selected_province:
            selected_features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": selected_features,
    }


# =========================================================
# 지도 만들기
# =========================================================

def create_map(
    map_df,
    geojson,
    selected_province,
    metric,
):
    # -----------------------------------------------------
    # 현재 선택 지역의 GeoJSON
    # -----------------------------------------------------

    display_geojson = filter_geojson(
        geojson,
        selected_province,
    )

    # -----------------------------------------------------
    # 지도에 표시되는 GeoJSON 코드
    # -----------------------------------------------------

    geo_codes = []

    for feature in display_geojson.get(
        "features",
        [],
    ):
        properties = feature.get(
            "properties",
            {},
        )

        code = properties.get("코드")

        if code is not None:
            geo_codes.append(
                str(code)
                .strip()
                .zfill(5)
            )

    geo_codes = list(
        dict.fromkeys(geo_codes)
    )

    # -----------------------------------------------------
    # 자료가 있는 지역
    # -----------------------------------------------------

    matched_df = map_df[
        map_df["시군구코드"].isin(geo_codes)
    ].copy()

    # -----------------------------------------------------
    # 값이 없는 지역
    # -----------------------------------------------------

    matched_codes = set(
        matched_df["시군구코드"]
        .astype(str)
    )

    missing_codes = [
        code
        for code in geo_codes
        if code not in matched_codes
    ]

    # -----------------------------------------------------
    # 지도 색상 단계
    # -----------------------------------------------------

    if metric == "고령화율":
        value_column = "고령화율"
        labels = AGING_LABELS
        colors = COLOR_SCALE

    else:
        value_column = "유소년율"
        labels = YOUTH_LABELS
        colors = COLOR_SCALE

    # -----------------------------------------------------
    # 단계별 데이터
    # -----------------------------------------------------

    if not matched_df.empty:
        matched_df["단계"] = matched_df[
            value_column
        ].apply(
            lambda x: make_stage(
                x,
                metric,
            )
        )

    # -----------------------------------------------------
    # Figure
    # -----------------------------------------------------

    fig = go.Figure()

    # -----------------------------------------------------
    # ① 모든 행정구역을 회색으로 먼저 그린다.
    #    → 자료가 없는 지역도 회색으로 표시
    # -----------------------------------------------------

    if geo_codes:
        fig.add_trace(
            go.Choropleth(
                geojson=display_geojson,
                locations=geo_codes,
                z=[0] * len(geo_codes),
                featureidkey="properties.코드",
                colorscale=[
                    [0, MISSING_COLOR],
                    [1, MISSING_COLOR],
                ],
                showscale=False,
                marker_line_color="white",
                marker_line_width=0.6,
                hoverinfo="skip",
                name="자료 없음",
            )
        )

    # -----------------------------------------------------
    # ② 단계별로 색칠
    # -----------------------------------------------------

    for label, color in zip(
        labels,
        colors,
    ):
        stage_df = matched_df[
            matched_df["단계"] == label
        ].copy()

        if stage_df.empty:
            continue

        fig.add_trace(
            go.Choropleth(
                geojson=display_geojson,
                locations=stage_df[
                    "시군구코드"
                ],
                z=[1] * len(stage_df),
                featureidkey="properties.코드",
                colorscale=[
                    [0, color],
                    [1, color],
                ],
                showscale=False,
                marker_line_color="white",
                marker_line_width=0.6,
                customdata=stage_df[
                    [
                        "시군구",
                        value_column,
                    ]
                ].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{metric}: "
                    "%{customdata[1]:.2f}%"
                    "<extra></extra>"
                ),
                name=label,
                showlegend=True,
            )
        )

    # -----------------------------------------------------
    # 지도 설정
    # -----------------------------------------------------

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator",
    )

    fig.update_layout(
        height=720,
        margin=dict(
            l=0,
            r=0,
            t=10,
            b=0,
        ),
        legend=dict(
            title="단계",
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig, missing_codes


# =========================================================
# 앱 시작
# =========================================================

st.title("🗺️ 전국 고령화 단계구분도")

st.markdown(
    """
    2026년 6월 기준 읍·면·동 인구자료를 시군구 단위로 집계하여
    **고령화율 또는 유소년율**을 지도에 표시합니다.
    """
)


# =========================================================
# 데이터 로딩
# =========================================================

try:
    population_raw = load_population()

    population = prepare_population(
        population_raw
    )

    geojson = load_geojson()

    geojson = normalize_geojson(
        geojson
    )

    geo_info = get_geo_info(
        geojson
    )

    indicators = calculate_indicators(
        population
    )

except Exception as e:
    st.error(
        "데이터를 불러오거나 처리하는 중 오류가 발생했습니다."
    )

    st.exception(e)

    st.stop()


# =========================================================
# 선택 영역
# =========================================================

years = sorted(
    indicators["연도"]
    .dropna()
    .astype(int)
    .unique()
)

if not years:
    st.error(
        "사용할 수 있는 연도 데이터가 없습니다."
    )

    st.stop()


st.subheader("조건 선택")

control1, control2, control3 = st.columns(3)


with control1:
    selected_year = st.slider(
        "연도",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=int(max(years)),
        step=1,
    )


with control2:
    metric = st.selectbox(
        "지표",
        [
            "고령화율",
            "유소년율",
        ],
    )


with control3:
    # GeoJSON에서 시도 목록 추출

    provinces = sorted(
        geo_info["시도"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_province = st.selectbox(
        "시도",
        ["전국"] + provinces,
        index=0,
    )


# =========================================================
# 선택 연도 데이터
# =========================================================

year_df = indicators[
    indicators["연도"] == selected_year
].copy()


# =========================================================
# GeoJSON의 지역명/시도 정보와 결합
# =========================================================

year_df = year_df.merge(
    geo_info,
    on="시군구코드",
    how="left",
)

# 혹시 merge 후 시도명이 없는 경우 빈 문자열

year_df["시도"] = (
    year_df["시도"]
    .fillna("")
)

year_df["시군구"] = (
    year_df["시군구"]
    .fillna("")
)


# =========================================================
# 현재 시도 선택 적용
# =========================================================

if selected_province == "전국":
    current_df = year_df.copy()

else:
    current_df = year_df[
        year_df["시도"] == selected_province
    ].copy()


# =========================================================
# 지표 값 선택
# =========================================================

value_column = (
    "고령화율"
    if metric == "고령화율"
    else "유소년율"
)


# =========================================================
# 지도 표시용 데이터
# =========================================================

map_df = current_df[
    [
        "시군구코드",
        "시군구",
        "시도",
        "전체인구",
        "65세이상인구",
        "유소년인구",
        "고령화율",
        "유소년율",
    ]
].copy()


# =========================================================
# 전국 고령화율
# =========================================================

national_df = year_df.copy()

national_total = (
    national_df["전체인구"].sum()
)

national_senior = (
    national_df["65세이상인구"].sum()
)

national_aging_rate = (
    national_senior
    / national_total
    * 100
    if national_total > 0
    else float("nan")
)


# =========================================================
# 최고 / 최저 시군구
# =========================================================

valid_df = current_df[
    current_df[value_column].notna()
    & (current_df["전체인구"] > 0)
].copy()

if valid_df.empty:
    highest_name = "-"
    highest_value = float("nan")

    lowest_name = "-"
    lowest_value = float("nan")

else:
    highest_row = valid_df.loc[
        valid_df[value_column].idxmax()
    ]

    lowest_row = valid_df.loc[
        valid_df[value_column].idxmin()
    ]

    highest_name = (
        highest_row["시군구"]
        if highest_row["시군구"]
        else highest_row["시군구코드"]
    )

    highest_value = highest_row[
        value_column
    ]

    lowest_name = (
        lowest_row["시군구"]
        if lowest_row["시군구"]
        else lowest_row["시군구코드"]
    )

    lowest_value = lowest_row[
        value_column
    ]


# =========================================================
# 지표 카드
# =========================================================

st.subheader(
    f"{selected_year}년 {metric}"
)

card1, card2, card3 = st.columns(3)


with card1:
    st.metric(
        "전국 고령화율",
        f"{national_aging_rate:.2f}%",
    )


with card2:
    if pd.isna(highest_value):
        st.metric(
            "가장 높은 시군구",
            "-",
        )

    else:
        st.metric(
            "가장 높은 시군구",
            highest_name,
            f"{highest_value:.2f}%",
        )


with card3:
    if pd.isna(lowest_value):
        st.metric(
            "가장 낮은 시군구",
            "-",
        )

    else:
        st.metric(
            "가장 낮은 시군구",
            lowest_name,
            f"{lowest_value:.2f}%",
        )


# =========================================================
# 지도
# =========================================================

st.subheader(
    f"{selected_year}년 {selected_province} "
    f"{metric} 단계구분도"
)

fig, missing_codes = create_map(
    map_df=map_df,
    geojson=geojson,
    selected_province=selected_province,
    metric=metric,
)

st.plotly_chart(
    fig,
    width="stretch",
    config={
        "scrollZoom": True,
        "displaylogo": False,
    },
)


# =========================================================
# 매칭되지 않은 지역 안내
# =========================================================

if missing_codes:
    st.warning(
        f"⚠️ 경계 파일과 코드가 일치하지 않아 "
        f"{len(missing_codes)}개 지역은 회색으로 표시했습니다. "
        "행정구역 개편 등으로 인해 해당 연도의 옛 코드와 "
        "현재 경계 코드가 일치하지 않는 지역일 수 있습니다."
    )


# =========================================================
# 색상 구간 설명
# =========================================================

if metric == "고령화율":
    st.caption(
        "고령화율 단계: "
        "19% 미만 · 19~23% · 23~28% · "
        "28~38% · 38% 이상"
    )

else:
    st.caption(
        "유소년율 단계: "
        "5% 미만 · 5~10% · 10~15% · "
        "15~20% · 20% 이상"
    )


# =========================================================
# 데이터 표
# =========================================================

with st.expander(
    f"{selected_year}년 시군구별 데이터 보기"
):
    table = current_df[
        [
            "시군구코드",
            "시도",
            "시군구",
            "전체인구",
            "65세이상인구",
            "유소년인구",
            "고령화율",
            "유소년율",
        ]
    ].copy()

    table = table.sort_values(
        value_column,
        ascending=False,
        na_position="last",
    )

    table.columns = [
        "코드",
        "시도",
        "시군구",
        "전체 인구",
        "65세 이상 인구",
        "0~14세 인구",
        "고령화율 (%)",
        "유소년율 (%)",
    ]

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
    )
