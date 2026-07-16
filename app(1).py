from pathlib import Path
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="종로구 공공시설 태양광 설치현황",
    page_icon="☀️",
    layout="wide",
)

DEFAULT_DATA = Path(__file__).with_name("data.csv")
CAPACITY_COL = "설치용량(킬로와트)"
YEAR_COL = "설치년도"


def read_csv_safely(file):
    """한글 CSV에서 자주 쓰이는 인코딩을 순서대로 시도합니다."""
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    raw = file.read() if hasattr(file, "read") else Path(file).read_bytes()

    for encoding in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("CSV 파일에 데이터가 없습니다.")

    raise ValueError("CSV 인코딩을 판별하지 못했습니다.")


def clean_data(raw_df):
    """열 이름과 자료형을 정리하고 분석 가능한 행만 남깁니다."""
    df = raw_df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    required = {"시설명", CAPACITY_COL, YEAR_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "필수 열이 없습니다: " + ", ".join(sorted(missing))
            + "\n필요한 열: 시설명, 설치용량(킬로와트), 설치년도"
        )

    text_cols = ["시설명", "도로명 주소", "지번 주소", "발전유형"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    df[CAPACITY_COL] = (
        df[CAPACITY_COL]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    )
    df[CAPACITY_COL] = pd.to_numeric(df[CAPACITY_COL], errors="coerce")
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64")

    if "기준일자" in df.columns:
        df["기준일자"] = pd.to_datetime(df["기준일자"], errors="coerce")

    df = df.dropna(subset=["시설명", CAPACITY_COL, YEAR_COL])
    df = df[df[CAPACITY_COL] >= 0].copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)

    bins = [-np.inf, 5, 10, 20, 30, np.inf]
    labels = ["5kW 이하", "5~10kW", "10~20kW", "20~30kW", "30kW 초과"]
    df["용량구간"] = pd.cut(
        df[CAPACITY_COL], bins=bins, labels=labels, include_lowest=True
    )

    return df


def make_insights(df):
    """현재 필터 결과를 바탕으로 자동 인사이트를 생성합니다."""
    if df.empty:
        return ["선택한 조건에 해당하는 데이터가 없습니다."]

    total = df[CAPACITY_COL].sum()
    avg = df[CAPACITY_COL].mean()
    median = df[CAPACITY_COL].median()

    top_row = df.loc[df[CAPACITY_COL].idxmax()]
    yearly = (
        df.groupby(YEAR_COL, as_index=False)
        .agg(설치건수=("시설명", "count"), 신규설치용량=(CAPACITY_COL, "sum"))
    )
    best_capacity_year = yearly.loc[yearly["신규설치용량"].idxmax()]
    best_count_year = yearly.loc[yearly["설치건수"].idxmax()]

    top_n = min(5, len(df))
    top_share = (
        df.nlargest(top_n, CAPACITY_COL)[CAPACITY_COL].sum() / total * 100
        if total > 0 else 0
    )

    recent_year = int(df[YEAR_COL].max())
    recent = df[df[YEAR_COL] == recent_year]
    recent_avg = recent[CAPACITY_COL].mean()

    insights = [
        f"분석 대상은 총 {len(df):,}개 시설이며, 합계 설치용량은 {total:,.2f}kW입니다.",
        f"시설당 평균 설치용량은 {avg:,.2f}kW, 중앙값은 {median:,.2f}kW입니다. "
        + ("평균이 중앙값보다 높아 일부 대용량 시설이 평균을 끌어올립니다."
           if avg > median else "평균과 중앙값의 차이가 크지 않아 용량 분포가 비교적 고른 편입니다."),
        f"최대 설치시설은 '{top_row['시설명']}'이며 설치용량은 "
        f"{top_row[CAPACITY_COL]:,.2f}kW, 설치연도는 {int(top_row[YEAR_COL])}년입니다.",
        f"연간 신규 설치용량이 가장 컸던 해는 {int(best_capacity_year[YEAR_COL])}년으로 "
        f"{best_capacity_year['신규설치용량']:,.2f}kW이며, 설치 건수가 가장 많았던 해는 "
        f"{int(best_count_year[YEAR_COL])}년으로 {int(best_count_year['설치건수'])}건입니다.",
        f"설치용량 상위 {top_n}개 시설이 전체 용량의 {top_share:,.1f}%를 차지합니다. "
        + ("대형 시설에 설비가 집중된 구조입니다." if top_share >= 50
           else "설비가 여러 시설에 비교적 분산되어 있습니다."),
        f"가장 최근 설치연도는 {recent_year}년이며, 해당 연도 {len(recent)}개 시설의 "
        f"평균 설치용량은 {recent_avg:,.2f}kW입니다.",
    ]
    return insights


def apply_chart_layout(fig, title):
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text="",
        hovermode="closest",
    )
    return fig


st.title("☀️ 종로구 공공시설 태양광 설치현황 분석")
st.caption("CSV 데이터를 자동 전처리하고 Plotly로 시각화하는 Streamlit 대시보드")

with st.sidebar:
    st.header("데이터 및 필터")
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    st.caption("업로드하지 않으면 앱에 포함된 기본 데이터가 사용됩니다.")

try:
    selected_file = uploaded_file if uploaded_file is not None else DEFAULT_DATA
    raw_df, used_encoding = read_csv_safely(selected_file)
    df = clean_data(raw_df)
except Exception as exc:
    st.error(f"데이터를 불러오지 못했습니다.\n\n{exc}")
    st.stop()

if df.empty:
    st.warning("분석 가능한 데이터가 없습니다.")
    st.stop()

with st.sidebar:
    min_year = int(df[YEAR_COL].min())
    max_year = int(df[YEAR_COL].max())
    year_range = st.slider(
        "설치연도 범위",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )

    min_capacity = float(df[CAPACITY_COL].min())
    max_capacity = float(df[CAPACITY_COL].max())
    capacity_range = st.slider(
        "설치용량 범위(kW)",
        min_value=min_capacity,
        max_value=max_capacity,
        value=(min_capacity, max_capacity),
        step=0.1,
    )

    if "발전유형" in df.columns:
        types = sorted(df["발전유형"].dropna().unique().tolist())
        selected_types = st.multiselect(
            "발전유형",
            options=types,
            default=types,
        )
    else:
        selected_types = None

filtered = df[
    df[YEAR_COL].between(year_range[0], year_range[1])
    & df[CAPACITY_COL].between(capacity_range[0], capacity_range[1])
].copy()

if selected_types is not None:
    filtered = filtered[filtered["발전유형"].isin(selected_types)]

with st.sidebar:
    st.divider()
    st.write(f"CSV 인코딩: `{used_encoding}`")
    st.write(f"전체 {len(df):,}행 / 선택 {len(filtered):,}행")

if filtered.empty:
    st.warning("현재 필터 조건에 해당하는 시설이 없습니다.")
    st.stop()

total_capacity = filtered[CAPACITY_COL].sum()
avg_capacity = filtered[CAPACITY_COL].mean()
median_capacity = filtered[CAPACITY_COL].median()
max_row = filtered.loc[filtered[CAPACITY_COL].idxmax()]

k1, k2, k3, k4 = st.columns(4)
k1.metric("시설 수", f"{len(filtered):,}개")
k2.metric("총 설치용량", f"{total_capacity:,.2f} kW")
k3.metric("평균 설치용량", f"{avg_capacity:,.2f} kW")
k4.metric("최대 설치용량", f"{max_row[CAPACITY_COL]:,.2f} kW")

st.caption(
    f"최대 설치시설: {max_row['시설명']} · 중앙값: {median_capacity:,.2f}kW"
)

yearly = (
    filtered.groupby(YEAR_COL, as_index=False)
    .agg(
        설치건수=("시설명", "count"),
        신규설치용량=(CAPACITY_COL, "sum"),
        평균설치용량=(CAPACITY_COL, "mean"),
    )
    .sort_values(YEAR_COL)
)
yearly["누적설치용량"] = yearly["신규설치용량"].cumsum()

tab1, tab2, tab3, tab4 = st.tabs(
    ["연도별 추세", "시설별 비교", "용량 분포", "인사이트·데이터"]
)

with tab1:
    left, right = st.columns(2)

    with left:
        fig_year = px.bar(
            yearly,
            x=YEAR_COL,
            y="신규설치용량",
            text_auto=".2f",
            hover_data=["설치건수", "평균설치용량"],
            labels={
                YEAR_COL: "설치연도",
                "신규설치용량": "신규 설치용량(kW)",
                "설치건수": "설치 건수",
                "평균설치용량": "평균 설치용량(kW)",
            },
        )
        st.plotly_chart(
            apply_chart_layout(fig_year, "연도별 신규 설치용량"),
            use_container_width=True,
        )

    with right:
        fig_cum = px.line(
            yearly,
            x=YEAR_COL,
            y="누적설치용량",
            markers=True,
            labels={
                YEAR_COL: "설치연도",
                "누적설치용량": "누적 설치용량(kW)",
            },
        )
        fig_cum.update_traces(line_width=3, marker_size=9)
        st.plotly_chart(
            apply_chart_layout(fig_cum, "연도별 누적 설치용량"),
            use_container_width=True,
        )

    fig_combo = go.Figure()
    fig_combo.add_bar(
        x=yearly[YEAR_COL],
        y=yearly["설치건수"],
        name="설치 건수",
        yaxis="y",
    )
    fig_combo.add_scatter(
        x=yearly[YEAR_COL],
        y=yearly["평균설치용량"],
        name="평균 설치용량",
        mode="lines+markers",
        yaxis="y2",
    )
    fig_combo.update_layout(
        title="연도별 설치 건수와 평균 설치용량",
        xaxis_title="설치연도",
        yaxis=dict(title="설치 건수"),
        yaxis2=dict(
            title="평균 설치용량(kW)",
            overlaying="y",
            side="right",
        ),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig_combo, use_container_width=True)

with tab2:
    ranking = filtered.sort_values(CAPACITY_COL, ascending=True)
    fig_rank = px.bar(
        ranking,
        x=CAPACITY_COL,
        y="시설명",
        orientation="h",
        hover_data=[YEAR_COL, "도로명 주소"] if "도로명 주소" in ranking.columns else [YEAR_COL],
        labels={
            CAPACITY_COL: "설치용량(kW)",
            "시설명": "시설명",
            YEAR_COL: "설치연도",
        },
    )
    fig_rank.update_layout(height=max(520, len(ranking) * 28))
    st.plotly_chart(
        apply_chart_layout(fig_rank, "시설별 설치용량 순위"),
        use_container_width=True,
    )

    scatter_hover = ["시설명"]
    if "도로명 주소" in filtered.columns:
        scatter_hover.append("도로명 주소")

    fig_scatter = px.scatter(
        filtered,
        x=YEAR_COL,
        y=CAPACITY_COL,
        size=CAPACITY_COL,
        hover_name="시설명",
        hover_data=scatter_hover,
        labels={
            YEAR_COL: "설치연도",
            CAPACITY_COL: "설치용량(kW)",
        },
        size_max=35,
    )
    st.plotly_chart(
        apply_chart_layout(fig_scatter, "설치연도와 시설별 설치용량"),
        use_container_width=True,
    )

with tab3:
    left, right = st.columns(2)

    with left:
        fig_hist = px.histogram(
            filtered,
            x=CAPACITY_COL,
            nbins=min(10, max(5, len(filtered) // 2)),
            labels={CAPACITY_COL: "설치용량(kW)", "count": "시설 수"},
        )
        st.plotly_chart(
            apply_chart_layout(fig_hist, "설치용량 분포"),
            use_container_width=True,
        )

    with right:
        range_data = (
            filtered.groupby("용량구간", observed=False)
            .size()
            .reset_index(name="시설수")
        )
        range_data = range_data[range_data["시설수"] > 0]
        fig_pie = px.pie(
            range_data,
            names="용량구간",
            values="시설수",
            hole=0.45,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(
            apply_chart_layout(fig_pie, "설치용량 구간별 시설 구성"),
            use_container_width=True,
        )

    fig_box = px.box(
        filtered,
        x=YEAR_COL,
        y=CAPACITY_COL,
        points="all",
        hover_name="시설명",
        labels={
            YEAR_COL: "설치연도",
            CAPACITY_COL: "설치용량(kW)",
        },
    )
    st.plotly_chart(
        apply_chart_layout(fig_box, "연도별 설치용량 분포"),
        use_container_width=True,
    )

with tab4:
    st.subheader("자동 도출 핵심 인사이트")
    for index, insight in enumerate(make_insights(filtered), start=1):
        st.markdown(f"**{index}.** {insight}")

    st.info(
        "정책적 활용을 위해서는 설치용량뿐 아니라 실제 발전량, 가동률, "
        "일사량, 유지보수비, 전력 절감액, 탄소감축량 데이터를 함께 관리하는 것이 좋습니다."
    )

    st.subheader("분석 데이터")
    display_df = filtered.sort_values(
        [YEAR_COL, CAPACITY_COL], ascending=[True, False]
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_bytes = display_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "필터링 결과 CSV 다운로드",
        data=csv_bytes,
        file_name="종로구_태양광_필터링결과.csv",
        mime="text/csv",
    )
