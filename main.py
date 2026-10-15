import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간 끝
MIN_DAYS = 300        # 연간 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")
st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온에 직선(선형 회귀)을 맞춰 원하는 해의 기온을 예상합니다.")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]  # 2025년 이후 제외

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS]  # 관측일 300일 미만 해 제외
    return yearly.reset_index(drop=True)


try:
    yearly = load_yearly(DATA_URL)
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.warning("회귀 직선을 만들기에 데이터가 부족합니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# 최근 20년(끝 연도 기준 20개 달력 연도) 회귀
RECENT = 20
recent = yearly[yearly["연도"] > end_year - RECENT]
slope_r = None
if len(recent) >= 3:
    xr = recent["연도"].to_numpy(dtype=float)
    yr = recent["연평균기온"].to_numpy(dtype=float)
    slope_r, intercept_r = np.polyfit(xr, yr, 1)
    r_recent = float(np.corrcoef(xr, yr)[0, 1])

# 기울기: 100년에 몇 도 오르는가
st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem; margin:0.5rem 0 1rem 0;
                border-radius:16px; background:rgba(214,39,40,0.12);">
        <div style="font-size:1.1rem; opacity:0.8;">전체 기간({start_year}~{end_year}년) 기준, 100년에</div>
        <div style="font-size:4rem; font-weight:800; line-height:1.2;">{slope*100:+.2f} ℃</div>
        <div style="font-size:1rem; opacity:0.8;">오르는 속도 (기울기 × 100)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("기울기 비교: 전체 기간 vs 최근 20년")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        f"""
        <div style="text-align:center; padding:1rem; border-radius:12px; background:rgba(31,119,180,0.12);">
            <div style="opacity:0.8;">전체 기간 ({start_year}~{end_year}년, {n_years}개 해)</div>
            <div style="font-size:2.6rem; font-weight:800;">{slope*100:+.2f} ℃ / 100년</div>
            <div style="opacity:0.8;">상관계수 r = {r:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    if slope_r is not None:
        r_start, r_end = int(recent["연도"].min()), int(recent["연도"].max())
        st.markdown(
            f"""
            <div style="text-align:center; padding:1rem; border-radius:12px; background:rgba(255,127,14,0.14);">
                <div style="opacity:0.8;">최근 20년 ({r_start}~{r_end}년, {len(recent)}개 해)</div>
                <div style="font-size:2.6rem; font-weight:800;">{slope_r*100:+.2f} ℃ / 100년</div>
                <div style="opacity:0.8;">상관계수 r = {r_recent:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("최근 20년 구간의 데이터가 부족해 기울기를 구할 수 없습니다.")

if slope_r is not None:
    diff = (slope_r - slope) * 100
    st.write(
        f"최근 20년의 기울기는 전체 기간보다 100년 환산 **{diff:+.2f}℃** "
        f"{'더 가파릅니다' if diff > 0 else '더 완만합니다'}. "
        "다만 20개 해만으로 구한 기울기는 해마다의 들쭉날쭉함에 크게 흔들리므로, "
        "100년 환산 값은 속도를 비교하기 위한 단위일 뿐 실제 예측값은 아닙니다."
    )

# 슬라이더
year = st.slider("예상할 연도를 고르세요", min_value=1900, max_value=2100, value=2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem; margin:0.5rem 0 1rem 0;
                border-radius:16px; background:rgba(255,120,60,0.12);">
        <div style="font-size:1.1rem; opacity:0.8;">{year}년 예상 연평균기온</div>
        <div style="font-size:4rem; font-weight:800; line-height:1.2;">{pred:.2f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수 (r)", f"{r:.3f}")

st.write(
    f"회귀식(전체 기간): **연평균기온 = {slope:.4f} × 연도 + ({intercept:.2f})**"
)

# 그래프
x_line = np.array([1900, 2100], dtype=float)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=yearly["연도"], y=yearly["연평균기온"], mode="markers",
    name="연평균기온", marker=dict(size=8, color="#1f77b4"),
    hovertemplate="%{x}년<br>%{y:.2f}℃<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=x_line, y=slope * x_line + intercept, mode="lines",
    name="회귀 직선", line=dict(color="#d62728", width=3),
    hoverinfo="skip",
))
if slope_r is not None:
    xr_line = np.array([recent["연도"].min(), recent["연도"].max()], dtype=float)
    fig.add_trace(go.Scatter(
        x=xr_line, y=slope_r * xr_line + intercept_r, mode="lines",
        name="최근 20년 직선", line=dict(color="#ff7f0e", width=3, dash="dash"),
        hoverinfo="skip",
    ))
fig.add_trace(go.Scatter(
    x=[year], y=[pred], mode="markers",
    name=f"{year}년 예상", marker=dict(size=16, color="gold", symbol="star",
                                      line=dict(color="black", width=1.5)),
    hovertemplate=f"{year}년 예상<br>{pred:.2f}℃<extra></extra>",
))
fig.update_layout(
    title=f"서울 연평균기온 산점도와 회귀 직선 (상관계수 r = {r:.3f})",
    xaxis_title="연도", yaxis_title="연평균기온 (℃)",
    xaxis=dict(range=[1900, 2100]),
    height=520, legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("연도별 평균기온 표 보기"):
    st.dataframe(
        yearly.rename(columns={"관측일수": "관측일수(일)"}).round({"연평균기온": 2}),
        use_container_width=True, hide_index=True,
    )

st.caption(
    f"기준: {LAST_YEAR}년까지의 데이터만 사용했고, 관측일이 {MIN_DAYS}일 미만인 해는 제외했습니다. "
    "먼 미래나 과거 값은 직선을 늘려 얻은 단순 추정이므로 참고용으로만 보세요."
)
