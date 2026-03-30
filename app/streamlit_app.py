from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_dj_copilot.config import db_path_from_env
from ai_dj_copilot.storage_sqlite import connect, fetch_all_tracks, init_db


@st.cache_data(show_spinner=False)
def load_tracks(db_path: str) -> pd.DataFrame:
    conn = connect(Path(db_path))
    try:
        init_db(conn)
        return fetch_all_tracks(conn)
    finally:
        conn.close()


def _clamp_range(min_v: float, max_v: float, *, low: float, high: float) -> tuple[float, float]:
    a = max(low, min_v)
    b = min(high, max_v)
    if a > b:
        a, b = b, a
    return a, b


st.set_page_config(page_title="AI DJ Copilot", layout="wide")
st.title("AI DJ Copilot - Library Visualizer")

db_path = str(db_path_from_env().resolve())

df = load_tracks(db_path)

st.caption(f"数据源：SQLite `{db_path}`")

if df.empty:
    st.info("当前 SQLite 里没有曲目。请先运行摄取脚本：`python scripts/ingest_kpop.py --db data/tracks.db`")
    st.stop()

st.sidebar.header("筛选与排序")

# BPM filter
if "bpm" in df.columns and df["bpm"].notna().any():
    bpm_min_data = float(df["bpm"].dropna().min())
    bpm_max_data = float(df["bpm"].dropna().max())
    bpm_step = 0.1
    # Use exact min/max from data for default coverage.
    if bpm_min_data == bpm_max_data:
        bpm_slider_min = bpm_min_data - bpm_step
        bpm_slider_max = bpm_max_data + bpm_step
        bpm_range = st.sidebar.slider(
            "BPM 范围",
            min_value=bpm_slider_min,
            max_value=bpm_slider_max,
            value=(bpm_min_data, bpm_max_data),
            step=bpm_step,
        )
    else:
        bpm_range = st.sidebar.slider(
            "BPM 范围",
            min_value=bpm_min_data,
            max_value=bpm_max_data,
            value=(bpm_min_data, bpm_max_data),
            step=bpm_step,
        )
else:
    bpm_range = None
    st.sidebar.caption("BPM：暂无可用数据（可能节奏估计失败或为空）。")

# Energy filter
if "energy_score" in df.columns and df["energy_score"].notna().any():
    energy_min_data = float(df["energy_score"].dropna().min())
    energy_max_data = float(df["energy_score"].dropna().max())
    energy_step = 0.01
    if energy_min_data == energy_max_data:
        energy_slider_min = energy_min_data - energy_step
        energy_slider_max = energy_max_data + energy_step
        energy_range = st.sidebar.slider(
            "能量范围 (0..1)",
            min_value=energy_slider_min,
            max_value=energy_slider_max,
            value=(energy_min_data, energy_max_data),
            step=energy_step,
        )
    else:
        energy_range = st.sidebar.slider(
            "能量范围 (0..1)",
            min_value=energy_min_data,
            max_value=energy_max_data,
            value=(energy_min_data, energy_max_data),
            step=energy_step,
        )
else:
    energy_range = None
    st.sidebar.caption("能量：暂无可用数据。")

sort_key = st.sidebar.selectbox(
    "排序",
    options=[
        "title",
        "bpm_desc",
        "bpm_asc",
        "energy_desc",
        "energy_asc",
        "updated_at_desc",
        "updated_at_asc",
    ],
    index=0,
)

filtered = df.copy()
if bpm_range is not None and "bpm" in filtered.columns:
    lo, hi = bpm_range
    # Ensure default includes all tracks by expanding to exact data bounds.
    lo = min(lo, bpm_min_data)
    hi = max(hi, bpm_max_data)
    filtered = filtered[(filtered["bpm"].isna()) | ((filtered["bpm"] >= lo) & (filtered["bpm"] <= hi))]
if energy_range is not None and "energy_score" in filtered.columns:
    lo, hi = energy_range
    lo = min(lo, energy_min_data)
    hi = max(hi, energy_max_data)
    filtered = filtered[(filtered["energy_score"].isna()) | ((filtered["energy_score"] >= lo) & (filtered["energy_score"] <= hi))]

if sort_key == "title":
    filtered = filtered.sort_values(by=["title"], ascending=[True])
elif sort_key == "bpm_desc":
    filtered = filtered.sort_values(by=["bpm", "title"], ascending=[False, True], na_position="last")
elif sort_key == "bpm_asc":
    filtered = filtered.sort_values(by=["bpm", "title"], ascending=[True, True], na_position="last")
elif sort_key == "energy_desc":
    filtered = filtered.sort_values(by=["energy_score", "title"], ascending=[False, True], na_position="last")
elif sort_key == "energy_asc":
    filtered = filtered.sort_values(by=["energy_score", "title"], ascending=[True, True], na_position="last")
elif sort_key == "updated_at_desc":
    filtered = filtered.sort_values(by=["updated_at", "title"], ascending=[False, True], na_position="last")
elif sort_key == "updated_at_asc":
    filtered = filtered.sort_values(by=["updated_at", "title"], ascending=[True, True], na_position="last")

st.subheader("快速摘要")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("曲目数量", int(filtered.shape[0]))
with col2:
    avg_bpm = float(filtered["bpm"].dropna().mean()) if "bpm" in filtered.columns else float("nan")
    st.metric("平均 BPM", f"{avg_bpm:.1f}" if pd.notna(avg_bpm) else "N/A")
with col3:
    avg_energy = float(filtered["energy_score"].dropna().mean()) if "energy_score" in filtered.columns else float("nan")
    st.metric("平均能量", f"{avg_energy:.3f}" if pd.notna(avg_energy) else "N/A")

if "energy_score" in filtered.columns and filtered["energy_score"].notna().any():
    top5 = filtered.sort_values(by=["energy_score", "title"], ascending=[False, True]).head(5)
    st.write("Top 5 高能量曲目")
    st.dataframe(top5[["title", "artist", "bpm", "energy_score", "file_path"]], hide_index=True, width="stretch")

display_cols = [
    "title",
    "artist",
    "duration_sec",
    "bpm",
    "estimated_key",
    "rms_mean",
    "loudness_db_mean",
    "energy_score",
    "spectral_centroid_mean_hz",
    "zero_crossing_rate_mean",
    "updated_at",
    "file_path",
]
display_cols = [c for c in display_cols if c in df.columns]

st.subheader("曲目列表")
st.dataframe(filtered[display_cols], width="stretch", hide_index=True)

csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="下载当前表格 (CSV)",
    data=csv_bytes,
    file_name="ai_dj_tracks_filtered.csv",
    mime="text/csv",
)

st.subheader("BPM 分布")
bpm_s = filtered["bpm"].dropna() if "bpm" in filtered.columns else pd.Series(dtype=float)
if bpm_s.empty:
    st.warning("没有可用的 BPM（可能是某些音频无法估计节奏）。")
else:
    bins = min(12, max(5, int(len(bpm_s) / 2)))
    hist = pd.cut(bpm_s, bins=bins).value_counts(sort=False)
    hist_df = hist.rename_axis("bpm_bin").reset_index(name="count")
    hist_df["bpm_bin"] = hist_df["bpm_bin"].astype(str)
    st.bar_chart(hist_df, x="bpm_bin", y="count", width="stretch")

st.subheader("能量分布（energy_score 0..1）")
if "energy_score" in filtered.columns:
    energy_bins = 10
    hist = pd.cut(filtered["energy_score"].dropna(), bins=energy_bins).value_counts(sort=False)
    energy_df = hist.rename_axis("energy_bin").reset_index(name="count")
    energy_df["energy_bin"] = energy_df["energy_bin"].astype(str)
    st.bar_chart(energy_df, x="energy_bin", y="count", width="stretch")
else:
    st.warning("缺少 energy_score 列。请先重新运行摄取脚本。")

st.subheader("BPM vs 能量散点图")
plot_df = filtered.dropna(subset=["bpm", "energy_score"]).copy()
if plot_df.empty:
    st.warning("没有足够的数据用于散点图（需要 bpm 和 energy_score 都存在）。")
else:
    scatter_spec = {
        "mark": {"type": "point"},
        "encoding": {
            "x": {"field": "bpm", "type": "quantitative", "title": "BPM"},
            "y": {"field": "energy_score", "type": "quantitative", "title": "Energy score (0..1)"},
            "tooltip": [{"field": "title", "type": "nominal"}, {"field": "artist", "type": "nominal"}],
        },
    }
    st.vega_lite_chart(plot_df, scatter_spec, width="stretch")

