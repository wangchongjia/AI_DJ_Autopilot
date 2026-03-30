from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from ai_dj_copilot.config import db_path_from_env
from ai_dj_copilot.storage_sqlite import (
    connect,
    fetch_all_tracks,
    fetch_transition_labels,
    init_db,
    upsert_transition_label,
)


def load_tracks(db_path: Path) -> pd.DataFrame:
    conn = connect(db_path)
    try:
        init_db(conn)
        return fetch_all_tracks(conn)
    finally:
        conn.close()


def load_labeled_pairs(db_path: Path) -> pd.DataFrame:
    conn = connect(db_path)
    try:
        init_db(conn)
        return fetch_transition_labels(conn)
    finally:
        conn.close()


def _is_nan_or_none(v: object) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v)


def generate_candidate_queue(
    tracks_df: pd.DataFrame,
    labeled_set: set[tuple[str, str]],
    *,
    include_labeled: bool,
    batch_size: int,
    seed: int,
    near_bpm_diff: float = 5.0,
    medium_bpm_diff: float = 15.0,
) -> list[tuple[str, str]]:
    """
    Simple sampling:
    - near: abs(BPM_a - BPM_b) <= near_bpm_diff
    - medium: near_bpm_diff < abs(diff) <= medium_bpm_diff
    - random negatives: the rest (including missing BPM)
    """

    tracks_by_path = tracks_df.set_index("file_path").to_dict("index")
    paths = list(tracks_by_path.keys())
    near_pairs: list[tuple[str, str]] = []
    medium_pairs: list[tuple[str, str]] = []
    random_pairs: list[tuple[str, str]] = []

    for a in paths:
        for b in paths:
            if a == b:
                continue

            bpm_a = tracks_by_path[a].get("bpm")
            bpm_b = tracks_by_path[b].get("bpm")

            if _is_nan_or_none(bpm_a) or _is_nan_or_none(bpm_b):
                random_pairs.append((a, b))
                continue

            diff = abs(float(bpm_a) - float(bpm_b))
            if diff <= near_bpm_diff:
                near_pairs.append((a, b))
            elif diff <= medium_bpm_diff:
                medium_pairs.append((a, b))
            else:
                random_pairs.append((a, b))

    def _keep(pair: tuple[str, str]) -> bool:
        if include_labeled:
            return True
        return pair not in labeled_set

    near_unl = [p for p in near_pairs if _keep(p)]
    medium_unl = [p for p in medium_pairs if _keep(p)]
    random_unl = [p for p in random_pairs if _keep(p)]

    rng = random.Random(seed)

    k_near = int(batch_size * 0.5)
    k_medium = int(batch_size * 0.3)
    k_random = max(0, batch_size - k_near - k_medium)

    queue: list[tuple[str, str]] = []
    used: set[tuple[str, str]] = set()

    def _take(src: list[tuple[str, str]], k: int) -> None:
        nonlocal queue, used
        if k <= 0 or not src:
            return
        take = src if len(src) <= k else rng.sample(src, k)
        for p in take:
            if p in used:
                continue
            used.add(p)
            queue.append(p)

    _take(near_unl, k_near)
    _take(medium_unl, k_medium)

    if len(queue) < batch_size:
        # Fill remaining with random negatives (then leftover near/medium if needed).
        remaining = batch_size - len(queue)
        pool: list[tuple[str, str]] = [p for p in random_unl if p not in used]
        pool += [p for p in near_unl if p not in used]
        pool += [p for p in medium_unl if p not in used]
        rng.shuffle(pool)
        for p in pool[:remaining]:
            used.add(p)
            queue.append(p)

    rng.shuffle(queue)
    return queue[:batch_size]


def _format_bpm(v: object) -> str:
    if _is_nan_or_none(v):
        return ""
    return f"{float(v):.1f}"


def _format_energy(v: object) -> str:
    if _is_nan_or_none(v):
        return ""
    return f"{float(v):.3f}"


def _track_value_str(row: dict, key: str) -> str:
    v = row.get(key)
    if _is_nan_or_none(v):
        return ""
    if key == "duration_sec":
        return f"{float(v):.1f}"
    if key == "bpm":
        return _format_bpm(v)
    if key == "energy_score":
        return _format_energy(v)
    return str(v)


st.set_page_config(page_title="AI DJ Copilot - Pair Labeling", layout="wide")
st.title("Transition Pair Labeling (Local Prototype)")
st.caption("仅做人工标注：compatible(1) / incompatible(0) / skip(不记录偏好)。没有推荐/建模。")

db_path = Path(db_path_from_env().resolve())

tracks_df = load_tracks(db_path)
if tracks_df.empty:
    st.error("SQLite 里没有曲目数据。请先运行 `python scripts/ingest_kpop.py`。")
    st.stop()

tracks_by_path = tracks_df.set_index("file_path").to_dict("index")
paths = list(tracks_by_path.keys())

with st.sidebar:
    st.header("标注控制")

    include_labeled = st.checkbox("显示已标注的对（调试用）", value=False)
    batch_size = st.selectbox("候选对批次大小", options=[25, 50, 100], index=2)

    near_bpm_diff = st.number_input("near BPM 阈值 (<=)", min_value=0.0, max_value=60.0, value=5.0, step=0.5)
    medium_bpm_diff = st.number_input(
        "medium BPM 阈值 (<=)",
        min_value=float(near_bpm_diff),
        max_value=80.0,
        value=15.0,
        step=1.0,
    )

    if st.button("生成下一批候选对", key="regen_queue"):
        st.session_state["candidate_queue"] = []
        st.session_state["current_idx"] = 0

labeled_df = load_labeled_pairs(db_path)
if labeled_df.empty:
    labeled_set: set[tuple[str, str]] = set()
else:
    labeled_set = set(zip(labeled_df["track_a_file_path"].astype(str), labeled_df["track_b_file_path"].astype(str)))

total_answered = int(len(labeled_set))
if labeled_df.empty:
    compat_n = 0
    incompat_n = 0
    skip_n = 0
else:
    compat_n = int((labeled_df["label"] == 1).sum())
    incompat_n = int((labeled_df["label"] == 0).sum())
    skip_n = int(labeled_df["label"].isna().sum())

st.subheader("当前进度")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("已回答对数", total_answered)
col_b.metric("compatible(1)", compat_n)
col_c.metric("incompatible(0)", incompat_n)
col_d.metric("skip(未偏好)", skip_n)

if "candidate_queue" not in st.session_state:
    st.session_state["candidate_queue"] = []
if "current_idx" not in st.session_state:
    st.session_state["current_idx"] = 0

# Build queue if empty or exhausted.
need_new_queue = (
    not st.session_state["candidate_queue"]
    or st.session_state["current_idx"] >= len(st.session_state["candidate_queue"])
)

if need_new_queue:
    seed = random.randint(0, 2**31 - 1)
    st.session_state["candidate_queue"] = generate_candidate_queue(
        tracks_df,
        labeled_set,
        include_labeled=include_labeled,
        batch_size=int(batch_size),
        seed=seed,
        near_bpm_diff=float(near_bpm_diff),
        medium_bpm_diff=float(medium_bpm_diff),
    )
    st.session_state["current_idx"] = 0

queue = st.session_state["candidate_queue"]
idx = int(st.session_state["current_idx"])

st.subheader("标注界面")
if not queue:
    st.warning("没有生成到候选对（可能是全部已标注且你未启用“显示已标注的对”）。")
    st.stop()

a_path, b_path = queue[idx]
a_row = tracks_by_path[a_path]
b_row = tracks_by_path[b_path]

st.write(f"当前对：{idx + 1}/{len(queue)}  （A -> B）")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Track A**")
    st.write(f"title: {a_row.get('title','')}")
    st.write(f"artist: {a_row.get('artist','')}")
    st.write(f"duration_sec: {_track_value_str(a_row, 'duration_sec')}")
    st.write(f"bpm: {_track_value_str(a_row, 'bpm')}")
    st.write(f"estimated_key: {a_row.get('estimated_key','')}")
    st.write(f"energy_score: {_track_value_str(a_row, 'energy_score')}")
with col2:
    st.markdown("**Track B**")
    st.write(f"title: {b_row.get('title','')}")
    st.write(f"artist: {b_row.get('artist','')}")
    st.write(f"duration_sec: {_track_value_str(b_row, 'duration_sec')}")
    st.write(f"bpm: {_track_value_str(b_row, 'bpm')}")
    st.write(f"estimated_key: {b_row.get('estimated_key','')}")
    st.write(f"energy_score: {_track_value_str(b_row, 'energy_score')}")

def _save_and_advance(new_label: Optional[int]) -> None:
    conn = connect(db_path)
    try:
        init_db(conn)
        upsert_transition_label(
            conn,
            track_a_file_path=a_path,
            track_b_file_path=b_path,
            label=new_label,
        )
    finally:
        conn.close()

    st.session_state["current_idx"] = idx + 1
    st.rerun()


col_btn1, col_btn2, col_btn3 = st.columns(3)
with col_btn1:
    if st.button("compatible (1)", key=f"btn_compat_{idx}"):
        _save_and_advance(1)
with col_btn2:
    if st.button("incompatible (0)", key=f"btn_incompat_{idx}"):
        _save_and_advance(0)
with col_btn3:
    if st.button("skip", key=f"btn_skip_{idx}"):
        _save_and_advance(None)

