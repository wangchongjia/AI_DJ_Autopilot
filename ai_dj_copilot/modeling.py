from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import db_path_from_env
from .storage_sqlite import connect, init_db

# Order must match sklearn column order for training and scoring.
PAIR_FEATURE_COLUMNS: list[str] = [
    # Whole-track (legacy)
    "bpm_diff",
    "energy_diff",
    "loudness_diff",
    "spectral_centroid_diff",
    "zero_crossing_rate_diff",
    "duration_diff",
    "same_artist",
    "key_match",
    "harmonic_compatibility",
    # DJ-style adjustment costs (relative / transition-local)
    "bpm_adjustment_cost",
    "loudness_adjustment_cost",
    # Transition-local (outro A vs intro B)
    "outro_intro_energy_diff",
    "outro_intro_loudness_diff",
    "a_outro_silence_ratio",
    "b_intro_onset_strength_mean",
    "a_outro_silence_at_end_flag",
]


def _norm_text(v: object) -> str:
    if v is None or pd.isna(v):
        return ""
    return str(v).strip().lower()


def _split_key(k: object) -> tuple[Optional[int], Optional[str]]:
    """
    Parse values like:
    - "C major"
    - "F# minor"
    Returns (pitch_class_0_11, mode) or (None, None) on parse failure.
    """

    s = _norm_text(k)
    if not s:
        return None, None

    parts = s.split()
    if len(parts) < 2:
        return None, None
    note_raw, mode_raw = parts[0], parts[1]

    if mode_raw not in {"major", "minor"}:
        return None, None

    note_map = {
        "c": 0,
        "c#": 1,
        "db": 1,
        "d": 2,
        "d#": 3,
        "eb": 3,
        "e": 4,
        "f": 5,
        "f#": 6,
        "gb": 6,
        "g": 7,
        "g#": 8,
        "ab": 8,
        "a": 9,
        "a#": 10,
        "bb": 10,
        "b": 11,
    }
    pitch = note_map.get(note_raw)
    if pitch is None:
        return None, None
    return pitch, mode_raw


def key_match_flag(a_key: object, b_key: object) -> int:
    ap, am = _split_key(a_key)
    bp, bm = _split_key(b_key)
    if ap is None or bp is None or am is None or bm is None:
        return 0
    return int(ap == bp and am == bm)


def harmonic_compatibility_flag(a_key: object, b_key: object) -> int:
    """
    Very simple heuristic:
    compatible if
    - same key/mode
    - relative major/minor (same pitch, different mode)
    - +/- 1 semitone in same mode (neighbor key)
    """

    ap, am = _split_key(a_key)
    bp, bm = _split_key(b_key)
    if ap is None or bp is None or am is None or bm is None:
        return 0

    if ap == bp and am == bm:
        return 1
    if ap == bp and am != bm:
        return 1

    semitone_distance = min((ap - bp) % 12, (bp - ap) % 12)
    if am == bm and semitone_distance == 1:
        return 1
    return 0


def _f(x: Any, default: float = np.nan) -> float:
    if x is None or pd.isna(x):
        return float(default)
    return float(x)


def _abs_diff(a: Any, b: Any) -> float:
    if pd.isna(a) or pd.isna(b):
        return float("nan")
    return abs(float(a) - float(b))


def pair_features_from_track_rows(a_row: pd.Series, b_row: pd.Series) -> dict[str, float]:
    """
    Compute one row of pairwise features for (A -> B), aligned with PAIR_FEATURE_COLUMNS.

    bpm_adjustment_cost: relative tempo gap |bpm_a - bpm_b| / min(bpm_a, bpm_b) (undefined if BPM missing).

    loudness_adjustment_cost: |dB_outro(A) - dB_intro(B)| using segment loudness (mix boundary).

    outro_intro_energy_diff: |intro_energy_score(B) - outro_energy_score(A)| (batch-normalized segment RMS).

    outro_intro_loudness_diff: segment loudness difference (same as loudness_adjustment_cost here).
    """
    same_artist = int(
        str(a_row.get("artist", "")).strip().lower() == str(b_row.get("artist", "")).strip().lower()
    )

    abpm = a_row.get("bpm")
    bbpm = b_row.get("bpm")
    if pd.isna(abpm) or pd.isna(bbpm):
        bpm_adjustment_cost = float("nan")
    else:
        lo = min(float(abpm), float(bbpm))
        bpm_adjustment_cost = abs(float(abpm) - float(bbpm)) / max(lo, 1e-6)

    a_ol = a_row.get("outro_loudness_db_mean")
    b_il = b_row.get("intro_loudness_db_mean")
    if pd.isna(a_ol) or pd.isna(b_il):
        loudness_adjustment_cost = float("nan")
        outro_intro_loudness_diff = float("nan")
    else:
        loudness_adjustment_cost = abs(float(a_ol) - float(b_il))
        outro_intro_loudness_diff = loudness_adjustment_cost

    a_oe = a_row.get("outro_energy_score")
    b_ie = b_row.get("intro_energy_score")
    if pd.isna(a_oe) or pd.isna(b_ie):
        outro_intro_energy_diff = float("nan")
    else:
        outro_intro_energy_diff = abs(float(b_ie) - float(a_oe))

    a_osr = a_row.get("outro_silence_ratio")
    b_ism = b_row.get("intro_onset_strength_mean")
    a_flag = a_row.get("outro_silence_at_end_flag")

    return {
        "bpm_diff": _abs_diff(a_row.get("bpm"), b_row.get("bpm")),
        "energy_diff": _abs_diff(a_row.get("energy_score"), b_row.get("energy_score")),
        "loudness_diff": _abs_diff(a_row.get("loudness_db_mean"), b_row.get("loudness_db_mean")),
        "spectral_centroid_diff": _abs_diff(
            a_row.get("spectral_centroid_mean_hz"), b_row.get("spectral_centroid_mean_hz")
        ),
        "zero_crossing_rate_diff": _abs_diff(
            a_row.get("zero_crossing_rate_mean"), b_row.get("zero_crossing_rate_mean")
        ),
        "duration_diff": _abs_diff(a_row.get("duration_sec"), b_row.get("duration_sec")),
        "same_artist": float(same_artist),
        "key_match": float(key_match_flag(a_row.get("estimated_key"), b_row.get("estimated_key"))),
        "harmonic_compatibility": float(
            harmonic_compatibility_flag(a_row.get("estimated_key"), b_row.get("estimated_key"))
        ),
        "bpm_adjustment_cost": float(bpm_adjustment_cost),
        "loudness_adjustment_cost": float(loudness_adjustment_cost),
        "outro_intro_energy_diff": float(outro_intro_energy_diff),
        "outro_intro_loudness_diff": float(outro_intro_loudness_diff),
        "a_outro_silence_ratio": _f(a_osr, 0.0),
        "b_intro_onset_strength_mean": _f(b_ism, 0.0),
        "a_outro_silence_at_end_flag": float(int(a_flag)) if not pd.isna(a_flag) else 0.0,
    }


def build_pair_dataset_from_sqlite(db_path: Optional[Path] = None, *, include_skip: bool = False) -> pd.DataFrame:
    """
    Join transition_labels with tracks A/B and compute pairwise features.
    """

    db_path = db_path or db_path_from_env()
    conn = connect(db_path)
    try:
        init_db(conn)
        df = pd.read_sql_query(
            """
            SELECT
              tl.track_a_file_path,
              tl.track_b_file_path,
              tl.label,
              tl.created_at AS label_created_at,
              tl.updated_at AS label_updated_at,

              a.title AS a_title,
              a.artist AS a_artist,
              a.duration_sec AS a_duration_sec,
              a.bpm AS a_bpm,
              a.estimated_key AS a_estimated_key,
              a.energy_score AS a_energy_score,
              a.loudness_db_mean AS a_loudness_db_mean,
              a.spectral_centroid_mean_hz AS a_spectral_centroid_mean_hz,
              a.zero_crossing_rate_mean AS a_zero_crossing_rate_mean,
              a.outro_energy_score AS a_outro_energy_score,
              a.intro_energy_score AS a_intro_energy_score,
              a.outro_loudness_db_mean AS a_outro_loudness_db_mean,
              a.intro_loudness_db_mean AS a_intro_loudness_db_mean,
              a.outro_silence_ratio AS a_outro_silence_ratio,
              a.intro_onset_strength_mean AS a_intro_onset_strength_mean,
              a.outro_silence_at_end_flag AS a_outro_silence_at_end_flag,

              b.title AS b_title,
              b.artist AS b_artist,
              b.duration_sec AS b_duration_sec,
              b.bpm AS b_bpm,
              b.estimated_key AS b_estimated_key,
              b.energy_score AS b_energy_score,
              b.loudness_db_mean AS b_loudness_db_mean,
              b.spectral_centroid_mean_hz AS b_spectral_centroid_mean_hz,
              b.zero_crossing_rate_mean AS b_zero_crossing_rate_mean,
              b.outro_energy_score AS b_outro_energy_score,
              b.intro_energy_score AS b_intro_energy_score,
              b.outro_loudness_db_mean AS b_outro_loudness_db_mean,
              b.intro_loudness_db_mean AS b_intro_loudness_db_mean,
              b.outro_silence_ratio AS b_outro_silence_ratio,
              b.intro_onset_strength_mean AS b_intro_onset_strength_mean,
              b.outro_silence_at_end_flag AS b_outro_silence_at_end_flag
            FROM transition_labels tl
            JOIN tracks a ON a.file_path = tl.track_a_file_path
            JOIN tracks b ON b.file_path = tl.track_b_file_path
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return df

    if not include_skip:
        df = df[df["label"].notna()].copy()

    feats: list[dict[str, float]] = []
    for _, row in df.iterrows():
        a = pd.Series({c[2:]: row[c] for c in row.index if isinstance(c, str) and c.startswith("a_")})
        b = pd.Series({c[2:]: row[c] for c in row.index if isinstance(c, str) and c.startswith("b_")})
        feats.append(pair_features_from_track_rows(a, b))

    feat_df = pd.DataFrame(feats)
    df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    for c in PAIR_FEATURE_COLUMNS:
        if c in df.columns:
            median_v = float(df[c].median()) if df[c].notna().any() else 0.0
            df[c] = df[c].fillna(median_v)

    if "label" in df.columns and df["label"].notna().any():
        df["label"] = df["label"].astype(int)

    return df
