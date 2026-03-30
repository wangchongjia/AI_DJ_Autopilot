from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ai_dj_copilot.config import db_path_from_env
from ai_dj_copilot.storage_sqlite import connect, fetch_all_tracks, init_db


def _describe_numeric(df: pd.DataFrame, col: str) -> str:
    s = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
    if s.empty:
        return f"{col}: N/A"
    return (
        f"{col}: n={len(s)} "
        f"min={s.min():.3f} p25={s.quantile(0.25):.3f} median={s.median():.3f} "
        f"p75={s.quantile(0.75):.3f} max={s.max():.3f} mean={s.mean():.3f}"
    )


def validate(db_path: Path) -> int:
    conn = connect(db_path)
    try:
        init_db(conn)
        df = fetch_all_tracks(conn)
    finally:
        conn.close()

    if df.empty:
        print("[validate] No tracks found in DB.")
        return 1

    print(f"[validate] Tracks: {len(df)}")
    print(_describe_numeric(df, "duration_sec"))
    print(_describe_numeric(df, "bpm"))
    print(_describe_numeric(df, "rms_mean"))
    print(_describe_numeric(df, "loudness_db_mean"))
    print(_describe_numeric(df, "energy_score"))
    print(_describe_numeric(df, "spectral_centroid_mean_hz"))
    print(_describe_numeric(df, "zero_crossing_rate_mean"))

    # Quick sanity checks for a small pop library (heuristics).
    warnings = 0

    if "duration_sec" in df.columns:
        short = df[df["duration_sec"] < 30]
        if not short.empty:
            warnings += 1
            print(f"[validate] WARNING: {len(short)} tracks shorter than 30s (maybe preview clips or decode issues).")

    if "bpm" in df.columns:
        missing_bpm = df["bpm"].isna().sum()
        if missing_bpm > 0:
            warnings += 1
            print(f"[validate] WARNING: {missing_bpm} tracks missing BPM (tempo estimation may fail on some tracks).")

        weird_bpm = df[df["bpm"].notna() & ((df["bpm"] < 50) | (df["bpm"] > 220))]
        if not weird_bpm.empty:
            warnings += 1
            print(f"[validate] WARNING: {len(weird_bpm)} tracks have BPM outside [50, 220]. Sample:")
            print(weird_bpm[["title", "artist", "bpm", "file_path"]].head(10).to_string(index=False))

    if "energy_score" in df.columns:
        out_of_range = df[df["energy_score"].notna() & ((df["energy_score"] < -1e-6) | (df["energy_score"] > 1 + 1e-6))]
        if not out_of_range.empty:
            warnings += 1
            print(f"[validate] WARNING: {len(out_of_range)} tracks have energy_score outside [0, 1].")

    print(f"[validate] Done with {warnings} warning(s).")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Validate extracted features in the local SQLite database.")
    parser.add_argument("--db", type=str, default=None, help="SQLite db path (defaults to AI_DJ_DB_PATH or data/tracks.db).")
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser().resolve() if args.db else db_path_from_env()
    raise SystemExit(validate(db_path))


if __name__ == "__main__":
    main()

