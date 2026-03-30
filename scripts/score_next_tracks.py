from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import joblib
import pandas as pd

from ai_dj_copilot.config import DATA_DIR, db_path_from_env
from ai_dj_copilot.modeling import PAIR_FEATURE_COLUMNS, pair_features_from_track_rows
from ai_dj_copilot.storage_sqlite import connect, fetch_all_tracks, init_db


def _get_tracks_df(db_path: Path) -> pd.DataFrame:
    conn = connect(db_path)
    try:
        init_db(conn)
        return fetch_all_tracks(conn)
    finally:
        conn.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Score candidate next tracks given one current track file_path.")
    parser.add_argument("--db", type=str, default=None, help="SQLite path; default uses AI_DJ_DB_PATH or data/tracks.db")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str((DATA_DIR / "modeling" / "baseline_logreg.joblib").resolve()),
        help="Path to trained model artifact",
    )
    parser.add_argument("--current-track", type=str, required=True, help="Current track file_path")
    parser.add_argument("--top-k", type=int, default=10, help="Number of ranked candidates to output")
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str((DATA_DIR / "modeling" / "next_track_scores.csv").resolve()),
        help="Where to save scored candidates as CSV",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser().resolve() if args.db else db_path_from_env()
    model_path = Path(args.model_path).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact.get("feature_columns", PAIR_FEATURE_COLUMNS)

    tracks_df = _get_tracks_df(db_path)
    if tracks_df.empty:
        raise SystemExit("No tracks found in DB.")

    if args.current_track not in set(tracks_df["file_path"].astype(str)):
        raise SystemExit(f"current-track not found in tracks table: {args.current_track}")

    a_row = tracks_df[tracks_df["file_path"] == args.current_track].iloc[0]
    candidates = tracks_df[tracks_df["file_path"] != args.current_track].copy()
    if candidates.empty:
        raise SystemExit("No candidate tracks available.")

    feature_rows: list[dict[str, float]] = []
    for _, b_row in candidates.iterrows():
        feature_rows.append(pair_features_from_track_rows(a_row, b_row))
    X = pd.DataFrame(feature_rows)
    X = X[feature_cols]

    # Probability of class 1 (compatible).
    compat_prob = model.predict_proba(X)[:, 1]
    result = candidates[["file_path", "title", "artist", "bpm", "energy_score", "estimated_key"]].copy()
    result["compatibility_score"] = compat_prob
    result = result.sort_values(by=["compatibility_score", "title"], ascending=[False, True]).reset_index(drop=True)
    result.to_csv(out_csv, index=False)

    top_k = max(1, int(args.top_k))
    top_df = result.head(top_k)
    print(f"[score_next_tracks] current={args.current_track}")
    print(f"[score_next_tracks] top_{top_k}:")
    print(top_df.to_string(index=False))
    print(f"[score_next_tracks] saved full ranking to {out_csv}")


if __name__ == "__main__":
    main()

