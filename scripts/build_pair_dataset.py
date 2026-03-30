from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from ai_dj_copilot.config import DATA_DIR, db_path_from_env
from ai_dj_copilot.modeling import build_pair_dataset_from_sqlite


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build pair modeling dataset from SQLite (transition_labels JOIN tracks A/B)."
    )
    parser.add_argument("--db", type=str, default=None, help="SQLite path; default uses AI_DJ_DB_PATH or data/tracks.db")
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str((DATA_DIR / "modeling" / "pair_dataset.csv").resolve()),
        help="Output CSV path",
    )
    parser.add_argument(
        "--include-skip",
        action="store_true",
        help="Include skip rows (label=NULL). Default excludes them.",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser().resolve() if args.db else db_path_from_env()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = build_pair_dataset_from_sqlite(db_path, include_skip=args.include_skip)
    df.to_csv(out_csv, index=False)

    usable = int(df["label"].notna().sum()) if "label" in df.columns else 0
    print(f"[build_pair_dataset] rows_total={len(df)} rows_with_label={usable} out={out_csv}")


if __name__ == "__main__":
    main()

