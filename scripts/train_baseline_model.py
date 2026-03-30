from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_dj_copilot.config import DATA_DIR, db_path_from_env
from ai_dj_copilot.modeling import PAIR_FEATURE_COLUMNS, build_pair_dataset_from_sqlite


def _class_balance(y: pd.Series) -> dict[str, float]:
    counts = y.value_counts().to_dict()
    total = float(len(y))
    return {
        "count_0": float(counts.get(0, 0)),
        "count_1": float(counts.get(1, 0)),
        "ratio_0": float(counts.get(0, 0)) / total if total else 0.0,
        "ratio_1": float(counts.get(1, 0)) / total if total else 0.0,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train local Logistic Regression baseline for transition compatibility.")
    parser.add_argument("--db", type=str, default=None, help="SQLite path; default uses AI_DJ_DB_PATH or data/tracks.db")
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default=str((DATA_DIR / "modeling" / "pair_dataset.csv").resolve()),
        help="Processed dataset CSV path to save/use",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=str((DATA_DIR / "modeling" / "baseline_logreg.joblib").resolve()),
        help="Output model path",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=str((DATA_DIR / "modeling" / "baseline_metrics.json").resolve()),
        help="Output metrics JSON path",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser().resolve() if args.db else db_path_from_env()
    dataset_csv = Path(args.dataset_csv).expanduser().resolve()
    model_out = Path(args.model_out).expanduser().resolve()
    metrics_out = Path(args.metrics_out).expanduser().resolve()
    dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    df = build_pair_dataset_from_sqlite(db_path=db_path, include_skip=False)
    if df.empty:
        raise SystemExit("No labeled rows found for training. (skip labels are excluded)")

    # Save processed training dataset for traceability.
    df.to_csv(dataset_csv, index=False)

    if "label" not in df.columns:
        raise SystemExit("Dataset has no label column.")
    if df["label"].nunique() < 2:
        raise SystemExit("Need at least two classes (0 and 1) to train Logistic Regression.")

    X = df[PAIR_FEATURE_COLUMNS].copy()
    y = df["label"].astype(int).copy()

    stratify = y if y.nunique() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.random_state)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows_usable": int(len(df)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "class_balance_total": _class_balance(y),
        "class_balance_train": _class_balance(y_train),
        "class_balance_val": _class_balance(y_val),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_val, y_pred, labels=[0, 1]).tolist(),
        "feature_columns": PAIR_FEATURE_COLUMNS,
        "model": "LogisticRegression(class_weight='balanced')",
    }

    joblib.dump({"model": pipeline, "feature_columns": PAIR_FEATURE_COLUMNS}, model_out)
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[train_baseline_model] done")
    print(f"  usable_rows: {metrics['n_rows_usable']}")
    print(f"  class_balance_total: {metrics['class_balance_total']}")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall: {metrics['recall']:.4f}")
    print(f"  f1: {metrics['f1']:.4f}")
    print(f"  confusion_matrix [ [tn, fp], [fn, tp] ]: {metrics['confusion_matrix']}")
    print(f"  dataset_csv: {dataset_csv}")
    print(f"  model_out: {model_out}")
    print(f"  metrics_out: {metrics_out}")


if __name__ == "__main__":
    main()

