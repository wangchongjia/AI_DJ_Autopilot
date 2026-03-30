from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import joblib


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Print top positive/negative Logistic Regression coefficients from a saved baseline model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="data/modeling/baseline_logreg.joblib",
        help="Path to joblib artifact from train_baseline_model.py",
    )
    parser.add_argument("--top", type=int, default=10, help="How many features to show on each side")
    parser.add_argument(
        "--out-json",
        type=str,
        default="data/modeling/logreg_coefficients.json",
        help="Optional JSON output path",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from ai_dj_copilot.modeling import PAIR_FEATURE_COLUMNS

    model_path = Path(args.model_path).expanduser().resolve()
    artifact = joblib.load(model_path)
    pipeline = artifact["model"]
    feature_cols: list[str] = list(artifact.get("feature_columns", PAIR_FEATURE_COLUMNS))

    clf = pipeline.named_steps["clf"]
    coef = clf.coef_.ravel()
    if len(coef) != len(feature_cols):
        raise SystemExit(f"coef length {len(coef)} != feature_columns {len(feature_cols)}")

    pairs = sorted(zip(feature_cols, coef.tolist()), key=lambda x: x[1], reverse=True)
    top_n = max(1, int(args.top))
    top_pos = pairs[:top_n]
    top_neg = list(reversed(pairs[-top_n:]))

    print("[analyze_logreg_coefficients] LogisticRegression coef (after StandardScaler; higher -> more compatible)")
    print(f"  model: {model_path}")
    print()
    print(f"Top {top_n} positive (favor class 1):")
    for name, c in top_pos:
        print(f"  {name:40s} {c:+.6f}")
    print()
    print(f"Top {top_n} negative (favor class 0):")
    for name, c in top_neg:
        print(f"  {name:40s} {c:+.6f}")

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_path": str(model_path),
        "top_positive": [{"feature": n, "coefficient": c} for n, c in top_pos],
        "top_negative": [{"feature": n, "coefficient": c} for n, c in top_neg],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"[analyze_logreg_coefficients] wrote {out_path}")


if __name__ == "__main__":
    main()
