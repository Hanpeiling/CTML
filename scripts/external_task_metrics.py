from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EFFECTS = (
    "biochemistry", "development", "genetics", "growth",
    "histology", "hormones", "morphology", "reproduction",
)


def summarize_by_effect(predictions: pd.DataFrame) -> pd.DataFrame:
    needed = {"task", "true_log10", "pred_log10"}
    missing = needed.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    df = predictions.copy()
    df["task"] = df["task"].astype(str).str.strip().str.lower()
    unexpected = set(df["task"]).difference(EFFECTS)
    absent = set(EFFECTS).difference(df["task"])
    if unexpected or absent:
        raise ValueError(f"Unexpected tasks: {sorted(unexpected)}; missing tasks: {sorted(absent)}")
    rows = []
    for task in EFFECTS:
        group = df.loc[df["task"] == task]
        y = pd.to_numeric(group["true_log10"], errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(group["pred_log10"], errors="coerce").to_numpy(dtype=float)
        if len(y) < 2 or not np.isfinite(y).all() or not np.isfinite(yp).all():
            raise ValueError(f"Missing/nonfinite predictions or too few records for {task}")
        mse = float(np.mean((y - yp) ** 2))
        variance = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - np.sum((y - yp) ** 2) / variance) if variance > 0 else float('nan')
        rows.append({
            "effect": task,
            "n_records": int(len(y)),
            "R2": r2,
            "RMSE": float(np.sqrt(mse)),
            "MAE": float(np.mean(np.abs(y - yp))),
        })
    return pd.DataFrame(rows)


def save_and_print(predictions: pd.DataFrame, destination: Path) -> pd.DataFrame:
    result = summarize_by_effect(predictions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, float_format="%.10f")
    print("\nExternal validation by sublethal effect (R2 / RMSE / MAE):", flush=True)
    print(result.to_string(index=False, formatters={
        "R2": lambda v: f"{v:.6f}",
        "RMSE": lambda v: f"{v:.6f}",
        "MAE": lambda v: f"{v:.6f}",
    }), flush=True)
    print(f"Per-effect metrics saved: {destination}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="CSV containing task, true_log10 and pred_log10")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    output = args.out or args.predictions.with_name("external_metrics_by_effect.csv")
    save_and_print(pd.read_csv(args.predictions), output)


if __name__ == "__main__":
    main()
