from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_corr(fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.nanstd(y_true) < 1e-12 or np.nanstd(y_pred) < 1e-12:
        return float("nan")
    try:
        return float(fn(y_true, y_pred)[0])
    except Exception:
        return float("nan")


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict:
    y_true = np.asarray(list(y_true), dtype=float).reshape(-1)
    y_pred = np.asarray(list(y_pred), dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {k: np.nan for k in [
            "R2", "MSE", "RMSE", "MAE", "MedianAE", "Bias_log10",
            "Pearson_r", "Spearman_rho", "Median_fold_error",
            "Pct_within_2fold", "Pct_within_5fold", "Pct_within_10fold",
        ]}

    err = y_pred - y_true
    abs_err = np.abs(err)
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    return {
        "R2": r2,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedianAE": float(np.median(abs_err)),
        "Bias_log10": float(np.mean(err)),
        "Pearson_r": _safe_corr(pearsonr, y_true, y_pred),
        "Spearman_rho": _safe_corr(spearmanr, y_true, y_pred),
        "Median_fold_error": float(10.0 ** np.median(abs_err)),
        "Pct_within_2fold": float(100.0 * np.mean(abs_err <= np.log10(2.0))),
        "Pct_within_5fold": float(100.0 * np.mean(abs_err <= np.log10(5.0))),
        "Pct_within_10fold": float(100.0 * np.mean(abs_err <= 1.0)),
    }


def summarize_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    
    rows = []
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(values) == 0:
            continue
        
        if col in {"repeat", "fold", "n_train", "n_valid"}:
            continue
        rows.append({
            "metric": col,
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
            "q05": float(values.quantile(0.05)),
            "q95": float(values.quantile(0.95)),
            "n": int(len(values)),
        })
    return pd.DataFrame(rows)
