"""Performance metrics of Table 2 (paper, section 4.1), computed from the master predictions table."""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_COLUMNS = [
    "N", "med_abs_eps", "avg_abs_eps", "med_eps", "avg_eps", "sd_eps",
    "rmse_y", "rmse_M", "r2oos_y", "r2oos_M", "rho",
]  # fmt: skip


def rmse(actual, predicted) -> float:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def r2_oos(actual, predicted, train_mean) -> float:
    """1 - sum (p - phat)^2 / sum (p - pbar)^2, where ``pbar`` is the mean of the training sample
    of the model that produced each prediction (one value per observation)."""
    a, p, m = np.asarray(actual, float), np.asarray(predicted, float), np.asarray(train_mean, float)
    den = np.sum((a - m) ** 2)
    return float(1.0 - np.sum((a - p) ** 2) / den) if den > 0 else np.nan


def compute_metrics(pred: pd.DataFrame) -> dict[str, float]:
    """Pooled metrics over a model's out-of-sample predictions."""
    if pred.empty:
        return dict.fromkeys([*METRIC_COLUMNS, "n_negative_Mhat"], np.nan) | {"N": 0}
    eps = pred["eps"].to_numpy(dtype=float)
    return {
        "N": int(len(pred)),
        "med_abs_eps": float(np.median(np.abs(eps))),
        "avg_abs_eps": float(np.mean(np.abs(eps))),
        "med_eps": float(np.median(eps)),
        "avg_eps": float(np.mean(eps)),
        "sd_eps": float(np.std(eps, ddof=1)) if len(eps) > 1 else np.nan,
        "rmse_y": rmse(pred["y"], pred["yhat"]),
        "rmse_M": rmse(pred["M"], pred["Mhat"]),
        "r2oos_y": r2_oos(pred["y"], pred["yhat"], pred["train_mean_y"]),
        "r2oos_M": r2_oos(pred["M"], pred["Mhat"], pred["train_mean_M"]),
        "rho": float(np.corrcoef(pred["M"], pred["Mhat"])[0, 1]) if len(pred) > 2 else np.nan,
        "n_negative_Mhat": int((pred["Mhat"] < 0).sum()),
    }


def metrics_table(preds: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """One row of metrics per model, in the given order."""
    models = models or list(pd.unique(preds["model"]))
    rows = [{"model": m, **compute_metrics(preds[preds["model"] == m])} for m in models]
    return pd.DataFrame(rows)


def seed_spread(preds_by_seed: dict[int, pd.DataFrame], models: list[str]) -> pd.DataFrame:
    """Spread of every Table 2 metric across block seeds (long format: model, metric, stats)."""
    rows = []
    for seed, preds in preds_by_seed.items():
        for m in models:
            met = compute_metrics(preds[preds["model"] == m])
            rows += [{"model": m, "seed": seed, "metric": k, "value": v} for k, v in met.items()]
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    out = (
        long.groupby(["model", "metric"], sort=False)["value"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    first = long[long["seed"] == next(iter(preds_by_seed))].set_index(["model", "metric"])["value"]
    out["headline"] = [first.get((m, k), np.nan) for m, k in zip(out["model"], out["metric"], strict=True)]
    return out
