"""Errors over time (Figure 1) and the share of months in which the GBM beats a benchmark."""

from __future__ import annotations

import pandas as pd

from relval.evaluation.metrics import metrics_table


def monthly_median_abs_error(preds: pd.DataFrame) -> pd.DataFrame:
    """med(|eps|) per model and month, wide format (index: date, columns: models)."""
    return (
        preds.assign(abs_eps=preds["eps"].abs())
        .groupby(["date", "model"])["abs_eps"]
        .median()
        .unstack("model")
    )


def share_of_months_below(monthly: pd.DataFrame, model: str, benchmarks: list[str]) -> pd.DataFrame:
    """Share of months in which ``model`` has a lower med(|eps|) than each benchmark."""
    rows = []
    for b in benchmarks:
        if model not in monthly or b not in monthly:
            continue
        both = monthly[[model, b]].dropna()
        rows.append(
            {
                "model": model,
                "benchmark": b,
                "months": len(both),
                "share_below": float((both[model] < both[b]).mean()),
            }
        )
    return pd.DataFrame(rows)


def metrics_by_period(preds: pd.DataFrame, models: list[str], freq: str = "decade") -> pd.DataFrame:
    """Table 2 metrics by decade or by year (sensitivity run 1: the sample is thinnest early on)."""
    year = preds["date"].dt.year
    key = (year // 10 * 10) if freq == "decade" else year
    frames = []
    for period, grp in preds.groupby(key.to_numpy()):
        frames.append(metrics_table(grp, models).assign(period=int(period)))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out[["period", *[c for c in out.columns if c != "period"]]] if len(out) else out
