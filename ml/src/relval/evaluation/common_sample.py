"""Common subsample (paper, online appendix Table IA.5): firm-months for which every model in the
comparison has a valid prediction."""

from __future__ import annotations

import pandas as pd

from relval.evaluation.metrics import metrics_table


def common_sample(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    present = [m for m in models if (preds["model"] == m).any()]
    sub = preds[preds["model"].isin(present)]
    counts = sub.groupby(["id", "date"])["model"].nunique()
    keys = counts[counts == len(present)].reset_index()[["id", "date"]]
    return sub.merge(keys, on=["id", "date"], how="inner")


def common_sample_table(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    present = [m for m in models if (preds["model"] == m).any()]
    return metrics_table(common_sample(preds, present), present)
