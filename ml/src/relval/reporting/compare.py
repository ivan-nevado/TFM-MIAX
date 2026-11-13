"""Paper-versus-ours comparisons. The paper's numbers live in ``docs/replication/paper_targets``
(machine-readable transcriptions of the published tables); nothing is hard-coded here."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def load_paper(cfg, name: str) -> pd.DataFrame | None:
    path = cfg.path("paper_targets") / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def strip_stars(value) -> float:
    """``'0.37***'`` -> 0.37."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = re.sub(r"\*+$", "", str(value).strip())
    try:
        return float(text)
    except ValueError:
        return np.nan


def compare_long(
    ours: pd.DataFrame, paper: pd.DataFrame | None, keys: list[str], metrics: list[str]
) -> pd.DataFrame:
    """Long table: keys, metric, paper, ours, diff (ours - paper)."""
    o = ours[keys + [m for m in metrics if m in ours.columns]].melt(
        keys, var_name="metric", value_name="ours"
    )
    if paper is None:
        return o.assign(paper=np.nan, diff=np.nan)[[*keys, "metric", "paper", "ours", "diff"]]
    p = paper[keys + [m for m in metrics if m in paper.columns]].melt(
        keys, var_name="metric", value_name="paper"
    )
    p["paper"] = p["paper"].map(strip_stars)
    out = o.merge(p, on=[*keys, "metric"], how="left")
    out["diff"] = out["ours"] - out["paper"]
    return out[[*keys, "metric", "paper", "ours", "diff"]]


def side_by_side(
    ours: pd.DataFrame, paper: pd.DataFrame | None, keys: list[str], metrics: list[str]
) -> pd.DataFrame:
    """Wide table with ``<metric>`` (ours) next to ``<metric>_paper``."""
    out = ours[keys + [m for m in metrics if m in ours.columns]].copy()
    if paper is None:
        return out
    p = paper[keys + [m for m in metrics if m in paper.columns]].copy()
    for m in metrics:
        if m in p.columns:
            p[m] = p[m].map(strip_stars)
    out = out.merge(p, on=keys, how="left", suffixes=("", "_paper"))
    ordered = list(keys)
    for m in metrics:
        if m in ours.columns:
            ordered += [m] + ([f"{m}_paper"] if f"{m}_paper" in out.columns else [])
    return out[ordered]
