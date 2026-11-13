"""Errors across firms (Figures 2 and 3): quintiles of size, market-to-book and ROE formed each
month, and Fama-French 12 industries."""

from __future__ import annotations

import numpy as np
import pandas as pd

SORTS = {"size": "M", "m2b": "lnm2b", "roe": "roe_wr"}


def add_quintiles(panel: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Monthly quintile (1..n) of market capitalisation, market-to-book and ROE for each firm-month."""
    out = panel[["id", "date", "ff12"]].copy()
    for name, col in SORTS.items():
        pct = panel.groupby("date")[col].rank(pct=True)
        out[f"q_{name}"] = np.ceil(pct * n).clip(1, n)
    return out


def errors_by_group(preds: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """med(|eps|) and med(eps) per model for each quintile and FF12 industry (long format)."""
    df = preds.merge(groups, on=["id", "date"], how="left")
    df["abs_eps"] = df["eps"].abs()
    frames = []
    for sort in ("q_size", "q_m2b", "q_roe", "ff12"):
        g = df.dropna(subset=[sort]).groupby(["model", sort])
        res = g.agg(
            med_abs_eps=("abs_eps", "median"), med_eps=("eps", "median"), n=("eps", "size")
        ).reset_index()
        res = res.rename(columns={sort: "group"}).assign(sort=sort.replace("q_", ""))
        frames.append(res)
    return pd.concat(frames, ignore_index=True)[["sort", "group", "model", "med_abs_eps", "med_eps", "n"]]
