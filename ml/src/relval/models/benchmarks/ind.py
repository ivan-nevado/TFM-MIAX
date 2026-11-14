"""IND: industry harmonic mean of the multiple (Liu, Nissim & Thomas 2002).

Predicted multiple = n / sum(1 / m_i) over the fitting-sample firms of the same FF49 industry and
month. Paper silent on a minimum number of firms: 5 (config ``benchmarks.ind_min_firms``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def harmonic_mean_by_industry(multiple: pd.Series, industry: pd.Series, min_firms: int) -> pd.Series:
    """Harmonic mean per industry, missing where fewer than ``min_firms`` valid firms."""
    ok = multiple.notna() & (multiple > 0) & industry.notna()
    grouped = (1.0 / multiple[ok]).groupby(industry[ok])
    hm = grouped.count() / grouped.sum()
    return hm.where(grouped.count() >= min_firms)


def industry_stat(value: pd.Series, industry: pd.Series, min_firms: int, how: str = "median") -> pd.Series:
    ok = value.notna() & industry.notna()
    grouped = value[ok].groupby(industry[ok])
    stat = grouped.median() if how == "median" else grouped.mean()
    return stat.where(grouped.count() >= min_firms)


def predict_ind(df: pd.DataFrame, fit: np.ndarray, multiple: str, min_firms: int) -> pd.Series:
    """Industry harmonic mean of ``multiple`` (m2b, v2a, v2s) from the fitting rows, mapped to all rows."""
    level = np.exp(df["ln" + multiple])
    hm = harmonic_mean_by_industry(level[fit], df.loc[fit, "industry"], min_firms)
    return df["industry"].map(hm)
