"""BL: monthly cross-sectional OLS of the multiple on ten variables (Bhojraj & Lee 2002).

The ten variables (paper, online appendix Table IA.3; section 4.2 of the master prompt):
the three industry harmonic means of m2b, v2a and v2s; the industry-adjusted EBIT margin and the
same variable times an indicator for negative values; industry-adjusted sales growth; leverage
(dltt + dlc) / book equity; pre-tax return on net operating assets; return on equity; R&D to sales.
Industry statistics use the fitting sample only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from relval.models.benchmarks.ind import harmonic_mean_by_industry, industry_stat
from relval.panel.book_equity import div

BL_FEATURES = [
    "ind_hm_m2b", "ind_hm_v2a", "ind_hm_v2s", "adj_opm", "adj_opm_neg", "adj_sgr",
    "lev", "pretret_noa_wr", "roe_wr", "rd_sale_wr",
]  # fmt: skip


def bl_features(df: pd.DataFrame, fit: np.ndarray, min_firms: int) -> pd.DataFrame:
    """The ten BL variables for every row of the month, with industry statistics from ``fit`` rows."""
    ind = df["industry"]
    out = pd.DataFrame(index=df.index)
    for mult in ("m2b", "v2a", "v2s"):
        level = np.exp(df["ln" + mult])
        out[f"ind_hm_{mult}"] = ind.map(harmonic_mean_by_industry(level[fit], ind[fit], min_firms))
    opm_med = industry_stat(df.loc[fit, "opmad_wr"], ind[fit], min_firms)
    sgr_med = industry_stat(df.loc[fit, "salesgrowth_an"], ind[fit], min_firms)
    out["adj_opm"] = df["opmad_wr"] - ind.map(opm_med)
    out["adj_opm_neg"] = out["adj_opm"] * (out["adj_opm"] < 0)
    out["adj_sgr"] = df["salesgrowth_an"] - ind.map(sgr_med)
    out["lev"] = div(df["acc_dltt"] + df["acc_dlc"], df["book"])
    for col in ("pretret_noa_wr", "roe_wr", "rd_sale_wr"):
        out[col] = df[col]
    return out[BL_FEATURES].replace([np.inf, -np.inf], np.nan)


def ols_fit_predict(X_fit: np.ndarray, y_fit: np.ndarray, X_new: np.ndarray) -> np.ndarray:
    """OLS with intercept (least squares, minimum-norm solution if X is rank deficient)."""
    A = np.column_stack([np.ones(len(X_fit)), X_fit])
    coef, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
    return np.column_stack([np.ones(len(X_new)), X_new]) @ coef
