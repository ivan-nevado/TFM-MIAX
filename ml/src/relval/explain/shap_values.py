"""SHAP-based variable importance (paper, section 5.2; Tables 7, 8, 9, IA.8; Figure 5).

Raw TreeSHAP values of the out-of-sample observations are stored by the training stage. Here they
are turned into the paper's measure: for each observation the absolute values are scaled so the
97 sum to 100%; importance is the average of these scaled values, in percentage points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from relval.features.registry import CATEGORIES, CATEGORY_OF, DUPLICATE_PAIRS, FEATURES, ORIENTATION


def scaled_abs_shap(raw: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    """Scaled absolute SHAP in percent: each row sums to 100 (rows with all-zero SHAP stay zero)."""
    a = np.abs(raw[features].to_numpy(dtype=float))
    total = a.sum(axis=1, keepdims=True)
    return pd.DataFrame(
        100.0 * a / np.where(total > 0, total, np.nan), columns=features, index=raw.index
    ).fillna(0.0)


def decade(date: pd.Series) -> pd.Series:
    return (date.dt.year // 10 * 10).astype(int)


def variable_importance(scaled: pd.DataFrame, date: pd.Series) -> pd.DataFrame:
    """Mean scaled SHAP per variable: full sample and one column per decade (``d1990`` ...)."""
    out = pd.DataFrame({"variable": scaled.columns, "shap": scaled.mean().to_numpy()})
    out["category"] = out["variable"].map(CATEGORY_OF)
    by_dec = scaled.groupby(decade(date).to_numpy()).mean().T
    by_dec.columns = [f"d{int(c)}" for c in by_dec.columns]
    out = out.merge(by_dec, left_on="variable", right_index=True, how="left")
    dec_cols = sorted(c for c in out.columns if c.startswith("d") and c[1:].isdigit())
    if len(dec_cols) >= 2:
        out["chg"] = out[dec_cols[-1]] - out[dec_cols[0]]
    return out


def category_importance(var_imp: pd.DataFrame) -> pd.DataFrame:
    """Category importance = sum of the importance of its variables."""
    num = [c for c in var_imp.columns if c not in ("variable", "category")]
    return var_imp.groupby("category", sort=False)[num].sum().reset_index()


def duplicate_pair_importance(var_imp: pd.DataFrame) -> pd.DataFrame:
    """Importance of each exact-duplicate pair summed (the split inside a pair is arbitrary)."""
    imp = var_imp.set_index("variable")["shap"]
    return pd.DataFrame(
        [
            {"pair": f"{a} + {b}", "shap_a": imp[a], "shap_b": imp[b], "shap_sum": imp[a] + imp[b]}
            for a, b in DUPLICATE_PAIRS
        ]
    )


def rank_pct(df: pd.DataFrame, by: pd.Series) -> pd.DataFrame:
    """Rank percentiles within month."""
    return df.groupby(by.to_numpy()).rank(pct=True)


def rank_correlations(
    features: pd.DataFrame, yhat: pd.Series, date: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """``rho`` of Tables 7 and 8.

    Variable: correlation between the within-month rank percentile of the variable and that of
    the predicted target. Category: the same with the average rank percentile of the category's
    variables, each re-oriented so a higher rank means more of the category label.
    """
    ranks = rank_pct(features[FEATURES], date)
    target_rank = yhat.groupby(date.to_numpy()).rank(pct=True)
    var_rho = ranks.corrwith(target_rank)
    oriented = ranks.copy()
    for col, sign in ORIENTATION.items():
        if sign < 0:
            oriented[col] = 1.0 - oriented[col]
    cat_rho = pd.Series(
        {cat: oriented[members].mean(axis=1).corr(target_rank) for cat, members in CATEGORIES.items()}
    )
    return var_rho, cat_rho


def waterfall(raw_row: pd.Series, prediction: float, top: int = 5) -> pd.DataFrame:
    """Raw SHAP waterfall of one observation: base value, the ``top`` contributions, "other",
    prediction (Figure 5)."""
    contrib = raw_row[FEATURES].astype(float)
    order = contrib.abs().sort_values(ascending=False).index
    rows = [{"label": "Training mean", "value": float(raw_row["base_value"]), "kind": "base"}]
    rows += [{"label": v, "value": float(contrib[v]), "kind": "contribution"} for v in order[:top]]
    rows.append({"label": "other", "value": float(contrib[order[top:]].sum()), "kind": "contribution"})
    rows.append({"label": "Prediction", "value": float(prediction), "kind": "total"})
    return pd.DataFrame(rows)
