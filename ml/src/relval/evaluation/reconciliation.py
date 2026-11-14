"""Sources of the performance gains (paper, section 4.4 and Table 4).

Rows 1-4 change one thing at a time on the observations that have all ten BL variables:
OLS on the multiple -> OLS on the log multiple -> GBM with the BL variables -> GBM with all 97.
Rows 5-6 compare LASSO and GBM on the complete-case subsample.
"""

from __future__ import annotations

import pandas as pd

from relval.evaluation.metrics import compute_metrics
from relval.models.specs import SPECS, TABLE4_ROWS

COLUMNS = ["N", "med_abs_eps", "avg_abs_eps", "rmse_y", "rmse_M", "r2oos_y", "r2oos_M", "rho"]


def reconciliation_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for panel, models in TABLE4_ROWS.items():
        for name in models:
            met = compute_metrics(preds[preds["model"] == name])
            rows.append(
                {"panel": panel, "model": SPECS[name].label, "model_id": name, **{c: met[c] for c in COLUMNS}}
            )
    return pd.DataFrame(rows)
