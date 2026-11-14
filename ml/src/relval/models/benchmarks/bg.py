"""BG: monthly cross-sectional OLS of market value on accounting items (Bartram & Grinblatt 2018).

The paper regresses market capitalisation on 28 Compustat items (online appendix Table IA.2):
at, seq, icapt, teq, pstkr, ppent, ceq, pstk, dltt, ao, lt, lo, che, aco, lco, ap, dv, dvp, sale,
ib, ni, xido, ibadj, ibcom, pi, txt, nopi, do. Our download has 16 of them
(``benchmarks.bg_items``); the result is labelled approximate in every report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from relval.models.benchmarks.bl import ols_fit_predict


def predict_bg(
    df: pd.DataFrame, fit: np.ndarray, test: np.ndarray, items: list[str], zero_fill: list[str] | None = None
) -> pd.DataFrame:
    """Predicted M (USD millions) for test rows with every item present.

    Items in ``zero_fill`` are blank in the source when the firm has none and are set to zero.
    """
    cols = ["acc_" + c for c in items]
    X = df[cols].copy()
    for item in zero_fill or []:
        if "acc_" + item in X:
            X["acc_" + item] = X["acc_" + item].fillna(0.0)
    complete = X.notna().all(axis=1).to_numpy() & df["M"].notna().to_numpy()
    f, t = fit & complete, test & complete
    empty = pd.DataFrame(columns=["yhat", "n_train", "train_mean_y", "train_mean_M"])
    if f.sum() <= len(cols) + 1 or t.sum() == 0:
        return empty
    yhat = ols_fit_predict(X[f].to_numpy(), df.loc[f, "M"].to_numpy(), X[t].to_numpy())
    mean_m = float(df.loc[f, "M"].mean())
    return pd.DataFrame(
        {"yhat": yhat, "n_train": int(f.sum()), "train_mean_y": mean_m, "train_mean_M": mean_m},
        index=df.index[t],
    )
