"""RRV: industry-month regression of log equity value (Rhodes-Kropf, Robinson & Viswanathan 2005).

Variables of the paper (online appendix Table IA.1): ln M on ln B, ln|NI|, a loss indicator
interacted with ln|NI|, and leverage. The appendix lists three leverage measures:

    LEV1 = (dltt + dlc) / at            (Golubov & Konstantinidi 2019)
    LEV2 = 1 - book / at                (book leverage, RRV 2005)
    LEV3 = 1 - M / (M + book debt)      (market leverage, RRV 2005)

LEV3 contains the market value being predicted, so it is never used here (it would leak the
target); LEV1 and LEV2 are. Minimum 30 fitting firms per industry-month (paper silent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from relval.models.benchmarks.bl import ols_fit_predict

RRV_FEATURES = ["ln_b", "ln_abs_ni", "loss_ln_abs_ni", "lev_debt", "lev_book"]


def rrv_features(df: pd.DataFrame) -> pd.DataFrame:
    ni = df["acc_ib"]
    out = pd.DataFrame(index=df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ln_b"] = np.log(df["B"].where(df["B"] > 0))
        out["ln_abs_ni"] = np.log(ni.abs().where(ni.abs() > 0))
    out["loss_ln_abs_ni"] = out["ln_abs_ni"] * (ni < 0)
    out["lev_debt"] = (df["acc_dltt"] + df["acc_dlc"]) / df["A"]
    out["lev_book"] = 1.0 - df["B"] / df["A"]
    return out[RRV_FEATURES].replace([np.inf, -np.inf], np.nan)


def predict_rrv(df: pd.DataFrame, fit: np.ndarray, test: np.ndarray, min_firms: int) -> pd.DataFrame:
    """Predicted ln M for test rows. Returns columns yhat, n_train, train_mean_y, train_mean_M."""
    feats = rrv_features(df)
    y = np.log(df["M"])
    complete = feats.notna().all(axis=1).to_numpy() & y.notna().to_numpy() & df["industry"].notna().to_numpy()
    rows = []
    for _industry, idx in df.index.to_series().groupby(df["industry"]):
        pos = idx.to_numpy()
        f = pos[fit[pos] & complete[pos]]
        t = pos[test[pos] & complete[pos]]
        if len(f) < min_firms or len(t) == 0:
            continue
        yhat = ols_fit_predict(feats.loc[f].to_numpy(), y.loc[f].to_numpy(), feats.loc[t].to_numpy())
        rows.append(
            pd.DataFrame(
                {
                    "yhat": yhat,
                    "n_train": len(f),
                    "train_mean_y": y.loc[f].mean(),
                    "train_mean_M": df.loc[f, "M"].mean(),
                },
                index=t,
            )
        )
    return (
        pd.concat(rows) if rows else pd.DataFrame(columns=["yhat", "n_train", "train_mean_y", "train_mean_M"])
    )
