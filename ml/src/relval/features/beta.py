"""CAPM beta: slope of the firm's monthly excess return on MKTRF over months t-60 to t-1.

It is the only input built from market data, and it uses returns strictly before the valuation
month, so it cannot contain the month-t price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_beta(y: np.ndarray, x: np.ndarray, window: int, min_obs: int) -> np.ndarray:
    """Beta at position t from observations ``t-window .. t-1`` (consecutive months).

    ``y`` may contain NaN (months without a return); those months are skipped and the estimate
    is missing when fewer than ``min_obs`` valid months remain.
    """
    valid = ~np.isnan(y) & ~np.isnan(x)
    yv, xv = np.where(valid, y, 0.0), np.where(valid, x, 0.0)

    def csum(a: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(a)])

    n_c, sx_c, sy_c, sxx_c, sxy_c = (
        csum(valid.astype(float)),
        csum(xv),
        csum(yv),
        csum(xv * xv),
        csum(xv * yv),
    )
    t = np.arange(len(y))
    lo = np.maximum(t - window, 0)
    n = n_c[t] - n_c[lo]
    sx, sy = sx_c[t] - sx_c[lo], sy_c[t] - sy_c[lo]
    sxx, sxy = sxx_c[t] - sxx_c[lo], sxy_c[t] - sxy_c[lo]
    den = n * sxx - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (n * sxy - sx * sy) / den
    beta[(n < min_obs) | (den <= 0)] = np.nan
    return beta


def compute_betas(
    market: pd.DataFrame, factors: pd.DataFrame, window: int = 60, min_obs: list[int] | None = None
) -> pd.DataFrame:
    """Betas for every firm-month. Returns ``id, date`` and one column per ``min_obs`` value,
    named ``CAPMbeta_<min_obs>``.

    ``market``: id, date (month end), ret (decimal). ``factors``: date, mktrf, rf (decimal).
    """
    min_obs = min_obs or [window]
    fac = factors.set_index("date")[["mktrf", "rf"]]
    out = []
    for firm, grp in market[["id", "date", "ret"]].groupby("id", sort=False):
        s = grp.set_index("date")["ret"].sort_index()
        grid = pd.date_range(s.index.min(), s.index.max(), freq="ME")
        ret = s.reindex(grid).to_numpy(dtype=float)
        f = fac.reindex(grid)
        excess = ret - f["rf"].to_numpy(dtype=float)
        x = f["mktrf"].to_numpy(dtype=float)
        frame = pd.DataFrame({"id": firm, "date": grid})
        for m in min_obs:
            frame[f"CAPMbeta_{m}"] = rolling_beta(excess, x, window, m)
        out.append(frame[frame["date"].isin(s.index)])
    return pd.concat(out, ignore_index=True)
