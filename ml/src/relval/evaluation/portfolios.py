"""Valuation errors and future returns (paper, section 4.3; Table 3 and Table IA.7).

At the end of month t stocks are sorted into quintiles by valuation error using NYSE breakpoints.
Quintile 1 holds the most expensive stocks according to the model (most negative error), quintile
5 the cheapest. Portfolios are held during month t+1; the strategy is long quintile 5 and short
quintile 1. Value weights use month-t equity value. Returns of t+1 are used for every stock that
has one (no size filter at t+1). t-statistics use Newey-West standard errors with three lags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

FF6 = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]


def nyse_breakpoints(values: np.ndarray, nyse: np.ndarray, probs: np.ndarray) -> np.ndarray | None:
    """Quantiles of ``values`` among NYSE firms only."""
    v = values[nyse & ~np.isnan(values)]
    return np.quantile(v, probs) if len(v) >= len(probs) + 1 else None


def assign_quantiles(values: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """Portfolio number 1..len(breaks)+1 (value <= first breakpoint -> 1)."""
    return np.searchsorted(breaks, values, side="left") + 1


def sorted_portfolio_returns(
    data: pd.DataFrame, n: int = 5, weighting: str = "vw", ew_min_size_quantile: float | None = None
) -> pd.DataFrame:
    """Monthly returns of the ``n`` portfolios and of the long-short strategy.

    ``data`` columns: id, date (formation month t), eps, M (equity value at t), nyse (bool),
    ret_next (return of month t+1). The output is indexed by the holding month t+1.
    """
    probs = np.arange(1, n) / n
    rows = []
    for date, grp in data.groupby("date", sort=True):
        eps = grp["eps"].to_numpy(dtype=float)
        nyse = grp["nyse"].to_numpy(dtype=bool)
        size = grp["M"].to_numpy(dtype=float)
        ret = grp["ret_next"].to_numpy(dtype=float)
        breaks = nyse_breakpoints(eps, nyse, probs)
        if breaks is None:
            continue
        q = assign_quantiles(eps, breaks)
        keep = ~np.isnan(ret) & ~np.isnan(eps) & (size > 0)
        if weighting == "ew" and ew_min_size_quantile:
            floor = nyse_breakpoints(size, nyse, np.array([ew_min_size_quantile]))
            if floor is not None:
                keep &= size >= floor[0]
        row = {"date": date + pd.offsets.MonthEnd(1)}
        for k in range(1, n + 1):
            sel = keep & (q == k)
            if not sel.any():
                row[f"q{k}"] = np.nan
            elif weighting == "vw":
                row[f"q{k}"] = float(np.average(ret[sel], weights=size[sel]))
            else:
                row[f"q{k}"] = float(ret[sel].mean())
            row[f"n{k}"] = int(sel.sum())
        row["ls"] = row[f"q{n}"] - row["q1"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def newey_west(y: np.ndarray, X: np.ndarray, lags: int = 3):
    """OLS with Newey-West (HAC) standard errors."""
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})


def performance(ls: pd.Series, factors: pd.DataFrame, lags: int = 3) -> dict[str, float]:
    """Mean return, Sharpe ratio and six-factor alpha of a long-short return series (decimals in,
    percentage points out). Sharpe and information ratios are annualised."""
    ls = ls.dropna()
    out: dict[str, float] = {"n_months": int(len(ls))}
    if len(ls) < 24:
        return out
    y = ls.to_numpy() * 100.0
    mean_fit = newey_west(y, np.ones((len(y), 1)), lags)
    out.update(
        {
            "mean": float(mean_fit.params[0]),
            "mean_t": float(mean_fit.tvalues[0]),
            "mean_p": float(mean_fit.pvalues[0]),
        }
    )
    out["sharpe"] = float(y.mean() / y.std(ddof=1) * np.sqrt(12))
    fac = (
        factors.set_index("date").reindex(ls.index)[FF6] * 100.0 if set(FF6) <= set(factors.columns) else None
    )
    if fac is None or fac.isna().any().any():
        return out
    fit = newey_west(y, sm.add_constant(fac.to_numpy()), lags)
    out.update(
        {"alpha": float(fit.params[0]), "alpha_t": float(fit.tvalues[0]), "alpha_p": float(fit.pvalues[0])}
    )
    for i, name in enumerate(FF6, start=1):
        out[name] = float(fit.params[i])
        out[f"{name}_p"] = float(fit.pvalues[i])
    out["adj_r2"] = float(fit.rsquared_adj)
    out["info_ratio"] = float(fit.params[0] / np.std(fit.resid, ddof=len(FF6) + 1) * np.sqrt(12))
    return out


def portfolio_table(perf: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Table 3 layout: rows (panel, row), one column per model, coefficients with stars."""

    def cell(model: str, key: str, pkey: str | None = None) -> str:
        d = perf.get(model, {})
        if key not in d or pd.isna(d[key]):
            return ""
        return f"{d[key]:.2f}{stars(d[pkey]) if pkey and pkey in d else ''}"

    layout = [("Mean", "Const", "mean", "mean_p"), ("Mean", "Sharpe Ratio", "sharpe", None)]
    layout += [("FF6", name.upper(), name, f"{name}_p") for name in FF6]
    layout += [("FF6", "Const", "alpha", "alpha_p"), ("FF6", "Adjusted R2", "adj_r2", None),
               ("FF6", "Information Ratio", "info_ratio", None), ("Sample", "Months", "n_months", None)]  # fmt: skip
    rows = []
    for panel, row, key, pkey in layout:
        rec = {"panel": panel, "row": row}
        for model in perf:
            rec[model] = str(perf[model].get(key, "")) if key == "n_months" else cell(model, key, pkey)
        rows.append(rec)
    return pd.DataFrame(rows)


def run_portfolios(
    preds: pd.DataFrame,
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    models: list[str],
    pcfg: dict,
    weighting: str = "vw",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """Returns (Table 3 layout, long-short return series per model, raw statistics)."""
    info = panel[["id", "date", "nyse", "ret_next"]]
    series, perf = {}, {}
    for model in models:
        p = preds.loc[preds["model"] == model, ["id", "date", "eps", "M"]]
        if p.empty:
            continue
        rets = sorted_portfolio_returns(
            p.merge(info, on=["id", "date"], how="left"),
            int(pcfg["n_quantiles"]), weighting,
            float(pcfg["ew_min_size_quantile"]) if weighting == "ew" else None,
        )  # fmt: skip
        if rets.empty:
            continue
        series[model] = rets["ls"]
        perf[model] = performance(rets["ls"], factors, int(pcfg["nw_lags"]))
    returns = pd.DataFrame(series)
    return portfolio_table(perf), returns, perf
