"""Accounting records: one row per firm and fiscal period with annualised flows and one-year lags.

Paper, appendix A.1: income-statement and cash-flow items are annualised by summing quarters
t-3 to t; balance-sheet items are those of quarter t. Prefix ``l_`` is the same quantity one
year (four quarters) earlier and ``d_x = x - l_x``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from relval.data.schemas import ALL_ITEMS, BALANCE_ITEMS, FLOW_ITEMS


def period_month(datadate: pd.Series) -> pd.Series:
    """Month index (year*12 + month-1) of the fiscal period.

    52/53-week fiscal calendars end a few days before or after the calendar month end
    (e.g. 2026-09-25 or 2026-10-04 for a September quarter); shifting by ten days assigns both
    to September.
    """
    shifted = datadate - pd.Timedelta(days=10)
    return (shifted.dt.year * 12 + shifted.dt.month - 1).astype("int64")


def _lag_positions(ids: np.ndarray, pm: np.ndarray, months: int, tolerance: int) -> np.ndarray:
    """Row position of the record of the same firm ``months`` earlier (-1 if there is none).

    Exact match first, then +-1..tolerance months to absorb changes in fiscal calendars.
    """
    n = len(ids)
    key = pd.DataFrame({"id": ids, "pm": pm, "pos": np.arange(n)})
    out = np.full(n, -1, dtype=np.int64)
    offsets = [0] + [s * k for k in range(1, tolerance + 1) for s in (-1, 1)]
    for off in offsets:
        want = pd.DataFrame({"id": ids, "pm": pm - months + off})
        pos = want.merge(key, on=["id", "pm"], how="left")["pos"].to_numpy()
        take = (out < 0) & ~np.isnan(pos)
        out[take] = pos[take].astype(np.int64)
    return out


def _take(values: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    ok = pos >= 0
    out[ok] = values[pos[ok]]
    return out


def apply_definitions(df: pd.DataFrame, defs: dict) -> pd.DataFrame:
    """Source-to-Compustat definition switches, applied to single-period items before any lag."""
    df = df.copy()
    has_bs = df["at"].notna()
    # Items the vendor leaves blank when the firm has none (Compustat reports them as zero).
    for col in defs.get("zero_fill_items") or []:
        df[col] = df[col].where(~(df[col].isna() & has_bs), 0.0)
    if defs.get("xsga_add_xrd", False):
        df["xsga"] = df["xsga"] + df["xrd"].fillna(0.0)
    adj = defs.get("dp_adjustment", "none")
    if adj in ("cogs", "xsga"):
        df[adj] = df[adj] - df["dp"].fillna(0.0)
    elif adj != "none":
        raise ValueError(f"dp_adjustment must be none, cogs or xsga, got {adj!r}")
    if defs.get("ebit_constructed", False):
        df["ebit"] = df["sale"] - df["cogs"] - df["xsga"]
        if adj != "none":
            df["ebit"] = df["ebit"] - df["dp"]
    if defs.get("lt_includes_mib", False):
        df["lt"] = df["lt"] - df["mib"].fillna(0.0)
    return df


def build_records(fund: pd.DataFrame, freq: str, defs: dict, lag_tolerance_months: int = 1) -> pd.DataFrame:
    """Annualise flows and attach one-year lags.

    Parameters
    ----------
    fund : standardised fundamentals (``relval.data.schemas.FUND_COLUMNS``).
    freq : ``"Q"`` (flows are summed over four consecutive quarters, all four required) or
        ``"A"`` (flows are already annual).

    Returns one row per (id, period) with: ``id, datadate, pm, fuente, origen``, the items
    (balance: latest period; flows: annualised) and the same items with prefix ``l_``.
    """
    if freq not in ("Q", "A"):
        raise ValueError("freq must be 'Q' or 'A'")
    df = apply_definitions(fund, defs)
    df["pm"] = period_month(df["datadate"])
    df = df.sort_values(["id", "datadate"]).drop_duplicates(["id", "pm"], keep="last").reset_index(drop=True)
    ids, pm = df["id"].to_numpy(), df["pm"].to_numpy()

    out = df[["id", "datadate", "pm", "origen"]].copy()
    out["fuente"] = "quarterly" if freq == "Q" else "annual"
    values = {c: df[c].to_numpy(dtype=float) for c in ALL_ITEMS}

    if freq == "Q":
        lags = [_lag_positions(ids, pm, 3 * k, lag_tolerance_months) for k in (1, 2, 3)]
        for col in FLOW_ITEMS:
            total = values[col].copy()
            for pos in lags:
                total = total + _take(values[col], pos)  # NaN propagates: all four quarters required
            values[col] = total
    year_ago = _lag_positions(ids, pm, 12, lag_tolerance_months)

    for col in BALANCE_ITEMS + FLOW_ITEMS:
        out[col] = values[col]
    lagged = {f"l_{col}": _take(values[col], year_ago) for col in BALANCE_ITEMS + FLOW_ITEMS}
    return pd.concat([out, pd.DataFrame(lagged, index=out.index)], axis=1)
