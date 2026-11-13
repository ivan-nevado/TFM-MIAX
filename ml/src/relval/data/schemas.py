"""Fixed schemas of the standardised raw tables every source adapter must return.

Units: money in USD millions, returns in decimals, dates as month-end timestamps for market data
and fiscal period-end timestamps for fundamentals. Income-statement and cash-flow items are
single-period amounts (one quarter in the quarterly table, one year in the annual table).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Compustat-named accounting items.
BALANCE_ITEMS = [
    "at", "act", "lct", "lt", "che", "invt", "rect", "ap", "ppent",
    "dltt", "dlc", "pstk", "ceq", "txditc", "mib",
]  # fmt: skip
FLOW_ITEMS = [
    "sale", "cogs", "xsga", "xrd", "xint", "ebit", "dp", "pi", "txt", "ib",
    "prefdiv", "ibadj", "ocf", "capx", "dv", "mii", "nopi", "spi", "oibdp",
]  # fmt: skip
ALL_ITEMS = BALANCE_ITEMS + FLOW_ITEMS

FUND_COLUMNS = ["id", "datadate", "origen", *ALL_ITEMS]
MARKET_COLUMNS = ["id", "date", "ret", "mv", "in_universe", "nyse", "origen"]
STATIC_COLUMNS = [
    "id", "name", "ticker", "sector", "sic", "exchange", "is_us", "is_reit", "is_financial", "origen",
]  # fmt: skip


@dataclass
class RawTables:
    """Standardised output of a source adapter."""

    fund_q: pd.DataFrame  # quarterly fundamentals, FUND_COLUMNS
    fund_a: pd.DataFrame  # annual fundamentals, FUND_COLUMNS (may be empty)
    market: pd.DataFrame  # monthly market data, MARKET_COLUMNS
    static: pd.DataFrame  # one row per firm, STATIC_COLUMNS


class SchemaError(ValueError):
    pass


def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}")


def _check_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    dup = int(df.duplicated(keys).sum())
    if dup:
        raise SchemaError(f"{name}: {dup} duplicated rows on {keys}")


def validate(raw: RawTables) -> dict[str, int]:
    """Structural and unit checks. Returns row counts for the manifest."""
    _check_columns(raw.fund_q, FUND_COLUMNS, "fund_q")
    _check_columns(raw.fund_a, FUND_COLUMNS, "fund_a")
    _check_columns(raw.market, MARKET_COLUMNS, "market")
    _check_columns(raw.static, STATIC_COLUMNS, "static")
    _check_unique(raw.fund_q, ["id", "datadate"], "fund_q")
    _check_unique(raw.fund_a, ["id", "datadate"], "fund_a")
    _check_unique(raw.market, ["id", "date"], "market")
    _check_unique(raw.static, ["id"], "static")

    for name, df in (("fund_q", raw.fund_q), ("fund_a", raw.fund_a)):
        values = df[ALL_ITEMS].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise SchemaError(f"{name}: infinite values")
        med_at = float(df["at"].median()) if len(df) else np.nan
        if len(df) and not (1.0 < med_at < 1e6):
            raise SchemaError(f"{name}: median total assets {med_at:,.1f} is not in USD millions")

    if not raw.market["date"].dt.is_month_end.all():
        raise SchemaError("market: dates must be month ends")
    med_mv = float(raw.market["mv"].median())
    med_ret = float(raw.market["ret"].abs().median())
    if not (1.0 < med_mv < 1e6):
        raise SchemaError(f"market: median market value {med_mv:,.1f} is not in USD millions")
    if not (0.0 < med_ret < 0.5):
        raise SchemaError(f"market: median absolute return {med_ret:.3f} is not a decimal return")
    if not set(raw.market["id"]).issubset(set(raw.static["id"])):
        raise SchemaError("market: ids without a static record")

    return {
        "fund_q": len(raw.fund_q),
        "fund_a": len(raw.fund_a),
        "market": len(raw.market),
        "static": len(raw.static),
    }
