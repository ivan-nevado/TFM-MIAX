"""Derived accounting quantities: book equity (Davis, Fama & French 2000) and the paper's aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def nz(x: pd.Series) -> pd.Series:
    """Missing treated as zero (Stata ``cond(missing(x), 0, x)``)."""
    return x.fillna(0.0)


def sas_sum(*args: pd.Series) -> pd.Series:
    """SAS ``sum()``: ignores missing arguments, missing only if all arguments are missing."""
    frame = pd.concat(args, axis=1)
    return frame.sum(axis=1, min_count=1)


def rowtotal(*args: pd.Series) -> pd.Series:
    """Stata ``rowtotal``: missing arguments count as zero (zero if all are missing)."""
    return pd.concat(args, axis=1).sum(axis=1, min_count=0)


def div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division that gives missing for a zero or missing denominator (never +-inf)."""
    den = den.where(den != 0)
    return (num / den).replace([np.inf, -np.inf], np.nan)


def derive(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Add derived quantities for the current period (``prefix=""``) or one year earlier (``"l_"``).

    * ``seq``    stockholders' equity = ceq + pstk; if ceq is missing, at - lt - mib.
    * ``be``     book equity (Davis, Fama & French 2000) = seq + txditc - pstk.
    * ``ebitda`` ebit + dp; where dp is missing, the reported value ``oibdp`` if the source has one.
    * ``icapt``  invested capital = dltt + pstk + ceq + mib (ceq required, the rest as zero if missing).
    * ``earnat`` total earning assets = ppent + act;  ``noa`` = ppent + act - lct.
    * ``ibcom``  income available to common = ibadj, else ib - preferred dividends.
    """
    p = prefix
    g = lambda name: df[p + name]  # noqa: E731
    seq = (g("ceq") + nz(g("pstk"))).where(g("ceq").notna(), g("at") - g("lt") - nz(g("mib")))
    df[p + "seq"] = seq
    df[p + "be"] = seq + nz(g("txditc")) - nz(g("pstk"))
    ebitda = g("ebit") + g("dp")
    df[p + "ebitda"] = ebitda.where(ebitda.notna(), g("oibdp")) if p + "oibdp" in df.columns else ebitda
    df[p + "icapt"] = g("ceq") + nz(g("dltt")) + nz(g("pstk")) + nz(g("mib"))
    df[p + "earnat"] = g("ppent") + g("act")
    df[p + "noa"] = g("ppent") + g("act") - g("lct")
    df[p + "ibcom"] = g("ibadj").where(g("ibadj").notna(), g("ib") - nz(g("prefdiv")))
    return df
