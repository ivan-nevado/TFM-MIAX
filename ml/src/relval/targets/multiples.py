"""Targets, conversion of a prediction to equity value, and the valuation error.

Paper, section 4.1 and online appendix IA.4 (section 3.2 of the master prompt):

    M = market value of equity, B = book equity, A = total assets, S = annualised sales,
    D = rowtotal(dltt, dlc, pstk), V = M + D
    lnm2b = ln(M/B),  lnv2a = ln((M+D)/A),  lnv2s = ln((M+D)/S)

Predicted equity value, with no retransformation correction:

    lnm2b: Mhat = exp(yhat) * B       lnv2a: Mhat = exp(yhat) * A - D       lnv2s: Mhat = exp(yhat) * S - D

Valuation error: eps = Mhat / M - 1 (negative = the model values the firm below the market).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOG_TARGETS = ["lnm2b", "lnv2a", "lnv2s"]
MULTIPLES = ["m2b", "v2a", "v2s"]
# target name -> (multiple it refers to or None for equity value, whether it is in logs)
TARGET_INFO: dict[str, tuple[str | None, bool]] = {
    "lnm2b": ("m2b", True), "lnv2a": ("v2a", True), "lnv2s": ("v2s", True),
    "m2b": ("m2b", False), "v2a": ("v2a", False), "v2s": ("v2s", False),
    "lnM": (None, True), "M": (None, False),
}  # fmt: skip
SCALE_OF = {"m2b": "B", "v2a": "A", "v2s": "S"}


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add D, V and the three log targets. Requires columns M, B, A, S, dltt, dlc, pstk.

    A target is missing unless numerator and denominator are both positive.
    """
    out = df.copy()
    out["D"] = out[["dltt", "dlc", "pstk"]].sum(axis=1, min_count=0)
    out["V"] = out["M"] + out["D"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["lnm2b"] = np.log((out["M"] / out["B"]).where((out["M"] > 0) & (out["B"] > 0)))
        out["lnv2a"] = np.log((out["V"] / out["A"]).where((out["V"] > 0) & (out["A"] > 0) & (out["M"] > 0)))
        out["lnv2s"] = np.log((out["V"] / out["S"]).where((out["V"] > 0) & (out["S"] > 0) & (out["M"] > 0)))
    return out


def target_values(df: pd.DataFrame, target: str) -> pd.Series:
    """Values of any supported target from a panel that has the three log targets and M."""
    if target in LOG_TARGETS:
        return df[target]
    if target in MULTIPLES:
        return np.exp(df["ln" + target])
    if target == "lnM":
        return np.log(df["M"].where(df["M"] > 0))
    if target == "M":
        return df["M"]
    raise KeyError(f"unknown target {target!r}")


def to_equity_value(yhat, target: str, B, A, S, D):
    """Predicted equity value in USD millions for a prediction of ``target``. Can be negative for
    heavily indebted firms; those observations are kept and counted, as the paper is silent."""
    multiple, is_log = TARGET_INFO[target]
    level = np.exp(yhat) if is_log else np.asarray(yhat, dtype=float)
    if multiple is None:
        return level
    if multiple == "m2b":
        return level * np.asarray(B, dtype=float)
    scale = np.asarray(A if multiple == "v2a" else S, dtype=float)
    return level * scale - np.asarray(D, dtype=float)


def valuation_error(m_hat, m):
    """eps = Mhat / M - 1."""
    return np.asarray(m_hat, dtype=float) / np.asarray(m, dtype=float) - 1.0
