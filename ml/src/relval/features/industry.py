"""Fama-French industry classifications from SIC codes (49 industries as input, 12 for reporting)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from relval.data import french


def classify(sic: pd.Series, definition_zip: Path, n_industries: int) -> tuple[pd.Series, dict[int, str]]:
    """Industry number for each SIC code. A SIC outside every range is the last industry
    ("Other"); a missing SIC gives a missing industry."""
    table, names = french.sic_lookup(definition_zip, n_industries)
    code = pd.to_numeric(sic, errors="coerce")
    ok = code.notna() & code.between(0, 9999)
    out = pd.Series(np.nan, index=sic.index, dtype=float)
    out[ok] = table[code[ok].astype(int).to_numpy()]
    return out, names


def add_industries(static: pd.DataFrame, external_dir: Path) -> pd.DataFrame:
    """Add ``industry`` (FF49), ``ff12`` and their short names to the static table."""
    st = static.copy()
    st["industry"], names49 = classify(st["sic"], external_dir / french.FILES["sic49"], 49)
    st["ff49_name"] = st["industry"].map(names49)
    path12 = external_dir / french.FILES["sic12"]
    if path12.exists():
        st["ff12"], names12 = classify(st["sic"], path12, 12)
        st["ff12_name"] = st["ff12"].map(names12)
    else:  # file not downloaded yet: keep the columns so downstream schemas are stable
        st["ff12"], st["ff12_name"] = np.nan, None
    return st
