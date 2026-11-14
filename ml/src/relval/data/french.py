"""Ken French data library: monthly factors and SIC-to-industry definitions."""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "ff3": "F-F_Research_Data_Factors_CSV.zip",
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "F-F_Momentum_Factor_CSV.zip",
    "sic49": "Siccodes49.zip",
    "sic12": "Siccodes12.zip",
}
FF6 = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]


def download(external_dir: Path, force: bool = False) -> dict[str, str]:
    """Download the public Ken French files that are missing. Records the download date."""
    import requests

    external_dir.mkdir(parents=True, exist_ok=True)
    log_file = external_dir / "DOWNLOADS.json"
    log = json.loads(log_file.read_text()) if log_file.exists() else {}
    for name in FILES.values():
        target = external_dir / name
        if target.exists() and not force:
            log.setdefault(name, "present before the pipeline ran (date unknown)")
            continue
        resp = requests.get(BASE_URL + name, timeout=60, headers={"User-Agent": "relval-replication"})
        resp.raise_for_status()
        target.write_bytes(resp.content)
        log[name] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_file.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


def _read_zip_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        return zf.read(zf.namelist()[0]).decode("latin-1")


def _monthly_block(text: str) -> pd.DataFrame:
    """Rows ``YYYYMM, v1, v2, ...`` of the first (monthly) block of a Ken French CSV."""
    lines = text.splitlines()
    header, rows = None, []
    for i, line in enumerate(lines):
        if re.match(r"^\s*\d{6}\s*,", line):
            if header is None:
                header = [h.strip() for h in lines[i - 1].split(",")]
            rows.append(line)
        elif rows:
            break  # the annual block follows the monthly one
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None)
    df.columns = ["yyyymm", *header[1 : df.shape[1]]]
    df["date"] = pd.to_datetime(df["yyyymm"].astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    values = df.drop(columns=["yyyymm", "date"]).apply(pd.to_numeric, errors="coerce")
    values = values.where(values > -99.0) / 100.0  # -99.99 marks missing; percent -> decimal
    return pd.concat([df[["date"]], values], axis=1)


def load_factors(external_dir: Path) -> pd.DataFrame:
    """Monthly factors in decimals: date, mktrf, smb, hml, rf and, if downloaded, rmw, cma, umd."""
    ff3 = _monthly_block(_read_zip_text(external_dir / FILES["ff3"]))
    ff3.columns = [c.lower().replace("mkt-rf", "mktrf") for c in ff3.columns]
    out = ff3[["date", "mktrf", "smb", "hml", "rf"]]
    ff5_path, mom_path = external_dir / FILES["ff5"], external_dir / FILES["mom"]
    if ff5_path.exists():
        ff5 = _monthly_block(_read_zip_text(ff5_path))
        ff5.columns = [c.lower().replace("mkt-rf", "mktrf") for c in ff5.columns]
        # Fama & French (2018) six-factor model: the five 2x3 factors plus momentum. The 5-factor
        # file has its own SMB; before July 1963 (not needed here) the 3-factor values are kept.
        five = ff5[["date", "smb", "hml", "rmw", "cma"]]
        out = ff3[["date", "mktrf", "rf"]].merge(five, on="date", how="left")
        old = ff3.set_index("date").reindex(out["date"])
        out["smb"] = out["smb"].fillna(pd.Series(old["smb"].to_numpy(), index=out.index))
        out["hml"] = out["hml"].fillna(pd.Series(old["hml"].to_numpy(), index=out.index))
    if mom_path.exists():
        mom = _monthly_block(_read_zip_text(mom_path))
        mom.columns = ["date", "umd"]
        out = out.merge(mom, on="date", how="left")
    return out.sort_values("date").reset_index(drop=True)


def parse_sic_ranges(path: Path) -> pd.DataFrame:
    """Parse ``SiccodesNN.zip`` into rows (industry number, short name, sic_from, sic_to)."""
    rows, current = [], None
    for line in _read_zip_text(path).splitlines():
        head = re.match(r"^\s*(\d+)\s+(\S+)\s+(.*)$", line)
        rng = re.match(r"^\s*(\d{4})-(\d{4})", line)
        if rng and current is not None:
            rows.append((current[0], current[1], int(rng.group(1)), int(rng.group(2))))
        elif head:
            current = (int(head.group(1)), head.group(2))
    return pd.DataFrame(rows, columns=["industry", "name", "sic_from", "sic_to"])


def sic_lookup(path: Path, n_industries: int) -> tuple[np.ndarray, dict[int, str]]:
    """Array indexed by SIC code (0-9999) with the industry number; unmatched codes get the last
    industry ("Other"), as in the Ken French definitions."""
    ranges = parse_sic_ranges(path)
    table = np.full(10000, n_industries, dtype=np.int16)
    for row in ranges.itertuples(index=False):
        table[row.sic_from : row.sic_to + 1] = row.industry
    names = dict(ranges.drop_duplicates("industry")[["industry", "name"]].itertuples(index=False))
    names.setdefault(n_industries, "Other")
    return table, names
