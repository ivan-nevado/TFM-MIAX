"""LSEG Workspace (Refinitiv) adapter.

Reads the download batches straight from the original zip files (never extracted, never modified):

* ``lseg_casa_trimestrales_*.zip``  quarterly fundamentals, files ``lseg_fund_{final,desl}_NNNNN.csv``
* ``lseg_casa_anuales_*.zip``       annual fundamentals, files ``lseg_fundanual_{final,desl}_NNNNN.csv``
* ``lseg_datos_mercado.zip``        ``lseg_mercado.csv`` and ``lseg_estaticos.csv`` of firms listed today
* ``lseg_casa_mercado_*.zip``       market data and statics of delisted firms, and the SIC codes

``final`` = listed today, ``desl`` = delisted. Facts established on the raw files (see DECISIONS.md):

* Fundamentals are in USD (converted here to USD millions); flows are single-quarter amounts.
* About two thirds of the downloaded rows are blank padding (every item empty) and are dropped.
* Market data of listed firms: ``mv`` in USD millions, ``ret`` in decimals, month-end dates.
* Market data of delisted firms: ``mv`` in USD, ``ret`` in percent, last-trading-day dates.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from relval.data.schemas import ALL_ITEMS, FUND_COLUMNS, MARKET_COLUMNS, STATIC_COLUMNS, RawTables

USD_PER_MILLION = 1e6
EXCHANGE_BY_SUFFIX = {"N": "NYSE", "A": "AMEX", "OQ": "NASDAQ", "O": "NASDAQ"}
REIT_TRBC_PREFIX = "601020"  # TRBC industry group "Residential & Commercial REITs"
FINANCIAL_SECTORS = {"Financials", "Real Estate"}


def _read_members(zips: list[Path], prefix: str) -> pd.DataFrame:
    frames = []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for member in sorted(zf.namelist()):
                if not member.startswith(prefix) or "errores" in member:
                    continue
                with zf.open(member) as fh:
                    df = pd.read_csv(fh)
                df["origen"] = "desl" if "_desl_" in member else "final"
                frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no member starting with {prefix!r} in {[z.name for z in zips]}")
    return pd.concat(frames, ignore_index=True)


def ric_root(ric: pd.Series) -> pd.Series:
    """RIC without the delisting marker: ``GORO.A^G26`` -> ``GORO.A``."""
    return ric.str.replace(r"\^.*$", "", regex=True)


def exchange_from_ric(ric: pd.Series) -> pd.Series:
    suffix = ric_root(ric).str.extract(r"\.([A-Za-z]+)$")[0]
    return suffix.map(EXCHANGE_BY_SUFFIX)


class LsegAdapter:
    name = "lseg"

    def __init__(self, directory: Path):
        self.dir = Path(directory)

    # ---- fundamentals --------------------------------------------------------------
    def _fundamentals(self, zip_glob: str, prefix: str) -> pd.DataFrame:
        zips = sorted(self.dir.glob(zip_glob))
        if not zips:
            raise FileNotFoundError(f"{self.dir}/{zip_glob} not found")
        df = _read_members(zips, prefix).rename(columns={"fecha": "datadate"})
        # Minority interest in income = net income before minus after minority interest.
        df["mii"] = df["ni_antes_mi"] - df["ni_despues_mi"]
        # Reported EBITDA plays the role of Compustat oibdp (operating income before depreciation).
        df["oibdp"] = df["ebitda_lseg"]
        raw_items = [c for c in ALL_ITEMS if c not in ("mii", "oibdp")] + [
            "ni_antes_mi",
            "ni_despues_mi",
            "ebitda_lseg",
        ]
        df = df[df[raw_items].notna().any(axis=1)].copy()  # drop the blank padding rows
        df["datadate"] = pd.to_datetime(df["datadate"])
        df[ALL_ITEMS] = df[ALL_ITEMS].astype(float) / USD_PER_MILLION
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.sort_values(["id", "datadate"]).drop_duplicates(["id", "datadate"], keep="last")
        return df[FUND_COLUMNS].reset_index(drop=True)

    # ---- statics -------------------------------------------------------------------
    def _static(self) -> pd.DataFrame:
        with zipfile.ZipFile(self.dir / "lseg_datos_mercado.zip") as zf, zf.open("lseg_estaticos.csv") as fh:
            listed = pd.read_csv(fh).assign(origen="final")
        market_zips = sorted(self.dir.glob("lseg_casa_mercado_*.zip"))
        delisted = _read_members(market_zips, "lseg_estaticos_desl_")
        sic = _read_members(market_zips, "lseg_sic_")[["id", "sic"]].drop_duplicates("id")
        st = (
            pd.concat([listed, delisted], ignore_index=True)
            .drop_duplicates("id")
            .merge(sic, on="id", how="left")
        )

        isin = st["isin"].astype("string")
        cusip = st["cusip"].astype("string")
        st["is_us"] = (isin.str.startswith("US") | (isin.isna() & cusip.str[:1].str.isdigit())).fillna(False)
        trbc = pd.to_numeric(st["trbc_industria"], errors="coerce").astype("Int64").astype("string")
        name = st["nombre"].astype("string")
        st["is_reit"] = (
            trbc.str.startswith(REIT_TRBC_PREFIX) | name.str.contains(r"\bREIT\b", case=False, regex=True)
        ).fillna(False)
        st["sic"] = pd.to_numeric(st["sic"], errors="coerce")
        # cash2assets_an is missing for financial firms (SIC 6000-6999). Where LSEG has no SIC
        # code the TRBC economic sector is used instead.
        by_sic = st["sic"].between(6000, 6999)
        by_sector = st["trbc_sector"].isin(FINANCIAL_SECTORS)
        st["is_financial"] = np.where(st["sic"].notna(), by_sic, by_sector).astype(bool)
        st["exchange"] = exchange_from_ric(st["id"])
        st = st.rename(columns={"nombre": "name", "trbc_sector": "sector"})
        return st[STATIC_COLUMNS].reset_index(drop=True)

    # ---- market --------------------------------------------------------------------
    def _market(self, static: pd.DataFrame) -> pd.DataFrame:
        with zipfile.ZipFile(self.dir / "lseg_datos_mercado.zip") as zf, zf.open("lseg_mercado.csv") as fh:
            listed = pd.read_csv(fh).assign(origen="final")
        listed["date"] = pd.to_datetime(listed["date"])

        delisted = _read_members(sorted(self.dir.glob("lseg_casa_mercado_*.zip")), "lseg_mercado_desl_")
        delisted = delisted.rename(columns={"fecha": "date"})
        delisted["date"] = pd.to_datetime(delisted["date"]) + pd.offsets.MonthEnd(0)
        delisted["ret"] = delisted["ret"] / 100.0  # percent -> decimal
        delisted["mv"] = delisted["mv"] / USD_PER_MILLION  # USD -> USD millions

        mk = pd.concat([listed, delisted], ignore_index=True)
        mk = mk[mk[["ret", "mv"]].notna().any(axis=1)]
        mk = mk.sort_values(["id", "date"]).drop_duplicates(["id", "date"], keep="last")
        # Universe proxy for CRSP shrcd 10/11 and exchcd 1/2/3: US-incorporated, not a REIT, and a
        # RIC on NYSE, NYSE American or NASDAQ. The exchange is today's (or the last) one.
        flags = static.assign(
            in_universe=static["is_us"] & ~static["is_reit"] & static["exchange"].notna(),
            nyse=static["exchange"].eq("NYSE"),
        )[["id", "in_universe", "nyse"]]
        mk = mk.merge(flags, on="id", how="left")
        mk[["in_universe", "nyse"]] = mk[["in_universe", "nyse"]].fillna(False).astype(bool)
        return mk[MARKET_COLUMNS].reset_index(drop=True)

    def load(self) -> RawTables:
        static = self._static()
        return RawTables(
            fund_q=self._fundamentals("lseg_casa_trimestrales_*.zip", "lseg_fund_"),
            fund_a=self._fundamentals("lseg_casa_anuales_*.zip", "lseg_fundanual_"),
            market=self._market(static),
            static=static,
        )
