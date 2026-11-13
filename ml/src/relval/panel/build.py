"""Firm-month panel: accounting records linked to valuation months, the 97 inputs, the three
targets and the Table 1 filters.

Timing (paper, section 3.2): "month" is the valuation month; accounting data are available three
months after the balance-sheet date (a quarter ending 31 March is first used in the June
cross-section). Filters (Table 1): universe -> positive, non-missing targets -> total assets,
book equity and sales each at or above that month's 10th percentile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from relval.data.schemas import ALL_ITEMS, RawTables
from relval.features.accounting import compute_accounting_features
from relval.features.beta import compute_betas
from relval.features.industry import add_industries
from relval.features.registry import ACCOUNTING_FEATURES, CATEGORY_OF, FEATURES
from relval.panel.records import build_records
from relval.targets.multiples import LOG_TARGETS, add_targets

ACC_PREFIX = "acc_"
ID_COLUMNS = ["id", "date", "datadate", "fuente", "origen", "name", "exchange", "nyse", "sic", "ff12"]
VALUE_COLUMNS = ["M", "B", "A", "S", "D", "V", "ret", "ret_next"]
PAPER_TABLE1 = {  # firm-months, firms (1980-2019, 480 months)
    "universe": (2_361_860, 19_697),
    "targets": (2_162_501, 19_041),
    "sample": (1_811_786, 16_201),
}


def month_index(date: pd.Series) -> pd.Series:
    return (date.dt.year * 12 + date.dt.month - 1).astype("int64")


def _records_with_features(fund: pd.DataFrame, freq: str, static: pd.DataFrame, pcfg: dict) -> pd.DataFrame:
    """Usable accounting records (total assets and annualised sales present) with their features."""
    rec = build_records(fund, freq, pcfg["definitions"], pcfg["lag_tolerance_months"])
    rec = rec[rec["at"].notna() & rec["sale"].notna()].reset_index(drop=True)
    fin = rec[["id"]].merge(static[["id", "is_financial"]], on="id", how="left")["is_financial"].fillna(False)
    feats = compute_accounting_features(rec, fin, pcfg["definitions"])
    items = rec[ALL_ITEMS].add_prefix(ACC_PREFIX)
    out = pd.concat([rec[["id", "datadate", "pm", "fuente"]], feats, items], axis=1)
    out["avail"] = out["pm"] + int(pcfg["availability_lag_months"])
    return out


def _link(months: pd.DataFrame, records: pd.DataFrame, max_age: int) -> pd.DataFrame:
    """Latest record available at each firm-month (``avail <= mi``), dropped if older than
    ``max_age`` months. Returns the record columns aligned with ``months`` (NaN where none)."""
    left = months[["id", "mi"]].reset_index().sort_values("mi", kind="stable")
    right = records.sort_values("avail", kind="stable")
    merged = pd.merge_asof(left, right, left_on="mi", right_on="avail", by="id", direction="backward")
    stale = (merged["mi"] - merged["pm"]) > max_age
    value_cols = [c for c in records.columns if c != "id"]
    merged.loc[stale | merged["pm"].isna(), value_cols] = np.nan
    return merged.set_index("index").loc[months.index, value_cols]


def build_panel(raw: RawTables, factors: pd.DataFrame, cfg) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return the panel (rows with valid targets, flag ``in_sample`` for the modelling sample)
    and the control tables (Table 1, summary statistics of inputs and targets)."""
    pcfg = cfg["panel"]
    static = add_industries(raw.static, cfg.path("external"))

    # ---- market data -----------------------------------------------------------------
    market = raw.market[~raw.market["id"].isin(pcfg.get("exclude_ids") or [])].copy()
    max_ret = pcfg.get("max_monthly_return")
    n_extreme = int((market["ret"] > max_ret).sum()) if max_ret else 0
    if max_ret:
        market.loc[market["ret"] > max_ret, "ret"] = np.nan
    market = market.sort_values(["id", "date"]).reset_index(drop=True)
    nxt = market[["id", "date", "ret"]].assign(date=lambda d: d["date"] - pd.offsets.MonthEnd(1))
    market = market.merge(nxt.rename(columns={"ret": "ret_next"}), on=["id", "date"], how="left")

    beta_cfg = pcfg["beta"]
    min_obs = sorted({int(beta_cfg["min_obs"]), *[int(m) for m in beta_cfg.get("extra_min_obs", [])]})
    betas = compute_betas(market, factors, int(beta_cfg["window"]), min_obs)

    start = pd.Timestamp(cfg["period"]["start"]) + pd.offsets.MonthEnd(0)
    end = pd.Timestamp(cfg["period"]["end"]) + pd.offsets.MonthEnd(0)
    months = market[(market["date"] >= start) & (market["date"] <= end)].copy()
    months = months[months["in_universe"] & months["mv"].notna()].reset_index(drop=True)
    months["mi"] = month_index(months["date"])

    # ---- accounting records -> firm-months --------------------------------------------
    rec_q = _records_with_features(raw.fund_q, "Q", static, pcfg)
    linked = _link(months, rec_q, int(pcfg["max_age_months_quarterly"]))
    if len(raw.fund_a):
        rec_a = _records_with_features(raw.fund_a, "A", static, pcfg)
        linked_a = _link(months, rec_a, int(pcfg["max_age_months_annual"]))
        use_annual = linked["pm"].isna() & linked_a["pm"].notna()
        linked.loc[use_annual] = linked_a.loc[use_annual]
    panel = pd.concat([months, linked], axis=1)

    panel = panel.merge(betas, on=["id", "date"], how="left")
    panel["CAPMbeta"] = panel[f"CAPMbeta_{int(beta_cfg['min_obs'])}"]
    panel = panel.merge(static[["id", "name", "exchange", "sic", "industry", "ff12"]], on="id", how="left")

    # ---- targets -----------------------------------------------------------------------
    panel["M"], panel["B"], panel["A"], panel["S"] = (
        panel["mv"],
        panel["book"],
        panel["assets"],
        panel["sale_ac"],
    )
    panel = panel.rename(columns={ACC_PREFIX + c: c for c in ("dltt", "dlc", "pstk")})
    panel = add_targets(panel)
    panel = panel.rename(columns={c: ACC_PREFIX + c for c in ("dltt", "dlc", "pstk")})

    # ---- Table 1 filters ---------------------------------------------------------------
    has_targets = panel[LOG_TARGETS].notna().all(axis=1)
    q = float(pcfg["size_filter_quantile"])
    valid = panel[has_targets]
    cut = valid.groupby("date")[["assets", "book", "sale_ac"]].quantile(q)
    cut.columns = ["cut_assets", "cut_book", "cut_sale"]
    panel = panel.merge(cut, left_on="date", right_index=True, how="left")
    above = (
        (panel["assets"] >= panel["cut_assets"])
        & (panel["book"] >= panel["cut_book"])
        & (panel["sale_ac"] >= panel["cut_sale"])
    )
    panel["in_sample"] = has_targets & above

    n_months = panel["date"].nunique()
    steps = [
        ("US common stocks on NYSE/AMEX/NASDAQ", panel, "universe"),
        ("Positive, non-missing targets", panel[has_targets], "targets"),
        (
            "Assets, book equity and sales above the monthly 10th percentile",
            panel[panel["in_sample"]],
            "sample",
        ),
    ]
    table1 = pd.DataFrame(
        [
            {
                "step": label,
                "firm_months": len(d),
                "firms": d["id"].nunique(),
                "per_month": round(len(d) / max(n_months, 1), 1),
                "paper_firm_months": PAPER_TABLE1[key][0],
                "paper_firms": PAPER_TABLE1[key][1],
                "paper_per_month": round(PAPER_TABLE1[key][0] / 480, 1),
            }
            for label, d, key in steps
        ]
    )

    extra_betas = [f"CAPMbeta_{m}" for m in min_obs]
    acc_cols = [ACC_PREFIX + c for c in ALL_ITEMS]
    keep = [*ID_COLUMNS, *FEATURES, *extra_betas, *LOG_TARGETS, *VALUE_COLUMNS, *acc_cols, "in_sample"]
    out = panel.loc[has_targets, keep].sort_values(["date", "id"]).reset_index(drop=True)
    out = out.replace([np.inf, -np.inf], np.nan)

    controls = {
        "table1": table1,
        "variables": summarise_variables(out[out["in_sample"]]),
        "targets": summarise_targets(out[out["in_sample"]]),
        "by_year": (
            out[out["in_sample"]]
            .assign(year=lambda d: d["date"].dt.year)
            .groupby("year")
            .agg(
                firm_months=("id", "size"),
                firms=("id", "nunique"),
                annual_source=("fuente", lambda s: (s == "annual").mean()),
            )
            .reset_index()
        ),
        "notes": pd.DataFrame(
            [
                {"item": "monthly returns above the guard set to missing", "value": n_extreme},
                {
                    "item": "share of sample firm-months from annual statements",
                    "value": round(float((out.loc[out["in_sample"], "fuente"] == "annual").mean()), 4),
                },
                {
                    "item": "share of sample firm-months with an industry",
                    "value": round(float(out.loc[out["in_sample"], "industry"].notna().mean()), 4),
                },
            ]
        ),
    }
    return out, controls


def summarise_variables(sample: pd.DataFrame, paper_file: Path | None = None) -> pd.DataFrame:
    """Coverage, mean and median of each input on the modelling sample (paper: Table IA.4)."""
    rows = []
    for var in FEATURES:
        s = sample[var]
        rows.append(
            {
                "variable": var,
                "category": CATEGORY_OF[var],
                "coverage": float(s.notna().mean()),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "p1": float(s.quantile(0.01)),
                "p99": float(s.quantile(0.99)),
            }
        )
    out = pd.DataFrame(rows)
    if paper_file is not None and Path(paper_file).exists():
        paper = pd.read_csv(paper_file)[["variable", "coverage"]].rename(
            columns={"coverage": "paper_coverage"}
        )
        out = out.merge(paper, on="variable", how="left")
        out["coverage_diff"] = out["coverage"] - out["paper_coverage"]
    return out


PAPER_TARGET_STATS = {"lnm2b": (0.53, 0.48), "lnv2a": (-0.02, -0.03), "lnv2s": (0.38, 0.35)}  # mean, median


def summarise_targets(sample: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": t,
                "n": int(sample[t].notna().sum()),
                "mean": float(sample[t].mean()),
                "median": float(sample[t].median()),
                "sd": float(sample[t].std()),
                "paper_mean": PAPER_TARGET_STATS[t][0],
                "paper_median": PAPER_TARGET_STATS[t][1],
            }
            for t in LOG_TARGETS
        ]
    )


__all__ = ["build_panel", "summarise_variables", "summarise_targets", "ACCOUNTING_FEATURES", "ACC_PREFIX"]
