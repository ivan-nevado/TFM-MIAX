"""Formula and timing tests on a small synthetic firm: four-quarter sums, one-year lags, the
missing-value rules of Table A.1, the three-month availability lag and the beta window."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from relval.data.schemas import ALL_ITEMS, BALANCE_ITEMS, FLOW_ITEMS
from relval.features import accounting
from relval.features.accounting import compute_accounting_features
from relval.features.beta import rolling_beta
from relval.features.registry import ACCOUNTING_FEATURES
from relval.panel.book_equity import div, rowtotal, sas_sum
from relval.panel.build import _link, month_index
from relval.panel.records import build_records, period_month

DEFS = {"zero_fill_items": [], "dp_adjustment": "none", "efftax_positive_pi_only": True}


def synthetic_fund(n_quarters: int = 9) -> pd.DataFrame:
    """One firm, quarter q = 1..n: every item is a known function of q."""
    dates = pd.date_range("2017-03-31", periods=n_quarters, freq="QE")
    q = np.arange(1, n_quarters + 1, dtype=float)
    data = {"id": "X", "datadate": dates, "origen": "final"}
    base = {"at": 1000, "act": 400, "lct": 200, "lt": 600, "che": 100, "invt": 80, "rect": 120, "ap": 60, "ppent": 300,
            "dltt": 250, "dlc": 50, "pstk": 0, "ceq": 400, "txditc": 10, "mib": 5,
            "sale": 200, "cogs": 120, "xsga": 40, "xrd": 8, "xint": 5, "ebit": 40, "dp": 10, "pi": 35, "txt": 9, "ib": 26,
            "prefdiv": 0, "ibadj": 26, "ocf": 30, "capx": 12, "dv": 4, "mii": 1, "nopi": 2, "spi": -1, "oibdp": 50}  # fmt: skip
    for item in ALL_ITEMS:
        data[item] = base[item] * (1 + 0.02 * q)  # grows 2% of the base per quarter
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def records() -> pd.DataFrame:
    return build_records(synthetic_fund(), "Q", DEFS)


def test_period_month_handles_52_53_week_calendars():
    d = pd.Series(pd.to_datetime(["2026-09-30", "2026-09-25", "2026-10-04", "2026-06-26"]))
    pm = period_month(d)
    assert pm[0] == pm[1] == pm[2] == 2026 * 12 + 8 and pm[3] == 2026 * 12 + 5


def test_flows_are_four_quarter_sums_and_lags_are_one_year(records):
    fund = synthetic_fund()
    last = records.iloc[-1]  # quarter 9
    assert last["sale"] == pytest.approx(fund["sale"].iloc[5:9].sum())
    assert last["l_sale"] == pytest.approx(fund["sale"].iloc[1:5].sum())
    assert last["at"] == pytest.approx(fund["at"].iloc[8]) and last["l_at"] == pytest.approx(
        fund["at"].iloc[4]
    )
    assert np.isnan(records.iloc[2]["sale"])  # fewer than four quarters: no annualised flow
    assert np.isnan(records.iloc[3]["l_at"])  # no record one year earlier
    assert set(BALANCE_ITEMS + FLOW_ITEMS) <= set(records.columns)


def test_a_missing_quarter_breaks_the_sum(records):
    fund = synthetic_fund().drop(index=6)  # quarter 7 not reported
    rec = build_records(fund, "Q", DEFS)
    assert np.isnan(rec.iloc[-1]["sale"])  # quarter 9 needs quarters 6-9


def test_selected_formulas_against_hand_values(records):
    r = records.iloc[[-1]].reset_index(drop=True)
    f = compute_accounting_features(r, pd.Series([False]), DEFS).iloc[0]
    x = r.iloc[0]
    be = x["ceq"] + x["pstk"] + x["txditc"] - x["pstk"]
    l_be = x["l_ceq"] + x["l_txditc"]
    ebitda = x["ebit"] + x["dp"]
    assert list(compute_accounting_features(r, pd.Series([False]), DEFS).columns) == ACCOUNTING_FEATURES
    expected = {
        "book": be, "assets": x["at"], "sale_ac": x["sale"], "debt_book_value": x["dltt"] + x["dlc"] + x["pstk"],
        "roe_wr": x["ib"] / be, "roa_wr": ebitda / x["at"], "npm_wr": x["ib"] / x["sale"],
        "gpm_wr": (x["sale"] - x["cogs"]) / x["sale"], "at_turn_wr": x["sale"] / x["at"],
        "opleverage_an": (x["cogs"] + x["xsga"]) / x["at"],
        "profitability_an": (x["sale"] - x["cogs"] - x["xint"] - x["xsga"]) / be,
        "roic_an": (x["ebit"] - x["nopi"]) / (x["ceq"] + x["lt"] - x["che"]),
        "salesgrowth_an": x["sale"] / x["l_sale"] - 1, "assetgrowth_an": (x["at"] - x["l_at"]) / x["l_at"],
        "chbe_an": (be - l_be) / x["l_at"], "investment_an": (x["at"] - x["l_at"]) / x["at"],
        "rd_sale_wr": x["xrd"] / x["sale"], "debt_at_wr": (x["dltt"] + x["dlc"]) / x["at"],
        "lt_ppent_wr": x["lt"] / x["ppent"], "sales2rec_an": x["sale"] / x["rect"],
        "quick_ratio_wr": (x["act"] - x["invt"]) / x["lct"], "cash2assets_an": x["che"] / x["at"],
        "de_ratio_wr": x["lt"] / (x["ceq"] + x["pstk"]), "dpr_wr": x["dv"] / x["ibadj"],
        "pretret_noa_wr": x["ebit"] / (x["ppent"] + x["act"] - x["lct"]),
        "pretret_earnat_wr": ebitda / (x["ppent"] + x["act"]),
        "liquid2assets_an": (x["che"] + 0.75 * (x["act"] - x["che"]) + 0.5 * (x["at"] - x["act"])) / x["l_at"],
        "debt2tang_an": (x["che"] + 0.715 * x["rect"] + 0.547 * x["invt"] + 0.535 * x["ppent"]) / x["at"],
        "aftret_invcapx_wr": (x["ib"] + x["xint"] + x["mii"]) / (x["l_dltt"] + x["l_pstk"] + x["l_ceq"] + x["l_mib"] + x["l_txditc"] - x["l_mib"]),
        "efftax_wr": x["txt"] / x["pi"], "intcov_wr": (x["xint"] + x["ib"]) / x["xint"],
        "Accrual_wr": (x["ocf"] - x["ib"]) / x["at"],
    }  # fmt: skip
    for name, value in expected.items():
        assert f[name] == pytest.approx(value, rel=1e-12), name
    for a, b in [
        ("gpm_wr", "grossmargin_an"),
        ("gprof_wr", "grossprofit_an"),
        ("opmbd_wr", "ebitda2revenue_an"),
        ("at_turn_wr", "assetturnover_an"),
    ]:
        assert f[a] == f[b]  # the four exact duplicate pairs


def test_missing_value_rules(records):
    r = records.iloc[[-1]].reset_index(drop=True).copy()
    r.loc[0, ["xrd", "lct"]] = [np.nan, 0.0]
    r.loc[0, "ib"] = -5.0
    r.loc[0, "pi"] = -2.0
    f = compute_accounting_features(r, pd.Series([True]), DEFS).iloc[0]
    assert f["rd_sale_wr"] == 0.0  # missing R&D is zero
    assert np.isnan(f["curr_ratio_wr"]) and np.isnan(
        f["quick_ratio_wr"]
    )  # zero denominator / lct > 0 not met
    assert np.isnan(f["cash2assets_an"])  # financial firm
    assert np.isnan(f["efftax_wr"])  # pre-tax income not positive
    assert f["roe_wr"] < 0  # be > 0 holds, negative income is kept
    assert not np.isinf(f.to_numpy(dtype=float)).any()
    assert (
        sas_sum(pd.Series([np.nan]), pd.Series([2.0])).iloc[0] == 2.0
        and sas_sum(pd.Series([np.nan]), pd.Series([np.nan])).isna().all()
    )
    assert rowtotal(pd.Series([np.nan]), pd.Series([np.nan])).iloc[0] == 0.0
    assert div(pd.Series([1.0]), pd.Series([0.0])).isna().all()


def test_no_feature_formula_reads_market_data():
    source = inspect.getsource(accounting)
    for token in ('"mv"', '"M"', '"ret"', "price", '"V"'):
        assert token not in source, f"accounting features must not use {token}"


def test_accounting_data_are_used_three_months_after_the_balance_sheet_date():
    rec = pd.DataFrame(
        {"id": "X", "datadate": pd.to_datetime(["2018-12-31", "2019-03-31"]), "value": [1.0, 2.0]}
    )
    rec["pm"] = period_month(rec["datadate"])
    rec["avail"] = rec["pm"] + 3
    months = pd.DataFrame({"id": "X", "date": pd.date_range("2019-03-31", "2020-05-31", freq="ME")})
    months["mi"] = month_index(months["date"])
    linked = _link(months, rec, max_age=12).assign(date=months["date"])
    by_date = linked.set_index("date")["value"]
    assert by_date["2019-03-31"] == 1.0 and by_date["2019-05-31"] == 1.0  # March quarter not yet public
    assert by_date["2019-06-30"] == 2.0  # first used in the June cross-section
    assert by_date["2020-03-31"] == 2.0 and np.isnan(
        by_date["2020-04-30"]
    )  # dropped when older than 12 months
    assert (month_index(months["date"]) - linked["pm"]).dropna().min() >= 3


def test_beta_uses_only_returns_before_the_valuation_month():
    rng = np.random.default_rng(4)
    x = rng.normal(scale=0.04, size=130)
    y = 1.3 * x + rng.normal(scale=0.01, size=130)
    beta = rolling_beta(y, x, window=60, min_obs=60)
    assert np.isnan(beta[:60]).all() and not np.isnan(beta[60])
    assert beta[100] == pytest.approx(np.polyfit(x[40:100], y[40:100], 1)[0])  # months t-60 .. t-1
    y2 = y.copy()
    y2[100] = 5.0  # changing the month-t return must not change beta at t
    assert rolling_beta(y2, x, 60, 60)[100] == beta[100]
    y3 = y.copy()
    y3[70] = np.nan
    assert np.isnan(rolling_beta(y3, x, 60, 60)[100]) and not np.isnan(rolling_beta(y3, x, 60, 48)[100])
