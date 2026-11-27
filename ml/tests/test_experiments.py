"""Tests of the extensions (E1 robust loss, E2 training window) and of the experiment tooling.
Synthetic data only, like test_method_fidelity.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from relval.config import load_config
from relval.experiments import compare, registry
from relval.features.registry import FEATURES
from relval.models import window
from relval.models.gbm import fit_gbm
from relval.models.runner import _history_training_rows, fit_gbm_block, run_month
from relval.models.specs import SPECS
from relval.splits.blocks import add_blocks, firm_blocks

ROOT = Path(__file__).resolve().parents[2]
GBM_CFG = {
    "params": {"objective": "regression", "learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 20},
    "num_boost_round": 300, "early_stopping_rounds": 20, "deterministic": True, "min_train_rows": 40,
    "industry_categorical": True,
}  # fmt: skip


def _cfg(**window_kw) -> dict:
    return {"splits": {"n_blocks": 5}, "gbm": GBM_CFG, "benchmarks": {"ind_min_firms": 5},
            "training_window": {"months": 12, **window_kw}}  # fmt: skip


def _panel(n_firms: int = 300, months: int = 4, seed: int = 0) -> pd.DataFrame:
    """Firm-months whose lnm2b encodes the firm (firm k -> k / 1000) so leaked rows are traceable."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-31", periods=months, freq="ME")
    rows = []
    for d in dates:
        frame = pd.DataFrame(rng.normal(size=(n_firms, len(FEATURES))), columns=FEATURES)
        frame["industry"] = rng.integers(1, 10, n_firms).astype(float)
        frame["id"] = [f"F{k:04d}" for k in range(n_firms)]
        frame["date"] = d
        frame["origen"] = "final"
        frame["lnm2b"] = np.arange(n_firms) / 1000.0
        frame["lnv2a"] = frame["lnm2b"]
        frame["lnv2s"] = frame["lnm2b"]
        frame["M"], frame["B"], frame["A"], frame["S"], frame["D"] = 100.0, 50.0, 80.0, 60.0, 10.0
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


# ---- E2: blocks ------------------------------------------------------------------------------
def test_firm_blocks_are_fixed_per_firm_and_balanced():
    ids = [f"F{k:04d}" for k in range(1000)]
    a, b = firm_blocks(ids, 7), firm_blocks(list(reversed(ids)), 7)
    assert a.equals(b)  # depends on the set of ids, not their order
    assert a.value_counts().max() - a.value_counts().min() <= 1
    month = pd.DataFrame({"id": ids[:300], "date": pd.Timestamp("2019-01-31")})
    assert (add_blocks(month, 1, by_firm=a)["block"].to_numpy() == a.loc[sorted(ids[:300])].to_numpy()).all()


def test_window_defaults_reproduce_the_paper():
    w = window.window_cfg({})
    assert w["months"] == 1 and not window.uses_window({}) and not window.uses_firm_blocks({})
    with pytest.raises(ValueError):
        window.window_cfg({"training_window": {"leakage_guard": "typo"}})


# ---- E2: leakage guard -----------------------------------------------------------------------
@pytest.mark.parametrize("guard", ["exclude_test_firms", "firm_blocks"])
def test_window_never_trains_on_the_test_or_validation_firms(guard):
    panel = _panel()
    cfg = _cfg(leakage_guard=guard)
    last = panel["date"].max()
    fmap = firm_blocks(panel["id"], 3) if guard == "firm_blocks" else None
    df = add_blocks(panel[panel["date"] == last], 3, by_firm=fmap)
    hist = panel[panel["date"] < last]
    spec = SPECS["ML_lnm2b"]
    for k in range(5):
        Xh, yh, wh = _history_training_rows(
            spec, df, k, cfg, {"history": hist, "firm_blocks": fmap}, np.zeros(1)
        )
        banned = df.loc[df["block"].isin({k, (k + 1) % 5}), "lnm2b"].to_numpy()
        assert not np.isin(np.round(yh, 6), np.round(banned, 6)).any()
        assert len(yh) > 0 and wh is None and Xh.shape == (len(yh), len(FEATURES))


def test_no_guard_keeps_the_leak_and_is_only_a_diagnostic():
    panel = _panel()
    last = panel["date"].max()
    df = add_blocks(panel[panel["date"] == last], 3)
    hist = panel[panel["date"] < last]
    _, yh, _ = _history_training_rows(
        SPECS["ML_lnm2b"], df, 0, _cfg(leakage_guard="none"), {"history": hist}, np.zeros(1)
    )
    assert len(yh) == len(hist)


def test_window_adds_rows_and_predicts_every_test_firm_once():
    panel = _panel(months=3)
    last = panel["date"].max()
    hist = panel[panel["date"] < last]
    cfg = _cfg()
    month = panel[panel["date"] == last]
    with_hist = run_month(SPECS["ML_lnm2b"], month, cfg, 11, {"history": hist})
    without = run_month(SPECS["ML_lnm2b"], month, cfg, 11, {})
    assert len(with_hist.pred) == len(without.pred) == len(month)
    assert with_hist.pred["id"].is_unique
    assert (with_hist.pred["n_train"] > without.pred["n_train"]).all()


def test_no_history_gives_exactly_the_paper_model():
    panel = _panel(months=1)
    df = add_blocks(panel, 5)
    y = df["lnm2b"] + np.random.default_rng(1).normal(scale=0.1, size=len(df))
    a = fit_gbm_block(SPECS["ML_lnm2b"], df, y, 0, _cfg(), 5, {})
    b = fit_gbm_block(SPECS["ML_lnm2b"], df, y, 0, _cfg(), 5, {"history": None})
    assert a.n_history == b.n_history == 0
    assert np.array_equal(a.model.predict(a.X[a.test]), b.model.predict(b.X[b.test]))


def test_lag_weights_and_level_alignment():
    date = pd.Timestamp("2019-12-31")
    d = pd.to_datetime(["2019-11-30", "2019-11-30", "2019-01-31"])
    assert np.allclose(window.lag_weights(d, date, 0.9), [0.9, 0.9, 0.9**11])
    y = window.align_levels(np.array([1.0, 3.0, 10.0]), d, reference_median=0.0)
    assert np.allclose(y, [-1.0, 1.0, 0.0])


# ---- E1: loss --------------------------------------------------------------------------------
@pytest.mark.parametrize("objective", ["regression_l1", "huber"])
def test_objective_from_the_config_reaches_lightgbm_and_resists_outliers(objective):
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(size=(n, 3))
    y = X[:, 0] + rng.normal(scale=0.2, size=n)
    bad = rng.random(n) < 0.05
    y[bad & (np.arange(n) < 1000)] += 8.0  # outliers in the training rows only
    cfg_l2 = dict(GBM_CFG)
    cfg_rob = {**GBM_CFG, "params": {**GBM_CFG["params"], "objective": objective, "alpha": 0.5}}
    names = ["a", "b", "c"]
    l2 = fit_gbm(X[:800], y[:800], X[800:1000], y[800:1000], names, cfg_l2, seed=1)
    rob = fit_gbm(X[:800], y[:800], X[800:1000], y[800:1000], names, cfg_rob, seed=1)
    assert rob.booster.params["objective"] == objective
    truth = X[1000:, 0]
    assert np.median(np.abs(rob.predict(X[1000:]) - truth)) < np.median(np.abs(l2.predict(X[1000:]) - truth))


def test_row_weights_change_the_fit():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(600, 2)), rng.normal(size=600)
    w = np.where(np.arange(400) < 200, 1.0, 0.01)
    a = fit_gbm(X[:400], y[:400], X[400:500], y[400:500], ["a", "b"], GBM_CFG, seed=1)
    b = fit_gbm(X[:400], y[:400], X[400:500], y[400:500], ["a", "b"], GBM_CFG, seed=1, w_train=w)
    assert not np.allclose(a.predict(X[500:]), b.predict(X[500:]))


# ---- experiment configs and comparison ---------------------------------------------------------
def test_every_experiment_config_is_valid():
    exps = registry.discover(ROOT)
    ids = {e.id for e in exps}
    assert {"e1_l1", "e1_huber", "e2_w12", "e2_w12_firmblocks", "e2_firmblocks_w1", "e2_w12_leaky"} <= ids
    for e in exps:
        assert e.cfg.run_name == f"experiments/{e.id}"
        assert set(e.models) <= set(SPECS)
        registry.baseline_config(e)  # the baseline config exists
    by_id = {e.id: e for e in exps}
    assert by_id["e1_l1"].cfg.get("gbm.params.objective") == "regression_l1"
    assert window.window_cfg(by_id["e2_w12"].cfg.data)["months"] == 12
    assert by_id["e2_w12_firmblocks"].baseline == "experiments/e2_firmblocks_w1"
    # an experiment changes only what it declares: the rest is the replication's
    base = load_config("replication.yaml", root=ROOT)
    assert by_id["e1_l1"].cfg["panel"] == base["panel"]
    assert by_id["e1_l1"].cfg.get("gbm.params.learning_rate") == base.get("gbm.params.learning_rate")


def test_comparison_is_paired_on_the_same_firm_months():
    dates = pd.to_datetime(["2019-01-31"] * 3 + ["2019-02-28"] * 3)
    base = pd.DataFrame({"id": list("abcabc"), "date": dates, "eps": [0.1, -0.2, 0.3, 0.2, 0.1, -0.1]})
    exp = base.iloc[[5, 4, 0, 1, 2]].assign(eps=lambda d: d["eps"] / 2)  # one row missing, shuffled
    b, e = compare.paired(base, exp)
    assert len(b) == len(e) == 5
    assert (b[["id", "date"]].to_numpy() == e[["id", "date"]].to_numpy()).all()
    assert np.allclose(e["eps"], b["eps"] / 2)
    lo, hi = compare.bootstrap_mean_ci(np.full(50, -0.01))
    assert lo == pytest.approx(-0.01) and hi == pytest.approx(-0.01)
