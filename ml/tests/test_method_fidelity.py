"""Method-fidelity tests (section 10 of the master prompt). They use synthetic data only, so they
run in CI without the licensed files and must pass exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from relval.evaluation import portfolios
from relval.evaluation.metrics import compute_metrics, r2_oos
from relval.explain import shap_values as sv
from relval.explain.peer_weights import gbm_peer_weights, peer_weights_from_leaves
from relval.features.registry import CATEGORIES, DUPLICATE_PAIRS, FEATURES
from relval.models.benchmarks.bl import bl_features
from relval.models.benchmarks.ind import harmonic_mean_by_industry
from relval.models.gbm import fit_gbm
from relval.splits.blocks import assign_blocks, roles
from relval.targets.multiples import to_equity_value, valuation_error

GBM_CFG = {
    "params": {"objective": "regression", "learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 20},
    "num_boost_round": 10000, "early_stopping_rounds": 50, "deterministic": True,
}  # fmt: skip


# ---- registry ---------------------------------------------------------------------------------
def test_registry_has_the_papers_97_variables_in_11_categories():
    assert len(FEATURES) == 97 and len(set(FEATURES)) == 97
    counts = {k: len(v) for k, v in CATEGORIES.items()}
    assert counts == {"Profitability": 22, "Growth": 17, "Financial soundness": 16, "Efficiency": 15, "Liquidity": 8,
                      "Solvency": 8, "Capitalization": 4, "Size": 4, "Industry": 1, "Payout": 1, "CAPMbeta": 1}  # fmt: skip
    assert sum(f.endswith("_wr") for f in FEATURES) == 52 and sum(f.endswith("_an") for f in FEATURES) == 39
    assert all(a in FEATURES and b in FEATURES for a, b in DUPLICATE_PAIRS)


# ---- 3. splits --------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [5, 13, 510, 1915])
def test_blocks_are_balanced_disjoint_and_reproducible(n):
    blocks = assign_blocks(n, 201912, 20230301)
    sizes = np.bincount(blocks, minlength=5)
    assert sizes.sum() == n and sizes.max() - sizes.min() <= 1
    assert np.array_equal(blocks, assign_blocks(n, 201912, 20230301))
    assert n < 30 or not np.array_equal(blocks, assign_blocks(n, 201911, 20230301))
    tested = np.zeros(n, dtype=int)
    for k in range(5):
        train, val, test = roles(blocks, k)
        assert not (train & val).any() and not (train & test).any() and not (val & test).any()
        assert (train | val | test).all()
        tested += test
    assert (tested == 1).all()  # every observation is out-of-sample exactly once


# ---- 5. conversion ------------------------------------------------------------------------------
def test_conversion_worked_examples():
    m_hat = to_equity_value(0.95, "lnm2b", B=2000, A=np.nan, S=np.nan, D=0)
    assert m_hat == pytest.approx(5171, abs=1)
    assert valuation_error(m_hat, 6000) == pytest.approx(-0.138, abs=5e-4)
    # enterprise multiples subtract book debt: S=3000, D=1000, M=4000, yhat=0.40
    m_hat = to_equity_value(0.40, "lnv2s", B=np.nan, A=np.nan, S=3000, D=1000)
    assert m_hat == pytest.approx(3475, abs=1)
    assert valuation_error(m_hat, 4000) == pytest.approx(-0.131, abs=5e-4)
    assert to_equity_value(0.10, "lnv2a", B=np.nan, A=500, S=np.nan, D=800) < 0  # kept, counted and reported


def test_m2b_error_is_exp_of_log_difference():
    rng = np.random.default_rng(0)
    y, yhat, b = rng.normal(0.5, 0.8, 50), rng.normal(0.5, 0.8, 50), rng.uniform(10, 1e4, 50)
    eps = valuation_error(to_equity_value(yhat, "lnm2b", b, None, None, None), np.exp(y) * b)
    assert np.allclose(eps, np.exp(yhat - y) - 1)


# ---- 6-7. GBM toy example and peer weights (paper, Table A.2 and appendix A.3) -------------------
TOY_V = np.array([3.6, 3.4, 2.6, 2.2, 1.8])  # firms A-E
TOY_TRAIN = np.array([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0]])  # tree 1: {A,C,D,E},{B}; tree 2: {A,C},{B,D,E}
TOY_TEST = np.array([[0], [1]])  # firm X follows A's leaves


def test_toy_gbm_predictions_and_peer_weights():
    assert TOY_V.mean() == pytest.approx(2.72)
    k1 = peer_weights_from_leaves(TOY_TRAIN[:1], TOY_TEST[:1], 0.1)
    assert (TOY_V @ k1)[0] == pytest.approx(2.703, abs=5e-4)
    k2 = peer_weights_from_leaves(TOY_TRAIN, TOY_TEST, 0.1)
    assert np.allclose(k2[:, 0], [0.235, 0.162, 0.235, 0.185, 0.185], atol=5e-4)
    assert k2.sum() == pytest.approx(1.0)
    assert (TOY_V @ k2)[0] == pytest.approx(2.743, abs=5e-4)
    # in-sample: predicting the training firms reproduces panel D of Table A.2
    g2 = TOY_V @ peer_weights_from_leaves(TOY_TRAIN, TOY_TRAIN, 0.1)
    assert np.allclose(g2, [2.743, 2.762, 2.743, 2.677, 2.677], atol=1e-3)


def _synthetic(n=900, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    industry = rng.integers(1, 30, size=n).astype(float)
    X[rng.random(X.shape) < 0.1] = np.nan
    y = (
        0.8 * np.nan_to_num(X[:, 0])
        + np.sin(np.nan_to_num(X[:, 1]))
        + 0.05 * industry
        + rng.normal(scale=0.3, size=n)
    )
    return np.column_stack([X, industry]), y, [*[f"x{i}" for i in range(6)], "industry"]


@pytest.mark.parametrize("categorical", [[], ["industry"]])
def test_peer_weights_reproduce_lightgbm_predictions(categorical):
    X, y, names = _synthetic()
    tr, va, te = slice(0, 540), slice(540, 720), slice(720, 900)
    fit = fit_gbm(X[tr], y[tr], X[va], y[va], names, GBM_CFG, seed=7, categorical=categorical)
    pw = gbm_peer_weights(fit, X[tr], y[tr], X[te], learning_rate=0.1)
    assert pw.weights.shape == (540, 180)
    assert pw.max_sum_error < 1e-9  # the weights of every test firm sum to one
    assert pw.max_abs_error < 1e-5  # v'K equals the LightGBM prediction


# ---- 8. SHAP ------------------------------------------------------------------------------------
def test_shap_additivity_and_scaling():
    X, y, names = _synthetic()
    fit = fit_gbm(X[:540], y[:540], X[540:720], y[540:720], names, GBM_CFG, seed=7, categorical=["industry"])
    contrib = fit.shap(X[720:])
    assert np.allclose(contrib.sum(axis=1), fit.predict(X[720:]), atol=1e-6)  # raw SHAP + base = prediction
    raw = pd.DataFrame(contrib, columns=[*names, "base_value"])
    scaled = sv.scaled_abs_shap(raw, names)
    assert np.allclose(scaled.sum(axis=1), 100.0)


def test_category_importance_is_the_sum_of_its_variables():
    rng = np.random.default_rng(3)
    raw = pd.DataFrame(rng.normal(size=(200, 97)), columns=FEATURES)
    dates = pd.Series(pd.to_datetime(rng.choice(["1995-06-30", "2005-06-30", "2015-06-30"], 200)))
    imp = sv.variable_importance(sv.scaled_abs_shap(raw), dates)
    cat = sv.category_importance(imp)
    assert cat["shap"].sum() == pytest.approx(100.0)
    for name, members in CATEGORIES.items():
        assert cat.set_index("category").loc[name, "shap"] == pytest.approx(
            imp[imp["variable"].isin(members)]["shap"].sum()
        )
    assert {"d1990", "d2000", "d2010", "chg"} <= set(imp.columns)


# ---- 9. metrics ---------------------------------------------------------------------------------
def test_metrics_against_hand_calculation():
    pred = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0], "yhat": [1.5, 2.0, 2.0, 5.0], "M": [100.0, 200.0, 400.0, 800.0],
         "Mhat": [150.0, 200.0, 300.0, 1000.0], "train_mean_y": [2.0, 2.0, 3.0, 3.0], "train_mean_M": [300.0] * 4}
    )  # fmt: skip
    pred["eps"] = pred["Mhat"] / pred["M"] - 1  # 0.5, 0, -0.25, 0.25
    m = compute_metrics(pred)
    assert m["N"] == 4
    assert m["med_abs_eps"] == pytest.approx(0.25) and m["avg_abs_eps"] == pytest.approx(0.25)
    assert m["med_eps"] == pytest.approx(0.125) and m["avg_eps"] == pytest.approx(0.125)
    assert m["rmse_y"] == pytest.approx(np.sqrt((0.25 + 0 + 1 + 1) / 4))
    assert m["rmse_M"] == pytest.approx(np.sqrt((2500 + 0 + 10000 + 40000) / 4))
    # R2oos uses each observation's own training mean: denominators (1-2)^2+(2-2)^2+(3-3)^2+(4-3)^2 = 2
    assert m["r2oos_y"] == pytest.approx(1 - 2.25 / 2)
    assert r2_oos([1, 2], [1, 2], [0, 0]) == pytest.approx(1.0)


# ---- 4. no leakage in benchmark inputs ------------------------------------------------------------
def test_industry_statistics_use_the_fitting_sample_only():
    n = 60
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {"industry": np.repeat([1.0, 2.0], n // 2), "lnm2b": rng.normal(size=n), "lnv2a": rng.normal(size=n),
         "lnv2s": rng.normal(size=n), "opmad_wr": rng.normal(size=n), "salesgrowth_an": rng.normal(size=n),
         "acc_dltt": 1.0, "acc_dlc": 1.0, "book": 10.0, "pretret_noa_wr": 0.1, "roe_wr": 0.1, "rd_sale_wr": 0.0}
    )  # fmt: skip
    fit = np.arange(n) % 5 != 0
    base = bl_features(df, fit, 5)
    changed = df.copy()
    changed.loc[~fit, ["lnm2b", "lnv2a", "lnv2s"]] += 10.0  # move the test firms' multiples
    assert np.allclose(
        bl_features(changed, fit, 5)[["ind_hm_m2b", "ind_hm_v2a", "ind_hm_v2s"]],
        base[["ind_hm_m2b", "ind_hm_v2a", "ind_hm_v2s"]],
    )
    hm = harmonic_mean_by_industry(pd.Series([1.0, 2.0, 4.0, 4.0, 4.0, 4.0]), pd.Series([1] * 6), 5)
    assert hm.loc[1] == pytest.approx(6 / (1 + 0.5 + 4 * 0.25))
    assert (
        harmonic_mean_by_industry(pd.Series([1.0, 2.0]), pd.Series([1, 1]), 5).isna().all()
    )  # below the minimum


def test_shuffled_target_has_no_out_of_sample_fit():
    X, y, names = _synthetic(1500, seed=11)
    y = np.random.default_rng(0).permutation(y)  # break the link between inputs and target
    fit = fit_gbm(X[:900], y[:900], X[900:1200], y[900:1200], names, GBM_CFG, seed=1)
    assert r2_oos(y[1200:], fit.predict(X[1200:]), np.full(300, y[:900].mean())) < 0.05


# ---- 10. portfolios -------------------------------------------------------------------------------
def test_breakpoints_come_from_nyse_firms_only():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0, 300.0])
    nyse = np.array([True] * 5 + [False] * 3)
    breaks = portfolios.nyse_breakpoints(values, nyse, np.array([0.2, 0.4, 0.6, 0.8]))
    assert np.allclose(breaks, np.quantile(values[:5], [0.2, 0.4, 0.6, 0.8]))
    assert (
        portfolios.assign_quantiles(values, breaks)[5:] == 5
    ).all()  # non-NYSE extremes do not move the cuts


def test_value_weights_use_month_t_equity_value_and_month_t_plus_1_returns():
    date = pd.Timestamp("2019-11-30")
    data = pd.DataFrame(
        {"id": list("abcdefghij"), "date": date, "eps": np.linspace(-0.9, 0.9, 10), "nyse": True,
         "M": [1.0, 3.0, 1, 1, 1, 1, 1, 1, 1.0, 9.0], "ret_next": [0.10, 0.02, 0, 0, 0, 0, 0, 0, 0.00, 0.10]}
    )  # fmt: skip
    out = portfolios.sorted_portfolio_returns(data)
    assert out.index[0] == pd.Timestamp("2019-12-31")  # held during month t+1
    assert out["q1"].iloc[0] == pytest.approx((1 * 0.10 + 3 * 0.02) / 4)
    assert out["q5"].iloc[0] == pytest.approx((1 * 0.00 + 9 * 0.10) / 10)
    assert out["ls"].iloc[0] == pytest.approx(out["q5"].iloc[0] - out["q1"].iloc[0])


def test_newey_west_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(2)
    dates = pd.date_range("1990-02-28", periods=120, freq="ME")
    fac = pd.DataFrame(rng.normal(scale=0.03, size=(120, 6)), columns=portfolios.FF6).assign(date=dates)
    ls = pd.Series(0.004 + 0.3 * fac["hml"].to_numpy() + rng.normal(scale=0.02, size=120), index=dates)
    perf = portfolios.performance(ls, fac, lags=3)
    ref = sm.OLS(ls.to_numpy() * 100, sm.add_constant(fac[portfolios.FF6].to_numpy() * 100)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    assert perf["alpha"] == pytest.approx(ref.params[0]) and perf["alpha_t"] == pytest.approx(ref.tvalues[0])
    assert perf["hml"] == pytest.approx(ref.params[3])
    assert perf["sharpe"] == pytest.approx(ls.mean() / ls.std() * np.sqrt(12))
