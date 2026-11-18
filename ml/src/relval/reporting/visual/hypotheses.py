"""Hypothesis tests for the defence of the replication.

Unit of observation: the month. Every comparison is paired (same firms, same month) and turned into
a monthly series, whose mean is tested with Newey-West standard errors (12 lags), because errors of
consecutive months are correlated. Tests of the same family are corrected with Holm. Equivalence
with the paper is tested with two one-sided tests (TOST) on a block bootstrap of months.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats import binomtest, spearmanr
from statsmodels.stats.multitest import multipletests

from relval.features.registry import CATEGORY_OF
from relval.reporting.visual.style import (
    FAMILY_COLOR,
    GRID,
    INK2,
    MUTED,
    SURFACE,
    TARGET_COLOR,
    TARGET_LABEL,
    TARGETS,
    family,
    suptitle,
)

SECTION = "hipotesis"
MULT = {"lnm2b": "m2b", "lnv2a": "v2a", "lnv2s": "v2s"}
PAPER_MED_ABS = {"lnm2b": 0.29, "lnv2a": 0.30, "lnv2s": 0.32}
TOLERANCE = 0.05
LAGS = 12


def nw_mean(x: pd.Series, lags: int = LAGS) -> dict:
    x = pd.Series(x).dropna()
    fit = sm.OLS(x.to_numpy(float), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    m, se = float(fit.params[0]), float(fit.bse[0])
    return {"mean": m, "se": se, "t": m / se if se > 0 else np.nan, "p_two": float(fit.pvalues[0]),
            "lo": m - 1.96 * se, "hi": m + 1.96 * se, "n": int(len(x))}  # fmt: skip


def paired_monthly(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Monthly difference median|eps_b| - median|eps_a| on the firms both models value (> 0 = a better)."""
    m = a[["id", "date", "eps"]].merge(b[["id", "date", "eps"]], on=["id", "date"], suffixes=("_a", "_b"))
    m["ea"], m["eb"] = m["eps_a"].abs(), m["eps_b"].abs()
    g = m.groupby("date")
    return (g["eb"].median() - g["ea"].median()).rename("diff"), len(m)


def _row(family_name, hypothesis, target, s, n_obs, alternative="greater", scale=1.0):
    r = nw_mean(s)
    p = r["p_two"] / 2 if alternative == "greater" and r["t"] > 0 else (
        1 - r["p_two"] / 2 if alternative == "greater" else r["p_two"])  # fmt: skip
    pos = int((s > 0).sum())
    sign_p = (
        binomtest(pos, int((s != 0).sum()), alternative="greater").pvalue
        if alternative == "greater"
        else np.nan
    )
    return {
        "familia": family_name, "hipótesis": hypothesis, "target": TARGET_LABEL.get(target, target),
        "efecto": r["mean"] * scale, "ic95_lo": r["lo"] * scale, "ic95_hi": r["hi"] * scale, "t": r["t"], "p": p,
        "meses": r["n"], "obs": n_obs, "% meses a favor": pos / r["n"] * 100, "p signo": sign_p,
    }  # fmt: skip


def _holm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_holm"] = np.nan
    for _fam, g in df.groupby("familia"):
        ok = g["p"].notna()
        if ok.any():
            df.loc[g.index[ok], "p_holm"] = multipletests(g.loc[ok, "p"], method="holm")[1]
    return df


def build(data, book, xai_facts: dict | None = None) -> dict:
    rows = []
    # ---- A: GBM against each benchmark ----------------------------------------------------
    for t in TARGETS:
        g = data.preds(f"ML_{t}", None, ["id", "date", "eps"])
        for bench in (f"IND_{MULT[t]}", f"BL_{MULT[t]}", "RRV", "BG"):
            b = data.preds(bench, None, ["id", "date", "eps"])
            if b.empty:
                continue
            s, n = paired_monthly(g, b)
            rows.append(_row("A. GBM mejor que el benchmark", f"GBM < {bench.split('_')[0]}", t, s, n))
        data.drop_cache()
    # ---- B: no bias ---------------------------------------------------------------------
    bias = []
    for t in TARGETS:
        g = data.preds(f"ML_{t}", None, ["date", "eps"])
        s = g.groupby("date")["eps"].median()
        r = nw_mean(s)
        bias.append({"familia": "B. GBM sin sesgo", "hipótesis": "mediana de ε = 0", "target": TARGET_LABEL[t],
                     "efecto": r["mean"], "ic95_lo": r["lo"], "ic95_hi": r["hi"], "t": r["t"], "p": r["p_two"],
                     "meses": r["n"], "obs": len(g)})  # fmt: skip
    rows += bias
    # ---- C: equivalence with the paper (TOST on a block bootstrap of months) -------------------
    equiv = []
    rng = np.random.default_rng(20230301)
    for t in TARGETS:
        g = data.preds(f"ML_{t}", None, ["date", "eps"])
        by_m = [np.abs(v.to_numpy()) for _, v in g.groupby("date")["eps"]]
        nm, block = len(by_m), 12
        stats = []
        for _ in range(1000):
            starts = rng.integers(0, nm, size=int(np.ceil(nm / block)))
            idx = np.concatenate([(np.arange(s, s + block) % nm) for s in starts])[:nm]
            stats.append(np.median(np.concatenate([by_m[i] for i in idx])))
        stats = np.array(stats)
        est = float(np.median(np.abs(g["eps"])))
        lo90, hi90 = np.percentile(stats, [5, 95])
        lo95, hi95 = np.percentile(stats, [2.5, 97.5])
        paper = PAPER_MED_ABS[t]
        # TOST p-value: largest of the two one-sided bootstrap p-values
        p_low = float(np.mean(stats <= paper - TOLERANCE))
        p_high = float(np.mean(stats >= paper + TOLERANCE))
        equiv.append({"familia": "C. Equivalencia con el paper", "hipótesis": f"|med|ε| − {paper}| < {TOLERANCE}",
                      "target": TARGET_LABEL[t], "efecto": est - paper, "ic95_lo": lo95 - paper,
                      "ic95_hi": hi95 - paper, "t": np.nan, "p": max(p_low, p_high), "meses": nm, "obs": len(g),
                      "ic90": (lo90, hi90)})  # fmt: skip
        data.drop_cache()
    rows += equiv
    # ---- D: reconciliation steps (Table 4) ----------------------------------------------------
    for t in TARGETS:
        m = MULT[t]
        steps = [(f"BL_{m}", f"OLSLOG_BL_{t}", "logaritmo mejora a niveles"),
                 (f"OLSLOG_BL_{t}", f"GBM_BL_{t}", "GBM mejora a OLS (10 variables)"),
                 (f"GBM_BL_{t}", f"GBM_ML_BLS_{t}", "97 variables mejoran a 10"),
                 (f"LASSO_NM_{t}", f"GBM_NM_{t}", "GBM mejora a LASSO")]  # fmt: skip
        for before, after, lab in steps:
            a, b = (
                data.preds(after, None, ["id", "date", "eps"]),
                data.preds(before, None, ["id", "date", "eps"]),
            )
            if a.empty or b.empty:
                continue
            s, n = paired_monthly(a, b)
            rows.append(_row("D. Reconciliación (Tabla 4)", lab, t, s, n))
        data.drop_cache()
    # ---- E: sensitivity (no difference expected) ------------------------------------------------
    for t in TARGETS:
        g = data.preds(f"ML_{t}", None, ["id", "date", "eps"])
        for suffix, lab in (("indnum", "sector numérico"), ("listed", "solo cotizadas hoy"), ("beta24", "beta 24 meses"),
                            ("beta36", "beta 36 meses"), ("beta48", "beta 48 meses")):  # fmt: skip
            v = data.preds(f"ML_{t}_{suffix}", None, ["id", "date", "eps"])
            if v.empty:
                continue
            s, n = paired_monthly(g, v)
            rows.append(_row("E. Sensibilidad (diferencia ≠ 0)", f"{lab} frente a base", t, s, n,
                             alternative="two-sided", scale=1000))  # fmt: skip
        data.drop_cache()
    # ---- G: seeds ---------------------------------------------------------------------------
    for t in TARGETS:
        seeds = data.seeds_available(f"ML_{t}")
        preds = {s: data.preds(f"ML_{t}", s, ["id", "date", "eps"]) for s in seeds}
        for s1, s2 in combinations(seeds, 2):
            s, n = paired_monthly(preds[s1], preds[s2])
            rows.append(_row("G. Semillas (diferencia ≠ 0)", f"semilla {str(s1)[-1]} frente a {str(s2)[-1]}", t, s, n,
                             alternative="two-sided", scale=1000))  # fmt: skip
        data.drop_cache()
    # ---- F: portfolios ------------------------------------------------------------------------
    for kind, name in (("vw", "ponderada por valor"), ("ew", "equiponderada")):
        st = data.table(f"portfolio_stats_{kind}")
        for t in TARGETS:
            r = st[st["model"] == f"ML_{t}"]
            if r.empty:
                continue
            r = r.iloc[0]
            for what, mean, tval, p in (("rentabilidad media > 0", r["mean"], r["mean_t"], r["mean_p"]),
                                        ("alfa FF6 > 0", r["alpha"], r["alpha_t"], r["alpha_p"])):  # fmt: skip
                se = mean / tval if tval else np.nan
                rows.append({"familia": f"F. Carteras long-short ({name})", "hipótesis": what, "target": TARGET_LABEL[t],
                             "efecto": mean, "ic95_lo": mean - 1.96 * se, "ic95_hi": mean + 1.96 * se, "t": tval,
                             "p": p / 2 if tval > 0 else 1 - p / 2, "meses": int(r["n_months"])})  # fmt: skip
    # ---- H, I: explainability -----------------------------------------------------------------
    xai_facts = xai_facts or {}
    for t, v in xai_facts.get("peers_same_sector", {}).items():
        rows.append({"familia": "H. Comparables del mismo sector", "hipótesis": "más que al azar (Wilcoxon)",
                     "target": TARGET_LABEL[t], "efecto": (v["obs"] - v["exp"]) * 100, "p": v["p"], "obs": v["n"]})  # fmt: skip
    t8 = data.table("table8_vs_paper")
    for t, sh in xai_facts.get("shares", {}).items():
        ours = sh.groupby(lambda v: CATEGORY_OF[v]).sum()
        paper = t8[t8["model"] == f"ML_{t}"].set_index("category")["paper_shap"]
        both = pd.concat([ours.rename("o"), paper.rename("p")], axis=1).dropna()
        r = spearmanr(both["o"], both["p"], alternative="greater")
        rows.append({"familia": "I. Ranking SHAP por categoría", "hipótesis": "coincide con el paper (Spearman)",
                     "target": TARGET_LABEL[t], "efecto": float(r.statistic), "p": float(r.pvalue), "obs": len(both)})  # fmt: skip

    res = _holm(pd.DataFrame(rows))
    alpha = 0.05
    pcol = res["p_holm"].fillna(res["p"])
    res["rechaza H0 (5 %)"] = np.where(pcol < alpha, "sí", "no")
    _forest(res, book)
    return {"tests": res, "equivalence": pd.DataFrame(equiv)}


def _forest(res: pd.DataFrame, book):
    a = res[res["familia"].str.startswith("A.")].copy()
    d = res[res["familia"].str.startswith("D.")].copy()
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.4))
    for axis, df, title in (
        (ax[0], a, "A. GBM frente a cada benchmark"),
        (ax[1], d, "D. Pasos de la Tabla 4"),
    ):
        df = df.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(df))
        colors = [TARGET_COLOR[[k for k, v in TARGET_LABEL.items() if v == tg][0]] for tg in df["target"]]
        axis.hlines(y, df["ic95_lo"], df["ic95_hi"], color=colors, linewidth=2)
        axis.scatter(df["efecto"], y, color=colors, s=36, zorder=3, edgecolor=SURFACE, linewidth=1)
        axis.axvline(0, color=INK2, linewidth=0.8)
        axis.set_yticks(
            y, [f"{h}  ·  {tg}" for h, tg in zip(df["hipótesis"], df["target"], strict=True)], fontsize=8.5
        )
        axis.set_xlabel("mejora mensual media de la mediana de |ε| (IC 95 % Newey-West)")
        axis.set_title(title)
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)
    suptitle(fig, "Contrastes pareados mes a mes: a la derecha de 0 = la hipótesis se cumple")
    fig.tight_layout()
    book.save(
        fig, "hip_01_forest", SECTION, "Contrastes pareados",
        "Cada línea es un contraste: la diferencia mensual de la mediana de |ε| entre dos modelos sobre las mismas "
        "empresas, con su intervalo de confianza del 95 % (errores de Newey-West con 12 retardos, porque los "
        "meses consecutivos están correlacionados). Si el intervalo queda entero a la derecha de 0, el primer "
        "modelo es significativamente mejor.",
    )  # fmt: skip
    _ = (FAMILY_COLOR, GRID, MUTED, family)
