"""Generates ``REPLICATION_REPORT.md``: every table next to the paper's values, the six qualitative
findings of section 7 of the master prompt with their numbers, and the working thresholds.

The report is written entirely from the tables produced by code; no number is typed by hand.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from relval.io import git_commit, library_versions, to_markdown, write_table
from relval.models.specs import BASE_MODELS, LOGS, MULTS

PAPER_MED_ABS = {"ML_lnm2b": 0.29, "ML_lnv2a": 0.30, "ML_lnv2s": 0.32}


def _get(t: dict, name: str) -> pd.DataFrame:
    return t.get(name, pd.DataFrame())


def _verdict(flag) -> str:
    if flag is None:
        return "not evaluated"
    return "holds" if flag else "does NOT hold"


def findings(t: dict[str, pd.DataFrame], cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The six qualitative findings and the four working thresholds, each with its evidence."""
    rows, thr = [], []
    th = cfg["thresholds"]
    t2 = _get(t, "table2").set_index("model") if len(_get(t, "table2")) else pd.DataFrame()

    # 1. GBM has the lowest med|eps| for each multiple and med(eps) close to zero
    if len(t2):
        ok, parts = True, []
        for tg, m in zip(LOGS, MULTS, strict=True):
            rivals = [x for x in (f"BL_{m}", f"IND_{m}", f"HP_{m}", "RRV", "BG") if x in t2.index]
            if f"ML_{tg}" not in t2.index or not rivals:
                continue
            g, best = t2.loc[f"ML_{tg}", "med_abs_eps"], t2.loc[rivals, "med_abs_eps"].min()
            ok &= bool(g < best)
            parts.append(
                f"{m}: GBM {g:.3f} vs best traditional {best:.3f} ({t2.loc[rivals, 'med_abs_eps'].idxmin()}), med(eps) {t2.loc[f'ML_{tg}', 'med_eps']:+.3f}"
            )
        rows.append(
            {
                "finding": "1. GBM has the lowest med(|eps|) for each multiple, med(eps) near zero",
                "verdict": _verdict(ok if parts else None),
                "evidence": "; ".join(parts),
            }
        )
        for model, paper in PAPER_MED_ABS.items():
            if model in t2.index:
                v = t2.loc[model, "med_abs_eps"]
                thr.append(
                    {
                        "threshold": f"med(|eps|) of {model} within +-{th['med_abs_eps_tolerance']} of the paper ({paper})",
                        "value": v,
                        "passes": abs(v - paper) <= th["med_abs_eps_tolerance"],
                    }
                )
                v = t2.loc[model, "med_eps"]
                thr.append(
                    {
                        "threshold": f"|med(eps)| of {model} below {th['med_eps_abs_max']}",
                        "value": v,
                        "passes": abs(v) < th["med_eps_abs_max"],
                    }
                )
                v = t2.loc[model, "r2oos_M"]
                thr.append(
                    {
                        "threshold": f"R2oos(M) of {model} at least {th['r2oos_M_min']}",
                        "value": v,
                        "passes": v >= th["r2oos_M_min"],
                    }
                )

    # 2. GBM below IND and BL in almost every month
    mb = _get(t, "months_gbm_below_benchmarks")
    if len(mb):
        core = mb[mb["benchmark"].str.startswith(("BL_", "IND_"))]
        ok = bool((core["share_below"] >= th["months_below_benchmarks_min"]).all()) if len(core) else None
        ev = "; ".join(
            f"{r.model} < {r.benchmark} in {r.share_below:.1%} of {r.months} months"
            for r in core.itertuples()
        )
        rows.append(
            {
                "finding": "2. GBM below IND and BL in med(|eps|) in almost every month",
                "verdict": _verdict(ok),
                "evidence": ev,
            }
        )
        for r in core.itertuples():
            thr.append(
                {
                    "threshold": f"{r.model} below {r.benchmark} in at least {th['months_below_benchmarks_min']:.0%} of months",
                    "value": r.share_below,
                    "passes": r.share_below >= th["months_below_benchmarks_min"],
                }
            )

    # 3. Table 4 ordering
    t4 = _get(t, "table4")
    if len(t4) and t4["N"].fillna(0).gt(0).any():
        ok, parts = True, []
        for m in MULTS:
            e = t4[t4["panel"] == m]["med_abs_eps"].to_numpy()
            if len(e) < 6 or np.isnan(e[:4]).any():
                continue
            logs, gbm, more = e[0] - e[1], e[1] - e[2], e[2] - e[3]
            lasso = e[4] - e[5] if not np.isnan(e[4:6]).any() else np.nan
            ok &= bool(logs > 0 and gbm > 0 and (np.isnan(lasso) or lasso > 0))
            parts.append(
                f"{m}: logs {logs:+.3f}, GBM {gbm:+.3f}, 97 variables {more:+.3f}, GBM vs LASSO {lasso:+.3f}"
            )
        rows.append(
            {
                "finding": "3. Table 4: logs help, the GBM helps more, extra variables matter mainly for v2s, GBM beats LASSO",
                "verdict": _verdict(ok if parts else None),
                "evidence": "reduction of med(|eps|) at each step - " + "; ".join(parts),
            }
        )

    # 4. SHAP
    t8, ia8 = _get(t, "table8"), _get(t, "ia8_variable_importance")
    if len(t8) and len(ia8):
        top_cat = (
            t8.sort_values("shap", ascending=False)
            .groupby("model")
            .head(2)
            .groupby("model")["category"]
            .apply(list)
        )
        top_var = (
            ia8.sort_values("shap", ascending=False)
            .groupby("model")
            .head(3)
            .groupby("model")["variable"]
            .apply(list)
        )
        v2s = ia8[ia8["model"] == "ML_lnv2s"].set_index("variable")
        eff_first = top_cat.get("ML_lnv2s", [""])[0] == "Efficiency"
        prof_growth = all(
            set(top_cat.get(m, [])) == {"Profitability", "Growth"} for m in ("ML_lnm2b", "ML_lnv2a")
        )
        neg = (
            bool(v2s.loc["opleverage_an", "rho"] < 0 and v2s.loc["at_turn_wr", "rho"] < 0)
            if "opleverage_an" in v2s.index
            else None
        )
        ev = "; ".join(
            f"{m}: top categories {top_cat.get(m)}, top variables {top_var.get(m)}"
            for m in BASE_MODELS
            if m in top_cat
        )
        if "opleverage_an" in v2s.index:
            ev += f"; v2s rho: opleverage_an {v2s.loc['opleverage_an', 'rho']:+.3f}, at_turn_wr {v2s.loc['at_turn_wr', 'rho']:+.3f}"
        rows.append(
            {
                "finding": "4. SHAP: Profitability and Growth lead m2b and v2a; Efficiency leads v2s with negative rho",
                "verdict": _verdict(bool(eff_first and prof_growth and neg)),
                "evidence": ev,
            }
        )

    # 5. Peer weights
    pv = _get(t, "peer_weights_validation")
    if len(pv):
        err, s = float(pv["max_abs_error"].max()), float(pv["max_sum_error"].max())
        rows.append(
            {
                "finding": "5. Peer weights reproduce every prediction and sum to one",
                "verdict": _verdict(err < 1e-5 and s < 1e-8),
                "evidence": f"{int(pv['n_test'].sum())} test firms in {len(pv)} models: max |v'K - prediction| = {err:.2e}, max |sum of weights - 1| = {s:.2e}",
            }
        )

    # 6. Portfolios
    ps = _get(t, "portfolio_stats_vw")
    if len(ps):
        base = ps[ps["model"].isin(BASE_MODELS)]
        ok = bool((base["mean"] > 0).all()) if len(base) and "mean" in base else None
        ev = "; ".join(
            f"{r.model}: mean {r.mean:.2f}% per month (p={r.mean_p:.3f}), Sharpe {r.sharpe:.2f}"
            for r in base.itertuples()
            if hasattr(r, "mean") and not pd.isna(r.mean)
        )
        rows.append(
            {
                "finding": "6. Long-short portfolios on GBM errors have positive mean returns",
                "verdict": _verdict(ok),
                "evidence": ev,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(thr)


def _section(
    title: str, df: pd.DataFrame, note: str = "", fmt: str = ".3f", cols: list[str] | None = None
) -> str:
    if df is None or df.empty:
        return f"### {title}\n\n_Not available in this run._\n"
    body = to_markdown(df[[c for c in cols if c in df.columns]] if cols else df, fmt)
    return f"### {title}\n\n{note + chr(10) + chr(10) if note else ''}{body}\n"


def write_report(cfg, t: dict[str, pd.DataFrame], figures: list[Path], log=print) -> Path:
    find, thr = findings(t, cfg)
    write_table(find, cfg.reports_dir("tables"), "findings")
    write_table(thr, cfg.reports_dir("tables"), "thresholds", ".4f")

    is_headline = cfg.run_name == "replication"
    out_dir = cfg.path("docs") if is_headline else cfg.reports_dir()
    out = out_dir / "REPLICATION_REPORT.md"
    rel = os.path.relpath(cfg.reports_dir("figures"), out_dir).replace("\\", "/")
    fig = {p.stem.split("_")[0]: f"![{p.stem}]({rel}/{p.name})" for p in figures}
    p = cfg["period"]
    lines = [
        "# Replication report: Geertsema & Lu (2023), Relative Valuation with Machine Learning",
        "",
        "> Generated by `relval report`. Do not edit by hand: rerun the pipeline instead.",
        "",
        f"- Run: `{cfg.run_name}`, source `{cfg['source']}`, valuation months {p['start']} to {p['end']}",
        f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}, git commit `{git_commit(cfg.root)[:10]}`",
        f"- Libraries: {', '.join(f'{k} {v}' for k, v in library_versions().items())}",
        f"- Block seeds: {cfg.seeds} (headline: {cfg.headline_seed})",
        "",
    ]
    if not is_headline:
        lines += [
            "**This is not the headline run** (reduced period or experiment). Do not quote its numbers.",
            "",
        ]
    lines += [
        "Our data are LSEG, 1990-2019, with survivorship bias; the paper uses CRSP/Compustat, 1980-2019.",
        "Method fidelity is exact and tested; numerical agreement is measured here and explained in",
        "`DEVIATIONS.md`. Paper values come from `paper_targets/`.",
        "",
        "## 1. Findings and thresholds",
        "",
        _section("Qualitative findings (section 7 of the master prompt)", find),
        _section("Working thresholds", thr, "A threshold that fails is flagged for investigation, never forced.", ".4f"),
        "## 2. Sample (Table 1, Table IA.4)",
        "",
        _section("Table 1: sample selection", _get(t, "panel_table1"), fmt=".1f"),
        _section("Targets", _get(t, "panel_targets")),
        _section("Table IA.4: coverage and medians of the 97 inputs", _get(t, "ia4_summary_statistics"),
                 "Coverage is the share of sample firm-months with a value.", cols=["variable", "category", "coverage", "paper_coverage", "coverage_diff", "median", "paper_median", "mean", "paper_mean"]),  # fmt: skip
        "## 3. Model performance (Table 2, Table IA.5, Figures 1-3)",
        "",
        _section("Table 2: ours next to the paper", _get(t, "table2_vs_paper"), "Columns ending in `_paper` are the paper's (1980-2019). BG is approximate."),
        _section("Seed spread of the base models", _get(t, "table2_seed_spread"), "Five block seeds; `headline` is the first seed.", ".4f"),
        _section("Table IA.5: common subsample", _get(t, "ia5_vs_paper") if len(_get(t, "ia5_vs_paper")) else _get(t, "ia5_common_sample")),
        _section("Share of months with the GBM below each benchmark", _get(t, "months_gbm_below_benchmarks")),
        _section("Negative predicted equity values", _get(t, "negative_equity_values"), fmt=".4f"),
        _section("Table 2 by decade", _get(t, "table2_by_decade"), cols=["period", "model", "N", "med_abs_eps", "med_eps", "r2oos_y", "r2oos_M", "rho"]),
        fig.get("figure1", ""), "", fig.get("figure2", ""), "", fig.get("figure3", ""), "",
        "## 4. Sources of the gains and robustness (Table 4, IA.6, sensitivity runs)",
        "",
        _section("Table 4: ours next to the paper", _get(t, "table4_vs_paper")),
        _section("CART, alternative targets, 80:20 split", _get(t, "ia6_vs_paper") if len(_get(t, "ia6_vs_paper")) else _get(t, "ia6_robustness"), cols=["model", "variant", "N", "med_abs_eps", "med_abs_eps_paper", "med_eps", "med_eps_paper", "r2oos_M", "r2oos_M_paper", "rho", "rho_paper"]),
        _section("Sensitivity runs", _get(t, "sensitivity_runs"), cols=["model", "variant", "N", "med_abs_eps", "med_abs_eps_minus_base", "med_eps", "r2oos_y", "r2oos_M"]),
        "## 5. Valuation errors and future returns (Table 3, Table IA.7)",
        "",
        "NYSE membership is today's exchange (RIC suffix), delisting returns are not available and the",
        "sample is a survivor sample: this is a weaker test than the paper's. The 13-factor model is a blocker.",
        "",
        _section("Table 3: value-weighted long-short portfolios", _get(t, "table3_value_weighted")),
        _section("Table 3 next to the paper", _get(t, "table3_vs_paper")),
        _section("Table IA.7: equal-weighted", _get(t, "ia7_equal_weighted")),
        "## 6. Variable importance (Tables 7, 8, 9, IA.8)",
        "",
        _section("Table 7: ten most important variables per model", _get(t, "table7")),
        _section("Paper's top-10 variables in our ranking", _get(t, "table7_vs_paper")),
        _section("Table 8: category importance next to the paper", _get(t, "table8_vs_paper") if len(_get(t, "table8_vs_paper")) else _get(t, "table8")),
        _section("Table 9: largest changes over time", _get(t, "table9")),
        _section("Exact duplicate pairs, summed", _get(t, "shap_duplicate_pairs")),
        "The full 97-variable table (IA.8) is in `tables/ia8_variable_importance.csv`.",
        "",
        "## 7. Peer weights (Table 5, Figures 4-5)",
        "",
        _section("Validation of the decomposition", _get(t, "peer_weights_validation"), fmt=".3g"),
        _section("Focus firm", _get(t, "table5_summary"), fmt=".4f"),
        _section("Table 5: most and least comparable peers", _get(t, "table5")),
        fig.get("figure4", ""), "", fig.get("figure5", ""), "",
    ]  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"[done] report: {out}")
    return out
