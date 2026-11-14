"""Every table of the replication, built from the master predictions table and the stored SHAP
values and peer weights. Each table is written as CSV and Markdown under ``reports/<run>/tables``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from relval.data import french
from relval.evaluation import by_group, by_time, portfolios
from relval.evaluation.common_sample import common_sample_table
from relval.evaluation.metrics import METRIC_COLUMNS, metrics_table, seed_spread
from relval.evaluation.reconciliation import COLUMNS as T4_COLUMNS
from relval.evaluation.reconciliation import reconciliation_table
from relval.explain import shap_values as sv
from relval.features.registry import CATEGORY_OF, DESCRIPTIONS, FEATURES, expected_sign
from relval.io import read_parquet, read_partitioned, write_table
from relval.models.runner import load_predictions, load_sample
from relval.models.specs import BASE_MODELS, LOGS, MULTS, SPECS, TABLE2_MODELS
from relval.reporting.compare import compare_long, load_paper, side_by_side


class Tables:
    """Builds and stores the tables; keeps them in memory for the figures and the report."""

    def __init__(self, cfg, log=print):
        self.cfg, self.log = cfg, log
        self.dir = cfg.reports_dir("tables")
        self.t: dict[str, pd.DataFrame] = {}
        self.sample = load_sample(cfg)
        self.preds = load_predictions(cfg, list(SPECS))
        self.available = set(self.preds["model"].unique())

    def save(self, name: str, df: pd.DataFrame, floatfmt: str = ".3f") -> pd.DataFrame:
        self.t[name] = df
        write_table(df, self.dir, name, floatfmt)
        return df

    def have(self, models: list[str]) -> list[str]:
        return [m for m in models if m in self.available]

    # ---- panel ----------------------------------------------------------------------------
    def panel_tables(self) -> None:
        ctl = self.cfg.path("processed") / "controls" / self.cfg["panel_name"]
        for name in ("table1", "targets", "by_year", "notes"):
            if (ctl / f"{name}.csv").exists():
                self.save(f"panel_{name}", pd.read_csv(ctl / f"{name}.csv"))
        if (ctl / "variables.csv").exists():
            var = pd.read_csv(ctl / "variables.csv")
            paper = load_paper(self.cfg, "ia4_ia8")
            if paper is not None:
                var = var.merge(
                    paper[["variable", "coverage"]].rename(columns={"coverage": "paper_coverage"}),
                    on="variable",
                    how="left",
                )
                var["coverage_diff"] = var["coverage"] - var["paper_coverage"]
            stats = load_paper(self.cfg, "ia4_statistics")
            if stats is not None:
                var = var.merge(stats[["variable", "paper_mean", "paper_median"]], on="variable", how="left")
            self.save("ia4_summary_statistics", var)

    # ---- performance ----------------------------------------------------------------------
    def performance(self) -> None:
        models = self.have(TABLE2_MODELS)
        t2 = metrics_table(self.preds, models)
        paper = load_paper(self.cfg, "table2")
        self.save("table2", t2)
        self.save("table2_vs_paper", side_by_side(t2, paper, ["model"], METRIC_COLUMNS))
        self.save("table2_long", compare_long(t2, paper, ["model"], METRIC_COLUMNS))
        self.save(
            "negative_equity_values",
            t2[["model", "N", "n_negative_Mhat"]].assign(share=lambda d: d["n_negative_Mhat"] / d["N"]),
            ".4f",
        )

        by_seed = {s: load_predictions(self.cfg, BASE_MODELS, s) for s in self.cfg.seeds}
        by_seed = {s: p for s, p in by_seed.items() if len(p)}
        if by_seed:
            self.save("table2_seed_spread", seed_spread(by_seed, BASE_MODELS), ".4f")

        common = [m for m in models if not m.startswith("HP_")] + [m for m in models if m.startswith("HP_")]
        no_hp = [m for m in common if not m.startswith("HP_")]
        ia5 = common_sample_table(self.preds, no_hp)
        self.save("ia5_common_sample", ia5)
        self.save("ia5_vs_paper", side_by_side(ia5, load_paper(self.cfg, "ia5"), ["model"], METRIC_COLUMNS))
        if any(m.startswith("HP_") for m in common):
            self.save("ia5_common_sample_with_hp", common_sample_table(self.preds, common))

        monthly = by_time.monthly_median_abs_error(self.preds[self.preds["model"].isin(models)])
        self.save("fig1_monthly_median_abs_error", monthly.reset_index(), ".4f")
        shares = []
        for t, m in zip(LOGS, MULTS, strict=True):
            shares.append(
                by_time.share_of_months_below(
                    monthly, f"ML_{t}", [f"BL_{m}", f"IND_{m}", f"HP_{m}", "RRV", "BG"]
                )
            )
        self.save("months_gbm_below_benchmarks", pd.concat(shares, ignore_index=True))

        core = self.have(
            [*BASE_MODELS, *[f"BL_{m}" for m in MULTS], *[f"IND_{m}" for m in MULTS], "RRV", "BG"]
        )
        sub = self.preds[self.preds["model"].isin(core)]
        self.save("table2_by_decade", by_time.metrics_by_period(sub, core, "decade"))
        self.save("table2_by_year", by_time.metrics_by_period(sub, self.have(BASE_MODELS), "year"))

        groups = by_group.add_quintiles(self.sample)
        self.save(
            "fig2_fig3_errors_by_group",
            by_group.errors_by_group(self.preds[self.preds["model"].isin(models)], groups),
            ".4f",
        )

    # ---- reconciliation and robustness -------------------------------------------------------
    def reconciliation(self) -> None:
        t4 = reconciliation_table(self.preds)
        self.save("table4", t4)
        paper = load_paper(self.cfg, "table4")
        self.save(
            "table4_vs_paper",
            side_by_side(t4.drop(columns="model_id"), paper, ["panel", "model"], T4_COLUMNS),
        )

    def robustness(self) -> None:
        rows = []
        for t, m in zip(LOGS, MULTS, strict=True):
            rows += [(f"ML_{t}", "base GBM (60:20:20)", m), (f"CART_{t}", "CART", m), (f"ML8020_{t}", "GBM 80:20 split", m),
                     (f"GBMALT_{m}", "GBM, multiple in levels", m)]  # fmt: skip
        rows += [("GBMALT_lnM", "GBM, log equity value", "all"), ("GBMALT_M", "GBM, equity value", "all")]
        mt = metrics_table(self.preds, self.have([r[0] for r in rows]))
        info = pd.DataFrame(rows, columns=["model", "variant", "multiple"])
        ia6 = info.merge(mt, on="model", how="inner")
        self.save("ia6_robustness", ia6)
        paper6 = load_paper(self.cfg, "ia6")
        if paper6 is not None:
            self.save(
                "ia6_vs_paper",
                side_by_side(ia6, paper6.drop(columns="paper_model"), ["model"], METRIC_COLUMNS),
            )

        sens = []
        for t in LOGS:
            for suffix, label in [("", "base"), ("_indnum", "industry as numeric"), ("_listed", "listed-today firms only"),
                                  ("_beta24", "beta min 24 months"), ("_beta36", "beta min 36 months"), ("_beta48", "beta min 48 months")]:  # fmt: skip
                sens.append((f"ML_{t}{suffix}", label, t))
        info = pd.DataFrame(sens, columns=["model", "variant", "target"])
        mt = metrics_table(self.preds, self.have(list(info["model"])))
        out = info.merge(mt, on="model", how="inner")
        base = out[out["variant"] == "base"].set_index("target")["med_abs_eps"]
        out["med_abs_eps_minus_base"] = out["med_abs_eps"] - out["target"].map(base)
        self.save("sensitivity_runs", out)
        # listed-only models are evaluated on fewer firms: also show the base model on those firms
        rows = []
        for t in LOGS:
            listed_ids = self.sample.loc[self.sample["origen"] == "final", ["id", "date"]]
            base_listed = self.preds[self.preds["model"] == f"ML_{t}"].merge(listed_ids, on=["id", "date"])
            if len(base_listed):
                rows.append(
                    metrics_table(base_listed, [f"ML_{t}"]).assign(
                        variant="base model, listed-today firms only"
                    )
                )
        if rows:
            self.save("sensitivity_listed_subsample", pd.concat(rows, ignore_index=True))

    # ---- portfolios ------------------------------------------------------------------------
    def portfolios(self) -> None:
        factors = french.load_factors(self.cfg.path("external"))
        models = self.have(TABLE2_MODELS)
        pcfg = self.cfg["portfolios"]
        t3, ls_vw, perf_vw = portfolios.run_portfolios(self.preds, self.sample, factors, models, pcfg, "vw")
        ia7, ls_ew, perf_ew = portfolios.run_portfolios(self.preds, self.sample, factors, models, pcfg, "ew")
        self.save("table3_value_weighted", t3)
        self.save("ia7_equal_weighted", ia7)
        self.save("portfolio_returns_vw", ls_vw.reset_index(), ".6f")
        self.save("portfolio_returns_ew", ls_ew.reset_index(), ".6f")
        self.save("portfolio_stats_vw", pd.DataFrame(perf_vw).T.rename_axis("model").reset_index(), ".4f")
        self.save("portfolio_stats_ew", pd.DataFrame(perf_ew).T.rename_axis("model").reset_index(), ".4f")
        paper = load_paper(self.cfg, "table3")
        if paper is not None and len(t3):
            both = t3[t3["panel"] != "Sample"].merge(
                paper, on=["panel", "row"], how="left", suffixes=("", "_paper")
            )
            cols = ["panel", "row"] + [c for m in models for c in (m, f"{m}_paper") if c in both.columns]
            self.save("table3_vs_paper", both[cols])

    # ---- SHAP ------------------------------------------------------------------------------
    def shap(self) -> None:
        seed = self.cfg.headline_seed
        ia8, t7, t8, t9, dup = [], [], [], [], []
        paper78 = load_paper(self.cfg, "ia4_ia8")
        for target in LOGS:
            model = f"ML_{target}"
            raw = read_partitioned(self.cfg.artifacts_dir("shap", model, seed))
            if raw.empty:
                continue
            scaled = sv.scaled_abs_shap(raw)
            imp = sv.variable_importance(scaled, raw["date"])
            pred = self.preds.loc[self.preds["model"] == model, ["id", "date", "yhat"]]
            feats = raw[["id", "date"]].merge(
                self.sample[["id", "date", *FEATURES]], on=["id", "date"], how="left"
            )
            feats = feats.merge(pred, on=["id", "date"], how="left")
            var_rho, cat_rho = sv.rank_correlations(feats, feats["yhat"], feats["date"])
            imp["rho"] = imp["variable"].map(var_rho)
            imp["description"] = imp["variable"].map(DESCRIPTIONS)
            imp = imp.sort_values("shap", ascending=False).reset_index(drop=True)
            imp["cshap"] = imp["shap"].cumsum()
            imp["expected_sign"] = [expected_sign(target, c) for c in imp["category"]]
            imp.insert(0, "model", model)
            if paper78 is not None and f"shap_{target}" in paper78:
                imp = imp.merge(
                    paper78[["variable", f"shap_{target}"]].rename(columns={f"shap_{target}": "paper_shap"}),
                    on="variable",
                    how="left",
                )
            ia8.append(imp)
            t7.append(imp.head(10))

            cat = sv.category_importance(
                imp[
                    [
                        "variable",
                        "category",
                        "shap",
                        *[c for c in imp if c.startswith("d") and c[1:].isdigit()],
                        *(["chg"] if "chg" in imp else []),
                    ]
                ]
            )
            cat["rho"] = cat["category"].map(cat_rho)
            cat["expected_sign"] = [expected_sign(target, c) for c in cat["category"]]
            cat = cat.sort_values("shap", ascending=False)
            cat.insert(0, "model", model)
            t8.append(cat)

            dec = sorted(c for c in imp if c.startswith("d") and c[1:].isdigit())
            if len(dec) >= 2:
                pool = set(imp.nlargest(10, dec[0])["variable"]) | set(imp.nlargest(10, dec[-1])["variable"])
                movers = imp[imp["variable"].isin(pool)].copy()
                movers = movers.reindex(movers["chg"].abs().sort_values(ascending=False).index).head(10)
                t9.append(movers[["model", "variable", "category", "shap", *dec, "chg"]])
            dup.append(sv.duplicate_pair_importance(imp).assign(model=model))

        if not ia8:
            self.log("[skip] SHAP tables: no stored SHAP values")
            return
        keep = ["model", "variable", "description", "category", "shap", "cshap", "rho", "expected_sign"]
        all_imp = pd.concat(ia8, ignore_index=True)
        extra = [c for c in all_imp if c.startswith("d") and c[1:].isdigit()] + [
            c for c in ("chg", "paper_shap") if c in all_imp
        ]
        self.save("ia8_variable_importance", all_imp[[*keep, *extra]])
        top = pd.concat(t7, ignore_index=True)[keep]
        self.save("table7", top)
        paper7 = load_paper(self.cfg, "table7")
        if paper7 is not None:
            p = paper7.rename(columns={"shap": "paper_shap", "rho": "paper_rho"})[
                ["model", "variable", "paper_shap", "paper_rho"]
            ]
            ours = all_imp[["model", "variable", "shap", "rho"]].copy()
            ours["rank_ours"] = ours.groupby("model")["shap"].rank(ascending=False).astype(int)
            self.save("table7_vs_paper", p.merge(ours, on=["model", "variable"], how="left"))
        cats = pd.concat(t8, ignore_index=True)
        self.save("table8", cats)
        paper8 = load_paper(self.cfg, "table8")
        if paper8 is not None:
            p = paper8.rename(
                columns={
                    "shap": "paper_shap",
                    "rho": "paper_rho",
                    "d1990": "paper_d1990",
                    "d2000": "paper_d2000",
                    "d2010": "paper_d2010",
                }
            )
            p = p[
                ["model", "category", "paper_shap", "paper_rho", "paper_d1990", "paper_d2000", "paper_d2010"]
            ]
            self.save("table8_vs_paper", cats.merge(p, on=["model", "category"], how="left"))
        if t9:
            self.save("table9", pd.concat(t9, ignore_index=True))
        self.save("shap_duplicate_pairs", pd.concat(dup, ignore_index=True))

    # ---- peers -----------------------------------------------------------------------------
    def peers(self) -> None:
        pdir = self.cfg.artifacts_dir("peers")
        if (pdir / "validation.parquet").exists():
            self.save("peer_weights_validation", read_parquet(pdir / "validation.parquet"), ".3g")
        ex = self.cfg["explain"]
        tag = f"{ex['focus_target']}_{ex['focus_month'].replace('-', '')}"
        focus_file = pdir / f"peers_focus_{tag}.parquet"
        if not focus_file.exists():
            self.log("[skip] Table 5: the focus firm has no stored peer weights")
            return
        full = read_parquet(focus_file)
        self.save("fig4_focus_peer_weights", full[["rank", "peer_id", "weight", "cumweight"]], ".5f")
        firms = read_parquet(self.cfg.path("processed") / f"firms_{self.cfg['source']}.parquet")
        date = pd.Timestamp(ex["focus_month"]) + pd.offsets.MonthEnd(0)
        month = self.sample[self.sample["date"] == date].merge(
            firms[["id", "ff12_name", "ff49_name"]], on="id", how="left"
        )
        cols = [
            "id",
            "name",
            ex["focus_target"],
            "ff12_name",
            "ff49_name",
            "rd_sale_wr",
            "cash2assets_an",
            "de_ratio_wr",
            "roe_wr",
        ]
        info = month[cols].rename(columns={"id": "peer_id", "name": "company"})
        most = full.head(10).assign(group="most")
        least = full.tail(10).assign(group="least")
        t5 = pd.concat([most, least])[["group", "peer_id", "weight", "cumweight"]].merge(
            info, on="peer_id", how="left"
        )
        test = info[info["peer_id"] == ex["focus_firm"]].assign(group="test", weight=np.nan, cumweight=np.nan)
        t5 = pd.concat([test, t5], ignore_index=True)
        t5 = t5[["group", "company", ex["focus_target"], "weight", "cumweight", "ff12_name", "ff49_name",
                 "rd_sale_wr", "cash2assets_an", "de_ratio_wr", "roe_wr"]]  # fmt: skip
        self.save("table5", t5)
        summary = pd.DataFrame(
            [{"firm": ex["focus_firm"], "month": ex["focus_month"], "target": ex["focus_target"],
              "actual": float(month.loc[month["id"] == ex["focus_firm"], ex["focus_target"]].iloc[0]),
              "predicted": float(full["prediction"].iloc[0]), "sum_of_weights": float(full["weight"].sum()),
              "weighted_average_of_peers": float((full["weight"] * full["peer_y"]).sum()),
              "top10_cumweight": float(full["cumweight"].iloc[min(9, len(full) - 1)]),
              "negative_weights": int((full["weight"] < 0).sum()), "n_peers": len(full),
              "top10_same_ff49": int((most.merge(info, on="peer_id")["ff49_name"] == test["ff49_name"].iloc[0]).sum())}]
        )  # fmt: skip
        self.save("table5_summary", summary, ".4f")
        # Figure 5: raw SHAP waterfall of the focus firm
        raw = read_partitioned(
            self.cfg.artifacts_dir("shap", f"ML_{ex['focus_target']}", self.cfg.headline_seed)
        )
        if len(raw):
            row = raw[(raw["id"] == ex["focus_firm"]) & (raw["date"] == date)]
            if len(row):
                r = row.iloc[0]
                pred = float(r["base_value"] + r[FEATURES].astype(float).sum())
                wf = sv.waterfall(r, pred)
                wf["category"] = wf["label"].map(CATEGORY_OF)
                self.save("fig5_focus_shap_waterfall", wf, ".4f")

    def build_all(self) -> dict[str, pd.DataFrame]:
        steps = [self.panel_tables]
        if len(self.preds):
            steps += [self.performance, self.reconciliation, self.robustness, self.portfolios]
        steps += [self.shap, self.peers]
        for step in steps:
            try:
                step()
            except Exception as exc:  # noqa: BLE001 - one missing input must not stop the other tables
                self.log(f"[warn] tables.{step.__name__} failed: {exc!r}")
        return self.t
