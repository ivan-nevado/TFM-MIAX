"""``relval compare``: every experiment against its baseline, on exactly the same firm-months.

For each experiment and model the predictions are paired with the baseline's (same model name,
headline seed, matched on firm and month), so a difference comes from the method, not the sample.
Outputs, under ``reports/experiments/``:

    comparison/leaderboard.csv|md   one row per experiment and model: Table 2 metrics of both runs on
                                    the common sample, their difference, share of months the
                                    experiment wins, a bootstrap CI over months and the seed noise
    comparison/by_decade.csv        the same differences by decade (E2 targets the thin 1990s)
    comparison/by_size.csv          ... by monthly size quintile
    comparison/monthly.csv          med(|eps|) per month of both runs (for plots)
    HISTORY.md                      catalogue: hypothesis, every training run (date, commit, image,
                                    status) and the headline result of each experiment
    history/<utc>/leaderboard.csv   snapshot of every comparison ever run (never overwritten)

Reading rule (fixed before any result): an experiment improves on the baseline for a model when
the CI of the mean monthly difference of med(|eps|) is below zero AND the pooled difference is
larger in size than the baseline's seed spread.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from relval.config import load_config
from relval.evaluation.by_group import add_quintiles
from relval.evaluation.metrics import compute_metrics
from relval.experiments.registry import Experiment, baseline_config, discover
from relval.io import read_partitioned, to_markdown, write_table
from relval.models.runner import load_sample

HEADLINE_METRICS = ["med_abs_eps", "avg_abs_eps", "med_eps", "rmse_y", "r2oos_y", "r2oos_M", "rho"]
N_BOOT = 2000


def _preds(cfg, model: str, seed: int) -> pd.DataFrame:
    return read_partitioned(cfg.predictions_dir(model, seed))


def paired(base: pd.DataFrame, exp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both prediction sets to the firm-months that both have, row-aligned (same order)."""
    keys = base[["id", "date"]].merge(exp[["id", "date"]], on=["id", "date"])

    def align(p: pd.DataFrame) -> pd.DataFrame:
        return p.merge(keys, on=["id", "date"]).sort_values(["date", "id"]).reset_index(drop=True)

    return align(base), align(exp)


def monthly_med_abs(pred: pd.DataFrame) -> pd.Series:
    return pred.assign(a=pred["eps"].abs()).groupby("date")["a"].median()


def bootstrap_mean_ci(d: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of the mean of ``d``, resampling months (fixed seed: reproducible)."""
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def seed_noise(cfg, model: str) -> float:
    """Range (max - min) of the pooled med(|eps|) across the baseline's seeds; NaN with one seed."""
    vals = []
    for s in cfg.seeds:
        p = _preds(cfg, model, s)
        if len(p):
            vals.append(float(np.median(np.abs(p["eps"]))))
    return float(max(vals) - min(vals)) if len(vals) > 1 else np.nan


def _delta_table(base: pd.DataFrame, exp: pd.DataFrame, key: pd.Series, name: str) -> pd.DataFrame:
    rows = []
    for g in sorted(key.dropna().unique()):
        b, e = base[key.loc[base.index] == g], exp[key.loc[exp.index] == g]
        if len(b) == 0:
            continue
        mb, me = np.median(np.abs(b["eps"])), np.median(np.abs(e["eps"]))
        rows.append({name: g, "N": len(b), "base_med_abs_eps": mb, "exp_med_abs_eps": me, "delta": me - mb})
    return pd.DataFrame(rows)


def compare_model(exp: Experiment, base_cfg, model: str, size_q: pd.DataFrame | None) -> dict | None:
    seed = exp.cfg.headline_seed
    e_all = _preds(exp.cfg, model, seed)
    b_all = _preds(base_cfg, model, base_cfg.headline_seed)
    if e_all.empty or b_all.empty:
        return None
    b, e = paired(b_all, e_all)
    mb, me = compute_metrics(b), compute_metrics(e)
    month_b, month_e = monthly_med_abs(b), monthly_med_abs(e)
    d = (month_e - month_b).dropna()
    lo, hi = bootstrap_mean_ci(d.to_numpy())
    noise = seed_noise(base_cfg, model)
    delta = me["med_abs_eps"] - mb["med_abs_eps"]
    row = {
        "experiment": exp.id, "family": exp.family, "title": exp.title, "model": model,
        "baseline": base_cfg.run_name,
        "N_common": mb["N"], "coverage_vs_base": len(e_all) / len(b_all),
        "years": f"{e['date'].min():%Y}-{e['date'].max():%Y}",
        **{f"base_{k}": mb[k] for k in HEADLINE_METRICS},
        **{f"exp_{k}": me[k] for k in HEADLINE_METRICS},
        "delta_med_abs_eps": delta,
        "share_months_better": float((d < 0).mean()) if len(d) else np.nan,
        "mean_monthly_delta": float(d.mean()) if len(d) else np.nan,
        "ci95_low": lo, "ci95_high": hi,
        "base_seed_noise": noise,
        "improves": bool(hi < 0 and (np.isnan(noise) or -delta > noise)),
        "mean_n_train": float(e["n_train"].mean()), "base_mean_n_train": float(b["n_train"].mean()),
        "mean_trees": float(e["best_iteration"].mean()), "base_mean_trees": float(b["best_iteration"].mean()),
    }  # fmt: skip
    ident = {"experiment": exp.id, "model": model}
    decade = (b["date"].dt.year // 10 * 10).rename("decade")
    by_dec = _delta_table(b, e, decade, "decade").assign(**ident)
    by_size = pd.DataFrame()
    if size_q is not None:
        q = b[["id", "date"]].merge(size_q, on=["id", "date"], how="left")["q_size"]
        q.index = b.index
        by_size = _delta_table(b, e, q, "size_quintile").assign(**ident)
    monthly = pd.DataFrame({"base": month_b, "exp": month_e}).reset_index().assign(**ident)
    return {"row": row, "by_decade": by_dec, "by_size": by_size, "monthly": monthly}


def _manifests(cfg) -> list[dict]:
    folder = cfg.artifacts_dir("manifests")
    out = []
    for f in sorted(folder.glob("train_*.json")) if folder.exists() else []:
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({k: m.get(k) for k in ("started_utc", "status", "git_commit", "image_tag", "seconds",
                                           "batch_job_id", "years", "models", "seeds")})  # fmt: skip
    return out


def _history_md(experiments: list[Experiment], board: pd.DataFrame) -> str:
    lines = [
        "# Experiment history",
        "",
        "> Generated by `relval compare`. Do not edit: the source of truth is the YAML of each",
        "> experiment (`ml/configs/experiments/`) and the run manifests.",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. Each experiment is paired",
        "with its baseline run (column `baseline`, headline seed) on the same firm-months. `improves` = CI of the mean monthly difference",
        "below zero and the pooled difference larger than the baseline's seed spread.",
        "",
    ]
    if len(board):
        cols = ["experiment", "model", "baseline", "N_common", "base_med_abs_eps", "exp_med_abs_eps", "delta_med_abs_eps",
                "share_months_better", "ci95_low", "ci95_high", "improves"]  # fmt: skip
        lines += ["## Summary", "", to_markdown(board[cols], ".4f"), ""]
    for exp in experiments:
        lines += [f"## {exp.id}: {exp.title}", "", f"- Family: {exp.family}",
                  f"- Config: `ml/configs/{exp.config_name}`, run `{exp.cfg.run_name}`",
                  f"- Models: {', '.join(exp.models)}; seeds {exp.cfg.seeds}",
                  f"- Hypothesis: {exp.hypothesis or '(none written)'}", ""]  # fmt: skip
        runs = _manifests(exp.cfg)
        if runs:
            lines += [to_markdown(pd.DataFrame(runs).astype(str)), ""]
        else:
            lines += ["_No training run recorded yet._", ""]
    return "\n".join(lines)


def run(log=print, root: Path | None = None) -> dict[str, int]:
    experiments = discover(root)
    if not experiments:
        log("[skip] compare: no experiment configs")
        return {}
    base_cfg = load_config("replication.yaml", root=experiments[0].cfg.root)  # main baseline
    out = base_cfg.path("reports") / "experiments"
    size_q = None
    try:
        size_q = add_quintiles(load_sample(base_cfg))[["id", "date", "q_size"]]
    except (OSError, KeyError) as exc:
        log(f"[warn] compare: no size quintiles ({exc!r})")
    rows, dec, size, monthly = [], [], [], []
    for exp in experiments:
        bcfg = baseline_config(exp)
        for model in exp.models:
            res = compare_model(exp, bcfg, model, size_q if bcfg.run_name == base_cfg.run_name else None)
            if res is None:
                log(f"[skip] compare {exp.id} {model}: predictions missing")
                continue
            rows.append(res["row"])
            dec.append(res["by_decade"])
            size.append(res["by_size"])
            monthly.append(res["monthly"])
            r = res["row"]
            log(f"[done] {exp.id} {model}: med|eps| {r['base_med_abs_eps']:.4f} -> {r['exp_med_abs_eps']:.4f} "
                f"(CI of monthly diff {r['ci95_low']:+.4f}, {r['ci95_high']:+.4f})")  # fmt: skip
    board = pd.DataFrame(rows)
    cmp_dir = out / "comparison"
    if len(board):
        write_table(board, cmp_dir, "leaderboard", ".4f")
        write_table(pd.concat(dec, ignore_index=True), cmp_dir, "by_decade", ".4f")
        if any(len(s) for s in size):
            write_table(pd.concat(size, ignore_index=True), cmp_dir, "by_size", ".4f")
        pd.concat(monthly, ignore_index=True).to_csv(cmp_dir / "monthly.csv", index=False)
        snap = out / "history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cmp_dir / "leaderboard.csv", snap / "leaderboard.csv")
    out.mkdir(parents=True, exist_ok=True)
    (out / "HISTORY.md").write_text(_history_md(experiments, board), encoding="utf-8")
    log(f"[done] compare: {len(board)} experiment-model rows in {cmp_dir}")
    return {"rows": len(board)}
