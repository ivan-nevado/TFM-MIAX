"""Month-by-month training and out-of-sample prediction for every model.

All models share the same five blocks. The GBM trains on three blocks and uses the fourth for
early stopping; benchmarks need no validation set and are fitted on the four non-test blocks.
Every firm-month gets exactly one out-of-sample prediction per model, written to the master
predictions table (one parquet file per model, seed and year), from which every metric, figure,
portfolio and comparison is computed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from relval.features.registry import FEATURES
from relval.io import read_parquet, read_partitioned, write_parquet
from relval.models.benchmarks import bg, hp, rrv
from relval.models.benchmarks.bl import BL_FEATURES, bl_features, ols_fit_predict
from relval.models.benchmarks.ind import predict_ind
from relval.models.cart import fit_predict_cart
from relval.models.gbm import fit_gbm
from relval.models.lasso import design_matrix, fit_predict_lasso
from relval.models.specs import BASE_MODELS, SPECS, ModelSpec
from relval.splits.blocks import add_blocks, roles
from relval.targets.multiples import target_values, to_equity_value, valuation_error

PRED_COLUMNS = [
    "model", "target", "id", "date", "block", "seed", "y", "yhat", "M", "Mhat", "eps",
    "n_train", "train_mean_y", "train_mean_M", "best_iteration",
]  # fmt: skip
RESULT_COLUMNS = ["yhat", "n_train", "train_mean_y", "train_mean_M", "best_iteration"]


@dataclass
class MonthResult:
    pred: pd.DataFrame
    shap: pd.DataFrame | None = None
    models: list[dict] = field(default_factory=list)


def _frame(index, yhat, n_train, mean_y, mean_m, best_iteration=np.nan) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "yhat": yhat,
            "n_train": n_train,
            "train_mean_y": mean_y,
            "train_mean_M": mean_m,
            "best_iteration": best_iteration,
        },
        index=index,
    )


def gbm_inputs(
    spec: ModelSpec, df: pd.DataFrame, fit: np.ndarray, cfg: dict
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Feature frame, column names and categorical columns of a GBM spec for one month."""
    if spec.features == "bl10":
        return bl_features(df, fit, int(cfg["benchmarks"]["ind_min_firms"])), list(BL_FEATURES), []
    X = df[FEATURES]
    beta_col = spec.options.get("beta_column")
    if beta_col:
        X = X.assign(CAPMbeta=df[beta_col].to_numpy())
    categorical = cfg["gbm"].get("industry_categorical", True) and not spec.options.get(
        "industry_numeric", False
    )
    return X, list(FEATURES), (["industry"] if categorical else [])


def sample_mask(spec: ModelSpec, df: pd.DataFrame, fit: np.ndarray, cfg: dict) -> np.ndarray:
    if spec.sample == "bl":
        return bl_features(df, fit, int(cfg["benchmarks"]["ind_min_firms"])).notna().all(axis=1).to_numpy()
    if spec.sample == "nm":
        return df[FEATURES].notna().all(axis=1).to_numpy()
    return np.ones(len(df), dtype=bool)


def block_seed(seed: int, date: pd.Timestamp, k: int) -> int:
    """LightGBM seed of one model: unique per base seed, month and test block."""
    return (int(seed) + date.year * 100 + date.month) * 10 + k


@dataclass
class GbmBlock:
    model: object  # relval.models.gbm.GbmFit
    X: np.ndarray
    columns: list[str]
    train: np.ndarray  # rows the trees were built on
    test: np.ndarray


def fit_gbm_block(
    spec: ModelSpec, df: pd.DataFrame, y: pd.Series, k: int, cfg: dict, seed: int, ctx: dict
) -> GbmBlock | None:
    """Fit the GBM of one test block. Deterministic given (spec, month sample, seed, k), so the
    explainability stages can rebuild exactly the model behind the stored predictions."""
    n_blocks = int(cfg["splits"]["n_blocks"])
    train, val, test = roles(df["block"].to_numpy(), k, n_blocks)
    fit = ~test
    yv = y.to_numpy(dtype=float)
    min_train = int(cfg["gbm"]["min_train_rows"])
    Xdf, cols, cat = gbm_inputs(spec, df, fit, cfg)
    mask = ~np.isnan(yv) & sample_mask(spec, df, fit, cfg)
    tr, va, te = train & mask, val & mask, test & mask
    X = Xdf.to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan
    bseed = block_seed(seed, df["date"].iloc[0], k)
    if spec.split == "80_20":
        rounds = cfg["gbm"].get("fixed_rounds_80_20", 20)
        tr = tr | va
        if rounds is None or tr.sum() < min_train or te.sum() == 0:
            return None
        model = fit_gbm(X[tr], yv[tr], None, None, cols, cfg["gbm"], bseed, cat, fixed_rounds=int(rounds))
    else:
        if tr.sum() < min_train or va.sum() < 5 or te.sum() == 0:
            return None
        model = fit_gbm(X[tr], yv[tr], X[va], yv[va], cols, cfg["gbm"], bseed, cat)
    return GbmBlock(model, X, cols, tr, te)


def _run_block(spec: ModelSpec, df: pd.DataFrame, y: pd.Series, k: int, cfg: dict, seed: int, ctx: dict):
    """One test block. Returns (result frame indexed like df, shap frame or None, model dict or None)."""
    n_blocks = int(cfg["splits"]["n_blocks"])
    train, val, test = roles(df["block"].to_numpy(), k, n_blocks)
    fit = ~test
    bench = cfg["benchmarks"]
    yv, m = y.to_numpy(dtype=float), df["M"].to_numpy(dtype=float)
    valid_y = ~np.isnan(yv)
    min_train = int(cfg["gbm"]["min_train_rows"])
    date = df["date"].iloc[0]

    if spec.kind == "gbm":
        blk = fit_gbm_block(spec, df, y, k, cfg, seed, ctx)
        if blk is None:
            return None, None, None
        model, X, cols, tr, te = blk.model, blk.X, blk.columns, blk.train, blk.test
        res = _frame(
            df.index[te],
            model.predict(X[te]),
            int(tr.sum()),
            yv[tr].mean(),
            m[tr].mean(),
            model.best_iteration,
        )
        shap = None
        if ctx.get("want_shap") and spec.options.get("shap"):
            shap = pd.DataFrame(
                model.shap(X[te]).astype(np.float32), columns=[*cols, "base_value"], index=df.index[te]
            )
        saved = None
        if ctx.get("want_models") and spec.options.get("save"):
            saved = {"date": date, "block": k, "best_iteration": model.best_iteration, "n_train": int(tr.sum()),
                     "model_text": model.to_string()}  # fmt: skip
        return res, shap, saved

    if spec.kind in ("bl", "ols_log"):
        feats = bl_features(df, fit, int(bench["ind_min_firms"]))
        complete = feats.notna().all(axis=1).to_numpy() & valid_y
        f, t = fit & complete, test & complete
        if f.sum() <= len(BL_FEATURES) + 1 or t.sum() == 0:
            return None, None, None
        yhat = ols_fit_predict(feats[f].to_numpy(), yv[f], feats[t].to_numpy())
        return _frame(df.index[t], yhat, int(f.sum()), yv[f].mean(), m[f].mean()), None, None

    if spec.kind == "ind":
        pred = predict_ind(df, fit, spec.target, int(bench["ind_min_firms"])).to_numpy(dtype=float)
        t = test & ~np.isnan(pred) & valid_y
        f = fit & valid_y
        if t.sum() == 0:
            return None, None, None
        return _frame(df.index[t], pred[t], int(f.sum()), yv[f].mean(), m[f].mean()), None, None

    if spec.kind == "hp":
        res = hp.predict_hp(df, fit, test, spec.target, ctx.get("tnic_pairs"), int(bench["hp_min_peers"]))
        return (res.assign(best_iteration=np.nan) if len(res) else None), None, None

    if spec.kind == "rrv":
        res = rrv.predict_rrv(df, fit, test, int(bench["rrv_min_firms"]))
        return (res.assign(best_iteration=np.nan) if len(res) else None), None, None

    if spec.kind == "bg":
        res = bg.predict_bg(df, fit, test, list(bench["bg_items"]), list(bench.get("bg_zero_fill") or []))
        return (res.assign(best_iteration=np.nan) if len(res) else None), None, None

    if spec.kind in ("lasso", "cart"):
        mask = valid_y & sample_mask(spec, df, fit, cfg)
        tr, va, te = train & mask, val & mask, test & mask
        if tr.sum() < min_train or va.sum() < 5 or te.sum() == 0:
            return None, None, None
        if spec.kind == "lasso":
            sub = df[mask]
            Z = design_matrix(sub, FEATURES, tr[mask])
            yhat, extra = fit_predict_lasso(
                Z, yv[mask], tr[mask], va[mask], te[mask], int(bench["lasso_n_alphas"])
            )
        else:
            X = df[FEATURES].to_numpy(dtype=float)
            X[~np.isfinite(X)] = np.nan
            yhat, extra = fit_predict_cart(
                X, yv, tr, va, te, list(bench["cart_depths"]), block_seed(seed, date, k)
            )
        return _frame(df.index[te], yhat, int(tr.sum()), yv[tr].mean(), m[tr].mean(), extra), None, None

    raise ValueError(f"unknown model kind {spec.kind!r}")


def run_month(
    spec: ModelSpec, month: pd.DataFrame, cfg: dict, seed: int, ctx: dict | None = None
) -> MonthResult:
    """Train and predict one model on one monthly cross-section (all five test blocks)."""
    ctx = ctx or {}
    n_blocks = int(cfg["splits"]["n_blocks"])
    df = add_blocks(month, seed, n_blocks)  # blocks come from the full monthly sample
    if spec.sample == "listed":
        df = df[df["origen"] == "final"].reset_index(drop=True)
    y = target_values(df, spec.target)

    results, shaps, models = [], [], []
    for k in range(n_blocks):
        res, shap, saved = _run_block(spec, df, y, k, cfg, seed, ctx)
        if res is None or len(res) == 0:
            continue
        results.append(res[RESULT_COLUMNS])
        if shap is not None:
            shaps.append(shap)
        if saved is not None:
            models.append(saved)

    if not results:
        return MonthResult(pd.DataFrame(columns=PRED_COLUMNS))
    res = pd.concat(results)
    base = df.loc[res.index]
    pred = pd.DataFrame(
        {
            "model": spec.name, "target": spec.target, "id": base["id"].to_numpy(), "date": base["date"].to_numpy(),
            "block": base["block"].to_numpy(), "seed": int(seed), "y": y.loc[res.index].to_numpy(dtype=float),
            "yhat": res["yhat"].to_numpy(dtype=float), "M": base["M"].to_numpy(dtype=float),
        }
    )  # fmt: skip
    pred["Mhat"] = to_equity_value(pred["yhat"], spec.target, base["B"], base["A"], base["S"], base["D"])
    pred["eps"] = valuation_error(pred["Mhat"], pred["M"])
    for col in ("n_train", "train_mean_y", "train_mean_M", "best_iteration"):
        pred[col] = res[col].to_numpy(dtype=float)
    shap_df = None
    if shaps:
        s = pd.concat(shaps)
        ident = df.loc[s.index, ["id", "date", "block"]].reset_index(drop=True)
        shap_df = pd.concat([ident, s.reset_index(drop=True)], axis=1)
    return MonthResult(pred[PRED_COLUMNS], shap_df, models)


# ---------------------------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------------------------
def load_sample(cfg) -> pd.DataFrame:
    """The modelling sample (Table 1, last step)."""
    panel = read_parquet(cfg.panel_path())
    start = pd.Timestamp(cfg["period"]["start"]) + pd.offsets.MonthEnd(0)
    end = pd.Timestamp(cfg["period"]["end"]) + pd.offsets.MonthEnd(0)
    keep = panel["in_sample"] & (panel["date"] >= start) & (panel["date"] <= end)
    return panel[keep].reset_index(drop=True)


def load_predictions(cfg, models: list[str], seed: int | None = None) -> pd.DataFrame:
    seed = cfg.headline_seed if seed is None else seed
    frames = [read_partitioned(cfg.predictions_dir(m, seed)) for m in models]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRED_COLUMNS)


def _mark_unavailable(cfg, model: str, reason: str) -> None:
    path = cfg.artifacts_dir("status") / "unavailable.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(path.read_text()) if path.exists() else {}
    status[model] = reason
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def _context(spec: ModelSpec, cfg, seed: int) -> tuple[dict | None, str]:
    """Per-model context shared by all months. ``None`` means the model cannot run."""
    if spec.kind == "hp":
        if not hp.tnic_available(cfg.path("external")):
            return None, "TNIC similarity data or the gvkey link file is missing (see BLOCKERS.md)"
        return {"tnic": hp.load_tnic(cfg.path("external"), int(cfg["benchmarks"]["hp_n_peers"]))}, ""
    return {}, ""


def train_models(cfg, models: list[str], seeds: list[int] | None = None, years: list[int] | None = None,
                 force: bool = False, log=print) -> dict[str, int]:  # fmt: skip
    """Train the given models. Resumable: a (model, seed, year) with a predictions file is skipped."""
    years = years or cfg.years()
    sample = None
    written: dict[str, int] = {}
    for name in models:
        spec = SPECS[name]
        model_seeds = seeds or (cfg.seeds if name in BASE_MODELS else [cfg.headline_seed])
        for seed in model_seeds:
            out_dir = cfg.predictions_dir(name, seed)
            pending = [y for y in years if force or not (out_dir / f"year={y}.parquet").exists()]
            if not pending:
                continue
            ctx_all, reason = _context(spec, cfg, seed)
            if ctx_all is None:
                _mark_unavailable(cfg, name, reason)
                log(f"[skip] {name}: {reason}")
                break
            if sample is None:
                sample = load_sample(cfg)
            headline = seed == cfg.headline_seed
            base_ctx = {
                "want_shap": headline and bool(cfg["gbm"].get("save_shap", True)),
                "want_models": headline and bool(cfg["gbm"].get("save_models", True)),
            }
            part = sample[sample["date"].dt.year.isin(pending)]

            def tasks(part=part, ctx_all=ctx_all, base_ctx=base_ctx, spec=spec, seed=seed):
                for date, month in part.groupby("date", sort=True):
                    ctx = dict(base_ctx)
                    if "tnic" in ctx_all:
                        ctx["tnic_pairs"] = ctx_all["tnic"].get(date.year - 1)
                    yield delayed(run_month)(spec, month, cfg.data, seed, ctx)

            results: list[MonthResult] = Parallel(n_jobs=cfg.n_jobs)(tasks())
            n_rows = 0
            for year in pending:
                preds = [
                    r.pred
                    for r in results
                    if len(r.pred) and pd.Timestamp(r.pred["date"].iloc[0]).year == year
                ]
                pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=PRED_COLUMNS)
                shaps = [
                    r.shap for r in results if r.shap is not None and r.shap["date"].iloc[0].year == year
                ]
                mods = [m for r in results for m in r.models if m["date"].year == year]
                if shaps:
                    write_parquet(
                        pd.concat(shaps, ignore_index=True),
                        cfg.artifacts_dir("shap", name, seed) / f"year={year}.parquet",
                    )
                if mods:
                    write_parquet(
                        pd.DataFrame(mods), cfg.artifacts_dir("models", name, seed) / f"year={year}.parquet"
                    )
                write_parquet(pred, out_dir / f"year={year}.parquet")  # written last: marks the year as done
                n_rows += len(pred)
            written[f"{name}|{seed}"] = n_rows
            log(f"[done] {name} seed={seed} years={pending[0]}-{pending[-1]} rows={n_rows:,}")
    return written
