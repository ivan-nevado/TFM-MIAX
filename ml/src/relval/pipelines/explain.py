"""Peer-weight stage: rebuild the December 2019 (configurable) base models, decompose every
out-of-sample prediction into peer weights and validate the decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from relval.explain.peer_weights import gbm_peer_weights
from relval.io import read_partitioned, write_parquet
from relval.models.runner import fit_gbm_block, load_sample
from relval.models.specs import SPECS
from relval.splits.blocks import add_blocks
from relval.targets.multiples import LOG_TARGETS, target_values

TOP_N, BOTTOM_N = 20, 10


def peer_weights_month(cfg, sample: pd.DataFrame, month: str, target: str, seed: int):
    """Peer weights of every test firm of one month and target.

    Returns (top/bottom peers in long format, full weights of the focus firm or None, validation rows).
    """
    spec = SPECS[f"ML_{target}"]
    date = pd.Timestamp(month) + pd.offsets.MonthEnd(0)
    df = add_blocks(sample[sample["date"] == date], seed, int(cfg["splits"]["n_blocks"]))
    if df.empty:
        return pd.DataFrame(), None, pd.DataFrame()
    y = target_values(df, target)
    stored = read_partitioned(cfg.predictions_dir(spec.name, seed))
    stored = stored[stored["date"] == date].set_index("id")["yhat"] if len(stored) else pd.Series(dtype=float)
    focus = cfg["explain"]["focus_firm"]
    lr = float(cfg["gbm"]["params"]["learning_rate"])
    ids = df["id"].to_numpy()
    tops, full, checks = [], None, []
    for k in range(int(cfg["splits"]["n_blocks"])):
        blk = fit_gbm_block(spec, df, y, k, cfg.data, seed, {})
        if blk is None:
            continue
        tr, te = blk.train, blk.test
        pw = gbm_peer_weights(blk.model, blk.X[tr], y.to_numpy(float)[tr], blk.X[te], learning_rate=lr)
        train_ids, test_ids = ids[tr], ids[te]
        diff_stored = np.nan
        if len(stored):
            diff_stored = float(np.nanmax(np.abs(pw.prediction_model - stored.reindex(test_ids).to_numpy())))
        checks.append(
            {"date": date, "target": target, "block": k, "n_train": int(tr.sum()), "n_test": int(te.sum()),
             "trees": blk.model.best_iteration, "max_abs_error": pw.max_abs_error, "max_sum_error": pw.max_sum_error,
             "leaves_with_cat_l2": pw.n_regularised_leaves, "max_diff_vs_stored_prediction": diff_stored}
        )  # fmt: skip
        K = pw.weights  # (n_train, n_test)
        order = np.argsort(-K, axis=0)
        for j, test_id in enumerate(test_ids):
            col, idx = K[:, j], order[:, j]
            cum = np.cumsum(col[idx])
            n = len(idx)
            pick = np.unique(np.concatenate([np.arange(min(TOP_N, n)), np.arange(max(n - BOTTOM_N, 0), n)]))
            tops.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "target": target,
                        "id": test_id,
                        "peer_id": train_ids[idx[pick]],
                        "weight": col[idx[pick]].astype(np.float32),
                        "cumweight": cum[pick].astype(np.float32),
                        "rank": pick + 1,
                        "n_peers": n,
                        "peer_y": pw.targets[idx[pick]].astype(np.float32),
                    }
                )  # fmt: skip
            )
            if test_id == focus:
                full = pd.DataFrame(
                    {"date": date, "target": target, "id": test_id, "peer_id": train_ids[idx], "weight": col[idx],
                     "cumweight": cum, "rank": np.arange(1, n + 1), "peer_y": pw.targets[idx],
                     "prediction": float(pw.prediction_model[j])}
                )  # fmt: skip
    top_df = pd.concat(tops, ignore_index=True) if tops else pd.DataFrame()
    return top_df, full, pd.DataFrame(checks)


def run(cfg, log=print) -> dict[str, int]:
    sample = load_sample(cfg)
    seed = cfg.headline_seed
    out_dir = cfg.artifacts_dir("peers")
    checks, n_rows = [], 0
    for month in cfg["explain"]["peer_months"]:
        if not (sample["date"] == pd.Timestamp(month) + pd.offsets.MonthEnd(0)).any():
            log(f"[skip] peers {month}: month not in the sample period")
            continue
        for target in LOG_TARGETS:
            top, full, chk = peer_weights_month(cfg, sample, month, target, seed)
            tag = f"{target}_{month.replace('-', '')}"
            if len(top):
                write_parquet(top, out_dir / f"peers_top_{tag}.parquet")
                n_rows += len(top)
            if full is not None:
                write_parquet(full, out_dir / f"peers_focus_{tag}.parquet")
            checks.append(chk)
            log(f"[done] peers {month} {target}: max |v'K - prediction| = {chk['max_abs_error'].max():.2e}"
                if len(chk) else f"[skip] peers {month} {target}")  # fmt: skip
    if checks:
        write_parquet(pd.concat(checks, ignore_index=True), out_dir / "validation.parquet")
    return {"peer_rows": n_rows}
