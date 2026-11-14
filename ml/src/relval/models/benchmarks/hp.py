"""HP: mean multiple of the 60 most similar firms (Hoberg & Phillips 2010, 2016).

For each firm, the 60 firms with the highest TNIC text-similarity score based on the prior year's
10-K; prediction = mean of those peers' current multiples among fitting-sample firms; at least
five peers with a multiple.

TNIC (https://hobergphillips.tuck.dartmouth.edu/industryclass.htm) is keyed by Compustat
``gvkey``, which the LSEG data do not have. The model runs only if both files below exist:

``data/external/tnic/tnic3_data.txt``     tab-separated: year, gvkey1, gvkey2, score
``data/external/tnic/gvkey_link.csv``     columns: gvkey, id   (id = the panel's firm identifier)

Without a legitimate gvkey <-> RIC mapping HP is a blocker (docs/replication/BLOCKERS.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TNIC_FILE = "tnic3_data.txt"
LINK_FILE = "gvkey_link.csv"


def tnic_available(external_dir: Path) -> bool:
    d = external_dir / "tnic"
    return (d / TNIC_FILE).exists() and (d / LINK_FILE).exists()


def load_tnic(external_dir: Path, n_peers: int) -> dict[int, pd.DataFrame]:
    """Top ``n_peers`` peers of each firm per TNIC year, expressed in panel ids."""
    d = external_dir / "tnic"
    tnic = pd.read_csv(d / TNIC_FILE, sep="\t")
    tnic.columns = [c.lower() for c in tnic.columns]
    link = pd.read_csv(d / LINK_FILE, dtype={"gvkey": int})
    to_id = dict(zip(link["gvkey"], link["id"], strict=False))
    tnic["id"] = tnic["gvkey1"].map(to_id)
    tnic["peer"] = tnic["gvkey2"].map(to_id)
    tnic = tnic.dropna(subset=["id", "peer"]).sort_values(
        ["year", "id", "score"], ascending=[True, True, False]
    )
    tnic = tnic.groupby(["year", "id"], sort=False).head(n_peers)
    return {int(year): grp[["id", "peer", "score"]] for year, grp in tnic.groupby("year")}


def predict_hp(
    df: pd.DataFrame,
    fit: np.ndarray,
    test: np.ndarray,
    multiple: str,
    pairs: pd.DataFrame | None,
    min_peers: int,
) -> pd.DataFrame:
    """Mean multiple of the test firm's TNIC peers in the fitting sample.

    ``pairs`` holds the prior year's similarity pairs (columns id, peer, score), already cut to
    the top peers of each firm; ``None`` when TNIC has no data for that year.
    """
    empty = pd.DataFrame(columns=["yhat", "n_train", "train_mean_y", "train_mean_M"])
    if pairs is None or len(pairs) == 0:
        return empty
    level = np.exp(df["ln" + multiple])
    fit_mult = pd.Series(level[fit].to_numpy(), index=df.loc[fit, "id"].to_numpy())
    test_ids = df.loc[test, "id"]
    p = pairs[pairs["id"].isin(test_ids)].copy()
    p["mult"] = p["peer"].map(fit_mult)
    p = p.dropna(subset=["mult"])
    stats = p.groupby("id")["mult"].agg(["mean", "count"])
    stats = stats[stats["count"] >= min_peers]
    yhat = test_ids.map(stats["mean"]).dropna()
    if yhat.empty:
        return empty
    return pd.DataFrame(
        {
            "yhat": yhat.to_numpy(),
            "n_train": test_ids[yhat.index].map(stats["count"]).to_numpy(),
            "train_mean_y": float(level[fit].mean()),
            "train_mean_M": float(df.loc[fit, "M"].mean()),
        },
        index=yhat.index,
    )
