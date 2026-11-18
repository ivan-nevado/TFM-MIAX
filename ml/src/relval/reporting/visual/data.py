"""Loading of everything the visual report reads: panel, firms, predictions, SHAP, models, peers
and the tables written by ``relval report``. Nothing here trains a model."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from relval.io import read_parquet, read_partitioned
from relval.models.runner import load_sample
from relval.models.specs import SPECS

PRED_COLS = [
    "id",
    "date",
    "block",
    "y",
    "yhat",
    "M",
    "Mhat",
    "eps",
    "n_train",
    "train_mean_y",
    "best_iteration",
]
FF12_NAMES = {
    1: "NoDur", 2: "Durbl", 3: "Manuf", 4: "Enrgy", 5: "Chems", 6: "BusEq",
    7: "Telcm", 8: "Utils", 9: "Shops", 10: "Hlth", 11: "Money", 12: "Other",
}  # fmt: skip
FF12_ES = {
    "NoDur": "Consumo no duradero", "Durbl": "Consumo duradero", "Manuf": "Manufacturas", "Enrgy": "Energía",
    "Chems": "Químicas", "BusEq": "Tecnología", "Telcm": "Telecomunicaciones", "Utils": "Utilities",
    "Shops": "Comercio", "Hlth": "Salud", "Money": "Finanzas", "Other": "Otros",
}  # fmt: skip


class VisualData:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        self._preds: dict[tuple, pd.DataFrame] = {}

    # ---- panel and firms -------------------------------------------------------------------
    @cached_property
    def sample(self) -> pd.DataFrame:
        """Modelling sample (Table 1, last step), with sector names."""
        self.log("[visual] loading the modelling sample")
        s = load_sample(self.cfg)
        s["ff12_name"] = s["ff12"].map(FF12_NAMES)
        firms = self.firms.set_index("id")
        s["trbc"] = s["id"].map(firms["sector"])
        s["ff49_name"] = s["id"].map(firms["ff49_name"])
        return s

    @cached_property
    def panel_counts(self) -> pd.DataFrame:
        """Firms per month at the two panel stages (positive targets, final sample)."""
        p = read_parquet(self.cfg.panel_path(), columns=["id", "date", "in_sample"])
        g = p.groupby("date").agg(targets=("id", "size"), sample=("in_sample", "sum"))
        return g.reset_index()

    @cached_property
    def firms(self) -> pd.DataFrame:
        return read_parquet(self.cfg.path("processed") / f"firms_{self.cfg['source']}.parquet")

    # ---- predictions ------------------------------------------------------------------------
    def models_available(self) -> list[str]:
        base = self.cfg.predictions_dir()
        return sorted(
            p.name.split("=", 1)[1] for p in base.glob("model=*") if p.name.split("=", 1)[1] in SPECS
        )

    def seeds_available(self, model: str) -> list[int]:
        d = self.cfg.predictions_dir(model)
        return sorted(int(p.name.split("=", 1)[1]) for p in d.glob("seed=*"))

    def preds(self, model: str, seed: int | None = None, columns: list[str] | None = None) -> pd.DataFrame:
        seed = self.cfg.headline_seed if seed is None else seed
        key = (model, seed, tuple(columns or PRED_COLS))
        if key not in self._preds:
            d = self.cfg.predictions_dir(model, seed)
            df = read_partitioned(d, columns=list(columns or PRED_COLS)) if d.exists() else pd.DataFrame()
            if len(df) and "date" in df:
                df["date"] = pd.to_datetime(df["date"])
            self._preds[key] = df
        return self._preds[key]

    def drop_cache(self) -> None:
        self._preds.clear()

    # ---- explainability ---------------------------------------------------------------------
    def shap(self, model: str) -> pd.DataFrame:
        d = self.cfg.artifacts_dir("shap", model, self.cfg.headline_seed)
        return read_partitioned(d) if d.exists() else pd.DataFrame()

    def saved_models(self, model: str) -> pd.DataFrame:
        d = self.cfg.artifacts_dir("models", model, self.cfg.headline_seed)
        return read_partitioned(d) if d.exists() else pd.DataFrame()

    def peers(self, kind: str, target: str, month: str = "201912") -> pd.DataFrame:
        f = self.cfg.artifacts_dir("peers") / f"peers_{kind}_{target}_{month}.parquet"
        return read_parquet(f) if f.exists() else pd.DataFrame()

    def peer_validation(self) -> pd.DataFrame:
        f = self.cfg.artifacts_dir("peers") / "validation.parquet"
        return read_parquet(f) if f.exists() else pd.DataFrame()

    def unavailable(self) -> dict:
        f = self.cfg.artifacts_dir("status") / "unavailable.json"
        return json.loads(f.read_text()) if f.exists() else {}

    # ---- tables of `relval report` ---------------------------------------------------------
    def table(self, name: str) -> pd.DataFrame:
        f = self.cfg.reports_dir("tables") / f"{name}.csv"
        return pd.read_csv(f) if f.exists() else pd.DataFrame()

    def manifests(self) -> list[dict]:
        d = self.cfg.artifacts_dir("manifests")
        out = []
        for f in sorted(Path(d).glob("*.json")):
            try:
                out.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                continue
        return out


def monthly_median(df: pd.DataFrame, value: str = "abs_eps") -> pd.Series:
    d = df.assign(abs_eps=df["eps"].abs()) if value == "abs_eps" else df
    return d.groupby("date")[value].median()


def finite(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]
