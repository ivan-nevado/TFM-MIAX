"""Gradient boosting machine of the paper: LightGBM with default hyperparameters.

The only tuned quantity is the number of trees, chosen by early stopping on the validation set
(paper, footnote 7). Paper silent, defaults used: patience 50 rounds, cap 10,000 rounds, L2 on the
validation set, prediction with ``best_iteration``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

LGB_MAJOR = int(lgb.__version__.split(".")[0])


@dataclass
class GbmFit:
    booster: lgb.Booster
    best_iteration: int
    feature_names: list[str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X, num_iteration=self.best_iteration)

    def shap(self, X: np.ndarray) -> np.ndarray:
        """Exact TreeSHAP values, shape (n, n_features + 1); the last column is the base value."""
        return self.booster.predict(X, num_iteration=self.best_iteration, pred_contrib=True)

    def leaves(self, X: np.ndarray) -> np.ndarray:
        """Leaf index of each observation in each tree, shape (n, best_iteration)."""
        out = self.booster.predict(X, num_iteration=self.best_iteration, pred_leaf=True)
        return np.asarray(out).reshape(len(X), -1)

    def to_string(self) -> str:
        return self.booster.model_to_string(num_iteration=self.best_iteration)


def lgb_params(gbm_cfg: dict, seed: int) -> dict:
    params = dict(gbm_cfg["params"])
    params.update({"seed": int(seed), "verbose": -1})
    if gbm_cfg.get("deterministic", True):
        params.update({"deterministic": True, "num_threads": 1, "force_row_wise": True})
    return params


def fit_gbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    feature_names: list[str],
    gbm_cfg: dict,
    seed: int,
    categorical: list[str] | None = None,
    fixed_rounds: int | None = None,
) -> GbmFit:
    """Train one model. With a validation set the number of trees is chosen by early stopping;
    with ``fixed_rounds`` (80:20 split) exactly that many trees are built."""
    params = lgb_params(gbm_cfg, seed)
    cat = [c for c in (categorical or []) if c in feature_names]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dtrain = lgb.Dataset(
            X_train, label=y_train, feature_name=list(feature_names),
            categorical_feature=cat or "auto", free_raw_data=False, params={"verbose": -1},
        )  # fmt: skip
        kwargs: dict = {}
        if LGB_MAJOR < 4:
            kwargs["verbose_eval"] = False
        if fixed_rounds is not None:
            booster = lgb.train(params, dtrain, num_boost_round=max(int(fixed_rounds), 1), **kwargs)
            best = booster.current_iteration()
        else:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, params={"verbose": -1})
            booster = lgb.train(
                params, dtrain,
                num_boost_round=int(gbm_cfg["num_boost_round"]),
                valid_sets=[dval], valid_names=["validation"],
                callbacks=[lgb.early_stopping(int(gbm_cfg["early_stopping_rounds"]), verbose=False)],
                **kwargs,
            )  # fmt: skip
            best = booster.best_iteration or booster.current_iteration()
    return GbmFit(booster, int(best), list(feature_names))


def load_gbm(model_text: str, best_iteration: int, feature_names: list[str]) -> GbmFit:
    return GbmFit(lgb.Booster(model_str=model_text), int(best_iteration), list(feature_names))
