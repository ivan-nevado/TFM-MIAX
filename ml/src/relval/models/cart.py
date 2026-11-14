"""CART: a single regression tree (paper, online appendix IA.6.2).

Paper silent on the settings; the team's plan is used: maximum depth chosen on the validation
block from ``benchmarks.cart_depths``, missing inputs replaced by the training-block median.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor


def impute_median(X: np.ndarray, train: np.ndarray) -> np.ndarray:
    med = np.nanmedian(np.where(np.isfinite(X[train]), X[train], np.nan), axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(X), med[None, :], X)


def fit_predict_cart(
    X: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    depths: list[int],
    seed: int,
) -> tuple[np.ndarray, int]:
    """Predictions for the test rows and the chosen depth."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # all-NaN columns in small months
        Xi = impute_median(X, train)
    best_depth, best_mse, best_tree = depths[0], np.inf, None
    for depth in depths:
        tree = DecisionTreeRegressor(max_depth=int(depth), random_state=int(seed)).fit(Xi[train], y[train])
        mse = float(((tree.predict(Xi[val]) - y[val]) ** 2).mean())
        if mse < best_mse:
            best_depth, best_mse, best_tree = int(depth), mse, tree
    return best_tree.predict(Xi[test]), best_depth
