"""LASSO on the 97 inputs (paper, Table 4, "LASSO : ML vars (nm)").

Only complete cases (every variable present). Industry enters as dummies; inputs are standardised
with the training-block mean and standard deviation. Paper silent on the penalty: it is chosen on
the validation block from the regularisation path fitted on the training block.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path


def design_matrix(df: pd.DataFrame, features: list[str], train: np.ndarray) -> np.ndarray:
    """Standardised numeric inputs plus industry dummies (categories seen in the training block)."""
    numeric = [f for f in features if f != "industry"]
    X = df[numeric].to_numpy(dtype=float)
    mean, sd = X[train].mean(axis=0), X[train].std(axis=0)
    keep = sd > 0
    Z = (X[:, keep] - mean[keep]) / sd[keep]
    if "industry" in features:
        cats = np.unique(df.loc[train, "industry"].to_numpy())
        dummies = (df["industry"].to_numpy()[:, None] == cats[None, :]).astype(float)
        if dummies.shape[1] > 1:
            dummies = dummies[:, 1:]  # drop one category (intercept)
        Z = np.column_stack([Z, dummies])
    return Z


def fit_predict_lasso(
    Z: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray, n_alphas: int
) -> tuple[np.ndarray, float]:
    """Predictions for the test rows and the chosen penalty."""
    y_mean = y[train].mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alphas, coefs, _ = lasso_path(Z[train], y[train] - y_mean, n_alphas=n_alphas, max_iter=5000)
    val_pred = Z[val] @ coefs + y_mean  # (n_val, n_alphas)
    mse = ((val_pred - y[val][:, None]) ** 2).mean(axis=0)
    best = int(np.argmin(mse))
    return Z[test] @ coefs[:, best] + y_mean, float(alphas[best])
