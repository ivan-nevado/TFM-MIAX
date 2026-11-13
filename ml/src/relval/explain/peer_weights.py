"""GBM predictions as weighted averages of training targets (paper, section 4.5 and appendix A.3).

Notation of the paper: N training firms, v their targets, lambda the learning rate, M trees.

    P_0 = (1/N) 1,  G_0 = P_0
    W_i[f, c] = 1 / (leaf size) if training firms f and c share a leaf of tree i, else 0
    P_i = lambda * W_i (I - G_{i-1}),   G_i = G_{i-1} + P_i
    K = sum_i P_i^T (L_i ./ (1 L_i)),   k = v^T K

All training firms in a leaf share the same row of ``P_i``, so the recursion is carried at leaf
level: ``Q_i[l, :] = lambda / n_l * sum_{c in l} (I - G_{i-1})[c, :]``, ``G_i = G_{i-1} + Q_i[leaf(f)]``
and the weights of a test firm accumulate ``Q_i[leaf(s)]``. Cost: O(M N^2).

Generalisation needed for exactness with LightGBM defaults: when the last split of a leaf is on a
categorical feature (here ``industry``), LightGBM divides the summed residual by ``n_l + cat_l2``
(default 10) instead of ``n_l``. Passing those denominators keeps the identity exact, and the
weights still sum to one because every row of ``I - G`` sums to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relval.models.gbm import GbmFit


def peer_weights_from_leaves(
    leaf_train: np.ndarray,
    leaf_test: np.ndarray,
    learning_rate: float,
    denominators: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Peer-weight matrix ``K`` of shape (N training firms, S test firms).

    ``leaf_train`` (M, N) and ``leaf_test`` (M, S) hold the leaf index of each firm in each tree.
    ``denominators[i][l]`` is the divisor of leaf ``l`` in tree ``i`` (default: its training count).
    """
    leaf_train, leaf_test = np.atleast_2d(leaf_train), np.atleast_2d(leaf_test)
    n_trees, n = leaf_train.shape
    s = leaf_test.shape[1]
    G = np.full((n, n), 1.0 / n)
    K = np.full((s, n), 1.0 / n)
    eye = np.eye(n)
    for i in range(n_trees):
        lt = leaf_train[i]
        n_leaves = int(max(lt.max(), leaf_test[i].max())) + 1
        Q = np.zeros((n_leaves, n))
        np.add.at(Q, lt, eye - G)
        den = (
            np.bincount(lt, minlength=n_leaves).astype(float)
            if denominators is None
            else np.asarray(denominators[i], float)
        )
        den = np.where(den > 0, den, np.inf)
        Q *= (learning_rate / den)[:, None]
        G += Q[lt]
        K += Q[leaf_test[i]]
    return K.T


@dataclass
class PeerWeights:
    weights: np.ndarray  # (N, S)
    targets: np.ndarray  # v: training targets as LightGBM stores them (float32)
    prediction_from_weights: np.ndarray  # v^T K
    prediction_model: np.ndarray
    max_abs_error: float
    max_sum_error: float  # max |sum of weights - 1|
    n_regularised_leaves: int  # leaves whose divisor is n + cat_l2


def leaf_denominators(
    fit: GbmFit, leaf_train: np.ndarray, v: np.ndarray, learning_rate: float, lambda_l2: float, cat_l2: float
) -> tuple[list[np.ndarray], int]:
    """Divisor of each leaf, read from the model: ``n + lambda_l2`` or ``n + lambda_l2 + cat_l2``.

    The in-sample prediction is tracked with the model's own leaf values; for each leaf the two
    candidate values ``lambda * sum(residual) / divisor`` are compared with the stored leaf value.
    """
    n_trees, _ = leaf_train.shape
    init = float(v.mean())
    pred = np.full(len(v), init)
    dens, n_reg = [], 0
    for i in range(n_trees):
        lt = leaf_train[i]
        n_leaves = int(lt.max()) + 1
        counts = np.bincount(lt, minlength=n_leaves).astype(float)
        resid_sum = np.bincount(lt, weights=v - pred, minlength=n_leaves)
        values = np.array([fit.booster.get_leaf_output(i, leaf) for leaf in range(n_leaves)])
        if i == 0:
            values = values - init  # LightGBM folds the initial score into the first tree
        plain = learning_rate * resid_sum / np.where(counts > 0, counts + lambda_l2, np.inf)
        regul = learning_rate * resid_sum / np.where(counts > 0, counts + lambda_l2 + cat_l2, np.inf)
        use_reg = np.abs(values - regul) < np.abs(values - plain)
        n_reg += int(use_reg.sum())
        dens.append(np.where(use_reg, counts + lambda_l2 + cat_l2, counts + lambda_l2))
        pred = pred + values[lt]
    return dens, n_reg


def gbm_peer_weights(
    fit: GbmFit, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    learning_rate: float = 0.1, lambda_l2: float = 0.0, cat_l2: float = 10.0,
) -> PeerWeights:  # fmt: skip
    """Peer weights of a fitted LightGBM model, with the checks of the paper's validation:
    the weights of every test firm sum to one and ``v^T K`` equals the model prediction."""
    v = np.asarray(y_train, dtype=np.float32).astype(float)  # LightGBM stores labels as float32
    leaf_train, leaf_test = fit.leaves(X_train).T, fit.leaves(X_test).T
    dens, n_reg = leaf_denominators(fit, leaf_train, v, learning_rate, lambda_l2, cat_l2)
    K = peer_weights_from_leaves(leaf_train, leaf_test, learning_rate, dens)
    from_weights = v @ K
    model_pred = fit.predict(X_test)
    return PeerWeights(
        weights=K,
        targets=v,
        prediction_from_weights=from_weights,
        prediction_model=model_pred,
        max_abs_error=float(np.max(np.abs(from_weights - model_pred))) if len(model_pred) else 0.0,
        max_sum_error=float(np.max(np.abs(K.sum(axis=0) - 1.0))) if K.size else 0.0,
        n_regularised_leaves=n_reg,
    )
