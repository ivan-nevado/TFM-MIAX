"""Extension E2: train the model of month t on a window of months t-(n-1) .. t.

The paper trains each model on one monthly cross-section. With the LSEG survivor sample a month
has about 700 training rows (paper: 2,265), so extension E2 adds the previous ``n - 1`` months.
Month t is still split into the paper's five blocks and only month t's test block is predicted;
the validation block (early stopping) is also month t's. Past months only add training rows.

Leakage. A firm's multiple barely moves from one month to the next, so if a test firm's own past
rows were in the training set the model could recognise the firm instead of valuing it, and the
error would fall for the wrong reason. ``leakage_guard`` decides how this is prevented:

    exclude_test_firms  past rows of every firm in month t's test or validation block are dropped
                        (blocks stay the paper's monthly random split: paired with the baseline)
    firm_blocks         blocks are assigned per firm for the whole period (``splits.firm_blocks``),
                        so past rows of a test/validation firm are in the same excluded block
    none                no guard. DIAGNOSTIC ONLY: measures how much the leak would flatter the
                        error. Never report it as a result.

Validation firms are excluded as well as test firms: otherwise early stopping would see firms
whose history is in the training set and choose too many trees.

Config (``training_window`` in the YAML; the defaults reproduce the paper exactly):

    months: 1                 # window length in months (1 = paper)
    leakage_guard: exclude_test_firms
    lag_weight_decay: 1.0     # weight of a row from ``lag`` months ago = decay ** lag
    align_history_level: false  # shift each past month's targets so its median equals the median
                                # of month t's training rows (removes market-wide level moves)

The first ``months - 1`` months of the period have a shorter window (the panel starts there).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GUARDS = ("exclude_test_firms", "firm_blocks", "none")
DEFAULTS = {"months": 1, "leakage_guard": "exclude_test_firms", "lag_weight_decay": 1.0,
            "align_history_level": False}  # fmt: skip


def window_cfg(cfg: dict) -> dict:
    """``training_window`` block merged over the defaults and validated."""
    out = dict(DEFAULTS)
    out.update(cfg.get("training_window") or {})
    out["months"] = int(out["months"])
    out["lag_weight_decay"] = float(out["lag_weight_decay"])
    if out["months"] < 1:
        raise ValueError("training_window.months must be at least 1")
    if out["leakage_guard"] not in GUARDS:
        raise ValueError(f"training_window.leakage_guard must be one of {GUARDS}")
    if not 0.0 < out["lag_weight_decay"] <= 1.0:
        raise ValueError("training_window.lag_weight_decay must be in (0, 1]")
    return out


def uses_window(cfg: dict) -> bool:
    return window_cfg(cfg)["months"] > 1


def uses_firm_blocks(cfg: dict) -> bool:
    return window_cfg(cfg)["leakage_guard"] == "firm_blocks"


def check_spec(spec) -> None:
    """The window is defined for GBMs on the 97 inputs with the 60:20:20 split only."""
    ok = spec.kind == "gbm" and spec.features == "ml97" and spec.split == "60_20_20"
    if not ok or spec.sample not in ("all", "listed", "nm"):
        raise ValueError(
            f"training_window.months > 1 is not defined for model {spec.name!r} "
            "(only GBMs on the 97 inputs with the 60:20:20 split); train the base group only"
        )


def month_number(dates) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return (d.year * 12 + d.month).to_numpy()


def history_columns(columns) -> list[str]:
    """Columns of the sample that the GBM needs for past rows (keeps the worker payload small)."""
    from relval.features.registry import FEATURES
    from relval.targets.multiples import LOG_TARGETS

    keep = {"id", "date", "origen", "M", *FEATURES, *LOG_TARGETS}
    return [c for c in columns if c in keep or c.startswith("CAPMbeta_")]


def history_for(by_date: dict, date: pd.Timestamp, months: int) -> pd.DataFrame | None:
    """Rows of the ``months - 1`` months before ``date`` (``by_date``: month-end -> frame)."""
    if months <= 1:
        return None
    frames = [
        by_date[d] for d in (date - pd.offsets.MonthEnd(lag) for lag in range(1, months)) if d in by_date
    ]
    return pd.concat(frames, ignore_index=True) if frames else None


def history_mask(
    hist: pd.DataFrame, month: pd.DataFrame, excluded_blocks: set[int], guard: str,
    firm_map: pd.Series | None = None,
) -> np.ndarray:  # fmt: skip
    """Past rows allowed to train the model whose test/validation blocks are ``excluded_blocks``.

    ``month`` is month t with its ``block`` column."""
    if guard == "none":
        return np.ones(len(hist), dtype=bool)
    if guard == "exclude_test_firms":
        banned = month.loc[month["block"].isin(excluded_blocks), "id"]
        return ~hist["id"].isin(set(banned)).to_numpy()
    if guard == "firm_blocks":
        if firm_map is None:
            raise ValueError("leakage_guard 'firm_blocks' needs the firm block map")
        blocks = hist["id"].astype(str).map(firm_map)
        return (~blocks.isin(excluded_blocks)).to_numpy()
    raise ValueError(f"unknown leakage_guard {guard!r}")


def lag_weights(hist_dates, date: pd.Timestamp, decay: float) -> np.ndarray:
    """``decay ** lag`` with lag in months (1 for the month before t)."""
    lags = month_number([date])[0] - month_number(hist_dates)
    return np.power(float(decay), lags.astype(float))


def align_levels(y_hist: np.ndarray, hist_dates, reference_median: float) -> np.ndarray:
    """Shift every past month's targets so that their median equals ``reference_median`` (the
    median of month t's training targets: no test information is used)."""
    y = np.asarray(y_hist, dtype=float).copy()
    keys = month_number(hist_dates)
    for key in np.unique(keys):
        sel = keys == key
        y[sel] += reference_median - np.nanmedian(y[sel])
    return y
